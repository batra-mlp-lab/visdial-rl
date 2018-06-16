import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import utilities as utils


class Encoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 useIm,
                 imgEmbedSize,
                 imgFeatureSize,
                 numRounds,
                 isAnswerer,
                 dropout=0,
                 startToken=None,
                 endToken=None,
                 **kwargs):
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useIm:
            self.useIm = useIm if useIm != True else 'early'
        else:
            self.useIm = False
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken

        # modules
        self.wordEmbed = nn.Embedding(
            self.vocabSize, self.embedSize, padding_idx=0)

        # question encoder
        # image fuses early with words
        if self.useIm == 'early':
            quesInputSize = self.embedSize + self.imgEmbedSize
            dialogInputSize = 2 * self.rnnHiddenSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.useIm == 'late':
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize + self.imgEmbedSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.isAnswerer:
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize
        else:
            dialogInputSize = self.rnnHiddenSize
        if self.isAnswerer:
            self.quesRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=0)

        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0)

        # dialog rnn
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)

    def reset(self):
        # batchSize is inferred from input
        self.batchSize = 0

        # Input data
        self.image = None
        self.imageEmbed = None

        self.captionTokens = None
        self.captionEmbed = None
        self.captionLens = None

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        # Hidden embeddings
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return (Variable(h), Variable(c))

    def observe(self,
                round,
                image=None,
                caption=None,
                ques=None,
                ans=None,
                captionLens=None,
                quesLens=None,
                ansLens=None):
        '''
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''
        if image is not None:
            assert round == -1
            self.image = image
            self.imageEmbed = None
            self.batchSize = len(self.image)
        if caption is not None:
            assert round == -1
            assert captionLens is not None, "Caption lengths required!"
            caption, captionLens = self.processSequence(caption, captionLens)
            self.captionTokens = caption
            self.captionLens = captionLens
            self.batchSize = len(self.captionTokens)
        if ques is not None:
            assert round == len(self.questionEmbeds)
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
        if ans is not None:
            assert round == len(self.answerEmbeds)
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)

    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        '''
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        '''
        # Embed image, occurs once per dialog
        if self.isAnswerer and self.imageEmbed is None:
            self.imageEmbed = self.imgNet(self.imgEmbedDropout(self.image))
        # Embed caption, occurs once per dialog
        if self.captionEmbed is None:
            self.captionEmbed = self.wordEmbed(self.captionTokens)
        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(
                self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        '''Embed facts i.e. caption and round 0 or question-answer pair otherwise'''
        # Caption
        if factIdx == 0:
            seq, seqLens = self.captionEmbed, self.captionLens
            factEmbed, states = utils.dynamicRNN(
                self.factRNN, seq, seqLens, returnStates=True)
        # QA pairs
        elif factIdx > 0:
            quesTokens, quesLens = \
                self.questionTokens[factIdx - 1], self.questionLens[factIdx - 1]
            ansTokens, ansLens = \
                self.answerTokens[factIdx - 1], self.answerLengths[factIdx - 1]

            qaTokens = utils.concatPaddedSequences(
                quesTokens, quesLens, ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            qaEmbed, states = utils.dynamicRNN(
                self.factRNN, qa, qaLens, returnStates=True)
            factEmbed = qaEmbed
        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useIm == 'early':
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx):
        currIns = [self.factEmbeds[histIdx][0]]
        if self.isAnswerer:
            currIns.append(self.questionRNNStates[histIdx][0])
        if self.useIm == 'late':
            currIns.append(self.imageEmbed)
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        self.dialogHiddens.append(hNew)

    def forward(self):
        '''
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''

        # Lazily embed input Image, Captions, Questions and Answers
        self.embedInputDialog()

        if self.isAnswerer:
            # For A-Bot, current round is the number of facts present,
            # which is number of questions observed - 1 (as opposed
            # to len(self.answerEmbeds), which may be inaccurate as
            round = len(self.questionEmbeds) - 1
        else:
            # For Q-Bot, current round is the number of facts present,
            # which is same as the number of answers observed
            round = len(self.answerEmbeds)

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # Infer any missing facts
        while len(self.factEmbeds) <= round:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        # Embed any un-embedded questions (A-Bot only)
        if self.isAnswerer:
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                self.embedQuestion(qIdx)

        # Concat facts and/or questions (i.e. history) for input to dialogRNN
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx)

        # Forward dialogRNN one step
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]

        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (quesRNN)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (quesRNN)
              Layer 1 : DialogRNN hidden state (dialogRNN)

        If Q-Bot:
            C_link == Fact encoding RNN cell state (factRNN)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (factRNN)
                Layer 1 : DialogRNN hidden state (dialogRNN)
        '''
        if self.isAnswerer:
            quesRNNstates = self.questionRNNStates[-1][1]  # Latest quesRNN states
            C_link = quesRNNstates[1]
            H_link = quesRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        else:
            factRNNstates = self.factEmbeds[-1][1]  # Latest factRNN states
            C_link = factRNNstates[1]
            H_link = factRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)

        return H_link, C_link
