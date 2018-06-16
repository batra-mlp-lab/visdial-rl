import os
import json
import h5py
import numpy as np
import torch
from six import iteritems
from six.moves import range
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


class VisDialDataset(Dataset):
    def __init__(self, params, subsets):
        '''
            Initialize the dataset with splits given by 'subsets', where
            subsets is taken from ['train', 'val', 'test']

            Notation:
                'dtype' is a split taking values from ['train', 'val', 'test']
                'stype' is a sqeuence type from ['ques', 'ans']
        '''

        # By default, load Q-Bot, A-Bot and dialog options for A-Bot
        self.useQuestion = True
        self.useAnswer = True
        self.useOptions = True
        self.useHistory = True
        self.useIm = True

        # Absorb parameters
        for key, value in iteritems(params):
            setattr(self, key, value)
        self.subsets = tuple(subsets)
        self.numRounds = params['numRounds']

        print('\nDataloader loading json file: ' + self.inputJson)
        with open(self.inputJson, 'r') as fileId:
            info = json.load(fileId)
            # Absorb values
            for key, value in iteritems(info):
                setattr(self, key, value)

        wordCount = len(self.word2ind)
        # Add <START> and <END> to vocabulary
        self.word2ind['<START>'] = wordCount + 1
        self.word2ind['<END>'] = wordCount + 2
        self.startToken = self.word2ind['<START>']
        self.endToken = self.word2ind['<END>']
        # Padding token is at index 0
        self.vocabSize = wordCount + 3
        print('Vocab size with <START>, <END>: %d' % self.vocabSize)

        # Construct the reverse map
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        # Read questions, answers and options
        print('Dataloader loading h5 file: ' + self.inputQues)
        quesFile = h5py.File(self.inputQues, 'r')

        if self.useIm:
            # Read images
            print('Dataloader loading h5 file: ' + self.inputImg)
            imgFile = h5py.File(self.inputImg, 'r')

        # Number of data points in each split (train/val/test)
        self.numDataPoints = {}
        self.data = {}

        # map from load to save labels
        ioMap = {
            'ques_%s': '%s_ques',
            'ques_length_%s': '%s_ques_len',
            'ans_%s': '%s_ans',
            'ans_length_%s': '%s_ans_len',
            'ans_index_%s': '%s_ans_ind',
            'img_pos_%s': '%s_img_pos',
            'opt_%s': '%s_opt',
            'opt_length_%s': '%s_opt_len',
            'opt_list_%s': '%s_opt_list'
        }

        # Processing every split in subsets
        for dtype in subsets:  # dtype is in [train, val, test]
            print("\nProcessing split [%s]..." % dtype)
            if ('ques_%s' % dtype) not in quesFile:
                self.useQuestion = False
            if ('ans_%s' % dtype) not in quesFile:
                self.useAnswer = False
            if ('opt_%s' % dtype) not in quesFile:
                self.useOptions = False
            # read the question, answer, option related information
            for loadLabel, saveLabel in iteritems(ioMap):
                if loadLabel % dtype not in quesFile:
                    continue
                dataMat = np.array(quesFile[loadLabel % dtype], dtype='int64')
                self.data[saveLabel % dtype] = torch.from_numpy(dataMat)

            # Read image features, if needed
            if self.useIm:
                print('Reading image features...')
                imgFeats = np.array(imgFile['images_' + dtype])

                if not self.imgNorm:
                    continue
                # normalize, if needed
                print('Normalizing image features..')
                imgFeats = normalize(imgFeats, axis=1, norm='l2')

                # save img features
                self.data['%s_img_fv' % dtype] = torch.FloatTensor(imgFeats)
                # Visdial
                if hasattr(self, 'unique_img_train') and params['cocoDir']:
                    coco_dir = params['cocoDir']
                    with open(params['cocoInfo'], 'r') as f:
                        coco_info = json.load(f)
                    id_to_fname = {
                        im['id']: im['file_path']
                        for im in coco_info['images']
                    }
                    cocoids = getattr(self, 'unique_img_%s'%dtype)
                    if '.jpg' not in cocoids[0]:
                        img_fnames = [
                            os.path.join(coco_dir, id_to_fname[int(cocoid)])
                            for cocoid in cocoids
                        ]
                    else:
                        img_fnames = cocoids
                    self.data['%s_img_fnames' % dtype] = img_fnames

            # read the history, if needed
            if self.useHistory:
                captionMap = {
                    'cap_%s': '%s_cap',
                    'cap_length_%s': '%s_cap_len'
                }
                for loadLabel, saveLabel in iteritems(captionMap):
                    mat = np.array(quesFile[loadLabel % dtype], dtype='int32')
                    self.data[saveLabel % dtype] = torch.from_numpy(mat)

            # Number of data points
            self.numDataPoints[dtype] = self.data[dtype + '_cap'].size(0)

        # Prepare dataset for training
        for dtype in subsets:
            print("\nSequence processing for [%s]..." % dtype)
            self.prepareDataset(dtype)
        print("")

        # Default pytorch loader dtype is set to train
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    #----------------------------------------------------------------------------
    # Dataset preprocessing
    #----------------------------------------------------------------------------

    def prepareDataset(self, dtype):
        if self.useHistory:
            self.processCaption(dtype)

        # prefix/postfix with <START> and <END>
        if self.useOptions:
            self.processOptions(dtype)
            # options are 1-indexed, changed to 0-indexed
            self.data[dtype + '_opt'] -= 1

        # process answers and questions
        if self.useAnswer:
            self.processSequence(dtype, stype='ans')
            # 1 indexed to 0 indexed
            self.data[dtype + '_ans_ind'] -= 1
        if self.useQuestion:
            self.processSequence(dtype, stype='ques')

    def processSequence(self, dtype, stype='ans'):
        '''
        Add <START> and <END> token to answers or questions.
        Arguments:
            'dtype'    : Split to use among ['train', 'val', 'test']
            'sentType' : Sequence type, either 'ques' or 'ans'
        '''
        assert stype in ['ques', 'ans']
        prefix = dtype + "_" + stype

        seq = self.data[prefix]
        seqLen = self.data[prefix + '_len']

        numConvs, numRounds, maxAnsLen = seq.size()
        newSize = torch.Size([numConvs, numRounds, maxAnsLen + 2])
        sequence = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        sequence[:, :, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for thId in range(numConvs):
            for rId in range(numRounds):
                length = seqLen[thId, rId]
                if length == 0:
                    print('Warning: Skipping empty %s sequence at (%d, %d)'\
                          %(stype, thId, rId))
                    continue

                sequence[thId, rId, 1:length + 1] = seq[thId, rId, :length]
                sequence[thId, rId, length + 1] = endTokenId

        # Sequence length is number of tokens + 1
        self.data[prefix + "_len"] = seqLen + 1
        self.data[prefix] = sequence

    def processCaption(self, dtype):
        '''
        Add <START> and <END> token to caption.
        Arguments:
            'dtype'    : Split to use among ['train', 'val', 'test']
        '''
        prefix = dtype + '_cap'

        seq = self.data[prefix]
        seqLen = self.data[prefix + '_len']

        numConvs, maxCapLen = seq.size()
        newSize = torch.Size([numConvs, maxCapLen + 2])
        sequence = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        sequence[:, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for thId in range(numConvs):
            length = seqLen[thId]
            if length == 0:
                print('Warning: Skipping empty %s sequence at (%d)' % (stype,
                                                                       thId))
                continue

            sequence[thId, 1:length + 1] = seq[thId, :length]
            sequence[thId, length + 1] = endTokenId

        # Sequence length is number of tokens + 1
        self.data[prefix + "_len"] = seqLen + 1
        self.data[prefix] = sequence

    def processOptions(self, dtype):
        ans = self.data[dtype + '_opt_list']
        ansLen = self.data[dtype + '_opt_len']

        ansListLen, maxAnsLen = ans.size()

        newSize = torch.Size([ansListLen, maxAnsLen + 2])
        options = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        options[:, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for ansId in range(ansListLen):
            length = ansLen[ansId]
            if length == 0:
                print('Warning: Skipping empty option answer list at (%d)'\
                        %ansId)
                continue

            options[ansId, 1:length + 1] = ans[ansId, :length]
            options[ansId, length + 1] = endTokenId

        self.data[dtype + '_opt_len'] = ansLen + 1
        self.data[dtype + '_opt_seq'] = options

    #----------------------------------------------------------------------------
    # Dataset helper functions for PyTorch's datalaoder
    #----------------------------------------------------------------------------

    def __len__(self):
        # Assert that loader_dtype is in subsets ['train', 'val', 'test']
        return self.numDataPoints[self._split]

    def __getitem__(self, idx):
        item = self.getIndexItem(self._split, idx)
        return item

    def collate_fn(self, batch):
        out = {}

        mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}
        for key in mergedBatch:
            if key == 'img_fname' or key == 'index':
                out[key] = mergedBatch[key]
            elif key == 'cap_len':
                # 'cap_lens' are single integers, need special treatment
                out[key] = torch.LongTensor(mergedBatch[key])
            else:
                out[key] = torch.stack(mergedBatch[key], 0)

        # Dynamic shaping of padded batch
        if 'ques' in out.keys():
            quesLen = out['ques_len'] + 1
            out['ques'] = out['ques'][:, :, :torch.max(quesLen)].contiguous()

        if 'ans' in out.keys():
            ansLen = out['ans_len'] + 1
            out['ans'] = out['ans'][:, :, :torch.max(ansLen)].contiguous()

        if 'cap' in out.keys():
            capLen = out['cap_len'] + 1
            out['cap'] = out['cap'][:, :torch.max(capLen)].contiguous()

        if 'opt' in out.keys():
            optLen = out['opt_len'] + 1
            out['opt'] = out['opt'][:, :, :, :torch.max(optLen) + 2].contiguous()

        return out

    #----------------------------------------------------------------------------
    # Dataset indexing
    #----------------------------------------------------------------------------

    def getIndexItem(self, dtype, idx):
        item = {'index': idx}

        # get question
        if self.useQuestion:
            ques = self.data[dtype + '_ques'][idx]
            quesLen = self.data[dtype + '_ques_len'][idx]
            item['ques'] = ques
            item['ques_len'] = quesLen

        # get answer
        if self.useAnswer:
            ans = self.data[dtype + '_ans'][idx]
            ansLen = self.data[dtype + '_ans_len'][idx]
            item['ans_len'] = ansLen
            item['ans'] = ans

        # get caption
        if self.useHistory:
            cap = self.data[dtype + '_cap'][idx]
            capLen = self.data[dtype + '_cap_len'][idx]
            item['cap'] = cap
            item['cap_len'] = capLen

        if self.useOptions:
            optInds = self.data[dtype + '_opt'][idx]
            ansId = self.data[dtype + '_ans_ind'][idx]

            optSize = list(optInds.size())
            newSize = torch.Size(optSize + [-1])

            indVector = optInds.view(-1)
            optLens = self.data[dtype + '_opt_len'].index_select(0, indVector)
            optLens = optLens.view(optSize)

            opts = self.data[dtype + '_opt_seq'].index_select(0, indVector)

            item['opt'] = opts.view(newSize)
            item['opt_len'] = optLens
            item['ans_id'] = ansId

        # if image needed
        if self.useIm:
            item['img_feat'] = self.data[dtype + '_img_fv'][idx]
            # item['img_fname'] = self.data[dtype + '_img_fnames'][idx]
            if dtype + '_img_labels' in self.data:
                item['img_label'] = self.data[dtype + '_img_labels'][idx]

        return item
