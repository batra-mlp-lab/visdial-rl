import os
import argparse
from six import iteritems
from itertools import product
from time import gmtime, strftime

def readCommandLine(argv=None):
    parser = argparse.ArgumentParser(description='Train and Test the Visual Dialog model')

    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-inputImg', default='data/visdial/data_img.h5',
                            help='HDF5 file with image features')
    parser.add_argument('-inputQues', default='data/visdial/chat_processed_data.h5',
                            help='HDF5 file with preprocessed questions')
    parser.add_argument('-inputJson', default='data/visdial/chat_processed_params.json',
                            help='JSON file with info and vocab')
    parser.add_argument('-cocoDir', default='',
                            help='Directory for coco images, optional')
    parser.add_argument('-cocoInfo', default='',
                            help='JSON file with coco split information')

    #-------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-verbose', type=int, default=1,
                            help='Level of verbosity (default 1 prints some info)',
                            choices=[1, 2])
    parser.add_argument('-savePath', default='checkpoints/',
                            help='Path to save checkpoints')
    parser.add_argument('-saveName', default='',
                            help='Name of save directory within savePath')
    parser.add_argument('-startFrom', type=str, default='',
                            help='Copy weights from model at this path')
    parser.add_argument('-qstartFrom', type=str, default='',
                            help='Copy weights from qbot model at this path')
    parser.add_argument('-continue', action='store_true',
                            help='Continue training from last epoch')
    parser.add_argument('-enableVisdom', type=int, default=0,
                            help='Flag for enabling visdom logging')
    parser.add_argument('-visdomEnv', type=str, default='',
                            help='Name of visdom environment for plotting')
    parser.add_argument('-visdomServer', type=str, default='127.0.0.1',
                            help='Address of visdom server instance')
    parser.add_argument('-visdomServerPort', type=int, default=8893,
                            help='Port of visdom server instance')

    #-------------------------------------------------------------------------
    # Model params for both a-bot and q-bot
    parser.add_argument('-randomSeed', default=32, type=int,
                            help='Seed for random number generators')
    parser.add_argument('-imgEmbedSize', default=300, type=int,
                            help='Size of the multimodal embedding')
    parser.add_argument('-imgFeatureSize', default=4096, type=int,
                            help='Size of the image feature')
    parser.add_argument('-embedSize', default=300, type=int,
                            help='Size of input word embeddings')
    parser.add_argument('-rnnHiddenSize', default=512, type=int,
                            help='Size of the LSTM state')
    parser.add_argument('-numLayers', default=2, type=int,
                            help='Number of layers in LSTM')
    parser.add_argument('-imgNorm', default=1, type=int,
                            help='Normalize the image feature. 1=yes, 0=no')

    # A-Bot encoder + decoder
    parser.add_argument('-encoder', default='hre-ques-lateim-hist',
                            help='Name of the encoder to use',
                            choices=['hre-ques-lateim-hist'])
    parser.add_argument('-decoder', default='gen',
                            help='Name of the decoder to use (gen)',
                            choices=['gen'])
    # Q-bot encoder + decoder
    parser.add_argument('-qencoder', default='hre-ques-lateim-hist',
                            help='Name of the encoder to use',
                            choices=['hre-ques-lateim-hist'])
    parser.add_argument('-qdecoder', default='gen',
                            help='Name of the decoder to use (only gen supported now)',
                            choices=['gen'])

    #-------------------------------------------------------------------------
    # Optimization / training params
    parser.add_argument('-trainMode', default='sl-abot',
                            help='What should train.py do?',
                            choices=['sl-abot', 'sl-qbot', 'rl-full-QAf'])
    parser.add_argument('-numRounds', default=10, type=int,
                            help='Number of rounds of dialog (max 10)')
    parser.add_argument('-batchSize', default=20, type=int,
                            help='Batch size (number of threads) '
                                    '(Adjust base on GPU memory)')
    parser.add_argument('-learningRate', default=1e-3, type=float,
                            help='Learning rate')
    parser.add_argument('-minLRate', default=5e-5, type=float,
                            help='Minimum learning rate')
    parser.add_argument('-dropout', default=0.0, type=float, help='Dropout')
    parser.add_argument('-numEpochs', default=65, type=int, help='Epochs')
    parser.add_argument('-lrDecayRate', default=0.9997592083, type=float,
                            help='Decay for learning rate')
    parser.add_argument('-CELossCoeff', default=200, type=float,
                            help='Coefficient for cross entropy loss')
    parser.add_argument('-featLossCoeff', default=1000, type=float,
                            help='Coefficient for feature regression loss')
    parser.add_argument('-useCurriculum', default=1, type=int,
                            help='Use curriculum or for RL training (1) or not (0)')
    parser.add_argument('-freezeQFeatNet', default=0, type=int,
                            help='Freeze weights of Q-bot feature network')
    parser.add_argument('-rlAbotReward', default=1, type=int,
                            help='Choose whether RL reward goes to A-Bot')

    # Other training environmnet settings
    parser.add_argument('-useGPU', action='store_true', help='Use GPU or CPU')
    parser.add_argument('-numWorkers', default=2, type=int,
                            help='Number of worker threads in dataloader')

    #-------------------------------------------------------------------------
    # Evaluation params
    parser.add_argument('-beamSize', default=1, type=int,
                            help='Beam width for beam-search sampling')
    parser.add_argument('-evalModeList', default=[], nargs='+',
                            help='What task should the evaluator perform?',
                            choices=['ABotRank', 'QBotRank', 'QABotsRank', 'dialog'])
    parser.add_argument('-evalSplit', default='val',
                            choices=['train', 'val', 'test'])
    parser.add_argument('-evalTitle', default='eval',
                            help='If generating a plot, include this in the title')
    #-------------------------------------------------------------------------

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg))

    if parsed['saveName']:
        # Custom save file path
        parsed['savePath'] = os.path.join(parsed['savePath'],
                                          parsed['saveName'])
    else:
        # Standard save path with time stamp
        import random
        timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
        parsed['savePath'] = os.path.join(parsed['savePath'], timeStamp)
        parsed['savePath'] += '_{:0>6d}'.format(random.randint(0, 10e6))

    # check if history is needed
    parsed['useHistory'] = True if 'hist' in parsed['encoder'] else False

    # check if image is needed
    if 'lateim' in parsed['encoder']:
        parsed['useIm'] = 'late'
    elif 'im' in parsed['encoder']:
        parsed['useIm'] = True
    else:
        parsed['useIm'] = False
    return parsed
