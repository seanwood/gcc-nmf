'''
The MIT License (MIT)

Copyright (c) 2017 Sean UN Wood

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: Sean UN Wood
'''

import numpy as np
from os import makedirs
from os.path import exists, join
import logging
from collections import OrderedDict

from gccNMF.gccNMFFunctions import performKLNMF
from gccNMF.defs import DATA_DIR

PRETRAINED_W_DIR = join(DATA_DIR, 'pretrainedW')
PRETRAINED_W_PATH_TEMPLATE = join(PRETRAINED_W_DIR, 'W_%d.npy')

SPARSITY_ALPHA = 0
NUM_PRELEARNING_ITERATIONS = 100
CHIME_DATASET_PATH = join(DATA_DIR, 'chimeTrainSet.npy')

def getDictionariesW(windowSize, dictionarySizes, ordered=False):
    fftSize = windowSize // 2 + 1
    dictionariesW = OrderedDict( [('Pretrained', OrderedDict( [(dictionarySize, loadPretrainedW(dictionarySize)) for dictionarySize in dictionarySizes] )),
                                  ('Random', OrderedDict( [(dictionarySize, np.random.rand(fftSize, dictionarySize).astype('float32')) for dictionarySize in dictionarySizes] )) ])#,
                                  #('Harmonic', OrderedDict( [(dictionarySize, getHarmonicDictionary(minF0, maxF0, fftSize, dictionarySize, sampleRate, windowFunction=np.hanning)[0]) for dictionarySize in dictionarySizes] ))] )
    
    if not ordered:
        return dictionariesW
    
    orderedDictionariesW = OrderedDict()
    for dictionaryType, dictionaries in dictionariesW.items():
        currentDictionaries = OrderedDict()
        for dictionarySize, dictionary in dictionaries.items():
            currentDictionaries[dictionarySize] = getOrderedDictionary(dictionary)
        orderedDictionariesW[dictionaryType] = currentDictionaries
    return orderedDictionariesW
    
def getOrderedDictionary(W):
    numFreq, _ = W.shape
    spectralCentroids = np.sum( np.arange(numFreq)[:, np.newaxis] * W, axis=0, keepdims=True ) / np.sum(W, axis=0, keepdims=True)
    spectralCentroids = np.squeeze(spectralCentroids)
    orderedAtomIndexes = np.argsort(spectralCentroids)#[::-1]
    orderedW = np.squeeze(W[:, orderedAtomIndexes])
    return orderedW
    
def loadPretrainedW(dictionarySize, retrainW=False):
    pretrainedWFilePath = PRETRAINED_W_PATH_TEMPLATE % dictionarySize
    logging.info('GCCNMFPretraining: Loading pretrained W (size %d): %s' % (dictionarySize, pretrainedWFilePath) )
    if exists(pretrainedWFilePath) and not retrainW:
        W = np.load(pretrainedWFilePath)
    else:
        if retrainW:
            logging.info('GCCNMFPretraining: Retraining W, saving as %s...' % pretrainedWFilePath)
        else:
            logging.info('GCCNMFPretraining: Pretrained W not found at %s, creating...' % pretrainedWFilePath)
        
        trainV = np.load(CHIME_DATASET_PATH)
        W, _ = performKLNMF(trainV, dictionarySize, numIterations=100, sparsityAlpha=0, epsilon=1e-16, seedValue=0)
        
        try:
            makedirs(PRETRAINED_W_DIR)
        except:
            pass
        np.save(pretrainedWFilePath, W)
    return W