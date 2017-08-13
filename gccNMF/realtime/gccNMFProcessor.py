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

import logging
from time import sleep, time
import numpy as np
from numpy.fft import rfft
from multiprocessing import Process
from threading import Thread
import os

from gccNMF.defs import SPEED_OF_SOUND_IN_METRES_PER_SECOND

TARGET_MODE_BOXCAR = 0
TARGET_MODE_MULTIPLE = 1
TARGET_MODE_WINDOW_FUNCTION = 2

GCC_NMF_PARAMETERS_REQUIRING_RESET = ['microphoneSeparationInMetres', 'numTDOAs', 'dictionarySize'] # 'numSources', 'targetMode', 'gccPHATNLEnabled', 'dictionaryType'

class GCCNMFProcess(Process):
    from theano.compile.sharedvalue import SharedVariable
    
    def __init__(self, oladProcessor, numTDOAs, sampleRate, windowSize, numTimePerChunk, dictionariesW, dictionaryType, dictionarySize, numHUpdates, microphoneSeparationInMetres,
                 gccPHATHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories,
                 gccNMFParams, gccNMFDirtyParamNames, processFramesEvent, processFramesDoneEvent, terminateEvent):
        super(GCCNMFProcess, self).__init__()
        
        self.oladProcessor = oladProcessor
        self.gccNMFProcessor = GCCNMFProcessor(numTDOAs, sampleRate, windowSize, numTimePerChunk, dictionariesW, dictionaryType, dictionarySize, numHUpdates, microphoneSeparationInMetres,
                                               gccPHATHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories)

        self.gccNMFParams = gccNMFParams
        self.gccNMFDirtyParamNames = gccNMFDirtyParamNames        
        self.processFramesEvent = processFramesEvent
        self.processFramesDoneEvent = processFramesDoneEvent
        self.terminateEvent = terminateEvent
        
    def run(self):
        try:
            os.nice(-20)
        except OSError:
            pass

        while True:
            #if self.shouldTerminate():
            #    return
            
            #logging.info('GCCNMFProcessor: Waiting for processFramesEvent')
            self.processFramesEvent.wait()
            #logging.info('GCCNMFProcessor: processFramesEvent is set!')
            self.processFramesEvent.clear()
            
            try:
                if not self.gccNMFProcessor.updating and len(self.gccNMFDirtyParamNames) != 0:
                    logging.info('GCCNMFProcessor: Starting update params thread')
                    self.gccNMFProcessor.updating = True
                    self.updatingThread = Thread(target=self.updateGCCNMFParams)
                    self.updatingThread.start()
            except Exception as e:
                logging.info('GCCNMFProcessor: Failed to update params')
                logging.info(e)
                
            self.oladProcessor.processFrames(self.gccNMFProcessor.processFrames)
            
            #logging.info('Setting processFramesDoneEvent')
            self.processFramesDoneEvent.set()
    
    def shouldTerminate(self):
        if self.terminateEvent.is_set():
            logging.debug('GCCNMFProcessor: received terminate')
            return True
        return False
    
    def updateGCCNMFParams(self):
        #startTime = time()
        #logging.info('GCCNMFProcessor: Update params thread started')
        
        dirtyParamNames = list(self.gccNMFDirtyParamNames)
        del self.gccNMFDirtyParamNames[:]
        params = self.gccNMFParams
        
        resetRequired = False
        for parameterName in dirtyParamNames:
            parameterValue = getattr(params, parameterName)
            currentParam = getattr(self.gccNMFProcessor, parameterName)
            if issubclass(type(currentParam), GCCNMFProcess.SharedVariable):
                if currentParam.get_value() != parameterValue:
                    logging.info('GCCNMFProcessor: setting %s: %s (shared)' % (parameterName, parameterValue))
                    if currentParam.dtype == 'float32':
                        currentParam.set_value(np.float32(parameterValue))
                    else:
                        currentParam.set_value(parameterValue)
                    resetRequired |= parameterName in GCC_NMF_PARAMETERS_REQUIRING_RESET
            else:
                if currentParam != parameterValue:
                    logging.info('GCCNMFProcessor: setting %s: %s' % (parameterName, parameterValue))
                    setattr(self.gccNMFProcessor, parameterName, parameterValue)
                    resetRequired |= parameterName in GCC_NMF_PARAMETERS_REQUIRING_RESET

        if resetRequired:
            self.gccNMFProcessor.reset()
        self.gccNMFProcessor.updating = False
        #elapsedTime = time() - startTime
        #logging.info('GCCNMFProcessor: Updating params took: %.10f' % elapsedTime)
    
class GCCNMFProcessor(object):
    def __init__(self, numTDOAs, sampleRate, windowSize, numTimePerChunk, dictionariesW, dictionaryType, dictionarySize, numHUpdates, microphoneSeparationInMetres,
                 gccPHATHistory=None, inputSpectrogramHistory=None, outputSpectrogramHistory=None, coefficientMaskHistories=None):
        super(GCCNMFProcessor, self).__init__()
        
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.numTimePerChunk = numTimePerChunk
        self.dictionariesW = dictionariesW
        self.dictionaryType = dictionaryType
        self.dictionarySize = dictionarySize
        self.microphoneSeparationInMetres = microphoneSeparationInMetres
        
        self.gccPHATHistory = gccPHATHistory
        self.inputSpectrogramHistory = inputSpectrogramHistory
        self.outputSpectrogramHistory = outputSpectrogramHistory
        self.coefficientMaskHistories = coefficientMaskHistories
        
        self.windowFunction = np.sqrt( np.hanning(self.windowSize).astype(np.float32) )[:, np.newaxis]
        self.synthesisWindowFunction = self.windowFunction
        
        self.numTDOAs = numTDOAs
        self.separationEnabled = True
        self.targetMode = TARGET_MODE_WINDOW_FUNCTION
        
        from theano import shared
        self.targetTDOAIndex = shared( np.float32(10.0) )
        self.targetTDOAEpsilon = shared( np.float32(2.0) )
        self.targetTDOABeta = shared( np.float32(1.0) )
        self.targetTDOANoiseFloor = shared( np.float32(0.0) )
        
        self.updating = False
        
        self.reset()
        
    def processFrames(self, windowedSamples):
        if self.updating:
            return np.zeros_like(windowedSamples)
        
        self.complexMixtureSpectrogram[:] = rfft(windowedSamples * self.windowFunction, axis=1).astype(np.complex64)
        self.spectrogram.set_value(self.complexMixtureSpectrogram)
        #self.spectrogram.set_value( rfft(windowedSamples * self.windowFunction, axis=1).astype(np.complex64) )
        
        realGCC = self.getComplexGCC()[0].real
        if self.separationEnabled:
            [inputMask, coefficientMask] = self.getTFMask(realGCC)
            outputSpectrogram = inputMask * self.complexMixtureSpectrogram
            
            if self.coefficientMaskHistories:
                self.coefficientMaskHistories[self.dictionarySize].set(1-coefficientMask)
        else:
            outputSpectrogram = self.complexMixtureSpectrogram.copy()
            if self.coefficientMaskHistories:
                self.coefficientMaskHistories[self.dictionarySize].set( np.zeros( (self.dictionarySize, windowedSamples.shape[-1]), np.float32) )
        
        if self.inputSpectrogramHistory:
            self.inputSpectrogramHistory.set( -np.mean(np.abs(self.complexMixtureSpectrogram), axis=0) ** (1/3.0) )
        if self.gccPHATHistory:
            self.gccPHATHistory.set( np.nanmean(realGCC, axis=0).T )
        if self.outputSpectrogramHistory:
            self.outputSpectrogramHistory.set( -np.nanmean(np.abs(outputSpectrogram), axis=0) ** (1/3.0) )
        
        return np.fft.irfft(outputSpectrogram, axis=1) * self.synthesisWindowFunction
        
    def reset(self):
        logging.info('GCCNMFProcessor: resetting...')
        self.buildTheanoFunctions()
        logging.info('GCCNMFProcessor: done reset.')
    
    def buildTheanoFunctions(self):
        from theano import shared, tensor, function
        
        self.W = self.dictionariesW[self.dictionaryType][self.dictionarySize]
        self.numFrequencies, self.numAtom = self.W.shape
        logging.info( 'Dictionary shape: %s' % str(self.W.shape))
        
        self.frequenciesInHz = np.linspace(0, self.sampleRate/2, self.numFrequencies).astype(np.float32)
        self.maxTDOA = self.microphoneSeparationInMetres / SPEED_OF_SOUND_IN_METRES_PER_SECOND
        self.hypothesisTDOAs = np.linspace(-self.maxTDOA, self.maxTDOA, self.numTDOAs).astype(np.float32)
        self.expJOmegaTau = np.exp( np.outer(self.frequenciesInHz, -(2j * np.pi) * self.hypothesisTDOAs) ).astype(np.complex64)
        self.omegaTau = np.outer(self.frequenciesInHz, -2 * np.pi * self.hypothesisTDOAs).astype(np.float32)
        
        self.complexMixtureSpectrogram = np.zeros( (2, self.numFrequencies, self.numTimePerChunk), 'complex64' )
        self.spectrogram = shared(self.complexMixtureSpectrogram)
        self.coherenceV = self.spectrogram[0] * self.spectrogram[1].conj() / np.abs(self.spectrogram[0]) / np.abs(self.spectrogram[1])                
        self.complexGCC = self.coherenceV[:, :, np.newaxis] * self.expJOmegaTau[:, np.newaxis]
        self.getComplexGCC = function([], [self.complexGCC])
        
        self.realGCC = tensor.tensor3('realGCC', dtype='float32')
        #self.realGCC = self.complexGCC.real
        self.gccNMF = tensor.dot( self.realGCC.T, self.W )
        self.getGCCNMF = function(inputs=[self.realGCC], outputs=[self.gccNMF])
        
        if self.targetMode == TARGET_MODE_BOXCAR:
            self.HMask = tensor.switch( abs(tensor.argmax(self.gccNMF, axis=0).T - self.targetTDOAIndex) < self.targetTDOAEpsilon, 1.0, 0.0 )
        elif self.targetMode == TARGET_MODE_WINDOW_FUNCTION:
            self.HMask = tensor.exp( - (abs(tensor.argmax(self.gccNMF, axis=0).T - self.targetTDOAIndex) / self.targetTDOAEpsilon) ** self.targetTDOABeta ) / (1+self.targetTDOANoiseFloor) + self.targetTDOANoiseFloor
            
        self.recSource = tensor.dot( self.W, self.HMask )
        self.recV = tensor.sum( self.W, axis=-1, keepdims=False )
        self.tfMask = ( self.recSource.T / self.recV ).T
        self.getTFMask = function(inputs=[self.realGCC], outputs=[self.tfMask, self.HMask])
        
    def setTargetTDOARange(self, targetTDOAIndex, targetTDOAEpsilon, targetTDOABeta, targetTDOANoiseFloor):
        self.targetTDOAIndex.set_value( np.float32(targetTDOAIndex) )
        self.targetTDOAEpsilon.set_value( np.float32(targetTDOAEpsilon) )
        self.targetTDOABeta.set_value( np.float32(targetTDOABeta) )
        self.targetTDOANoiseFloor.set_value( np.float32(targetTDOANoiseFloor) )