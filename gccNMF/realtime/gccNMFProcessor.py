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
from time import sleep
import numpy as np
from numpy.fft import rfft
from multiprocessing import Process

from gccNMF.defs import SPEED_OF_SOUND_IN_METRES_PER_SECOND

TARGET_MODE_BOXCAR = 0
TARGET_MODE_MULTIPLE = 1
TARGET_MODE_WINDOW_FUNCTION = 2

class GCCNMFProcess(Process):
    def __init__(self, oladProcessor, sampleRate, windowSize, numTimePerChunk, dictionariesW, dictionaryType, dictionarySize, numHUpdates, microphoneSeparationInMetres, localizationEnabled, localizationWindowSize,
                 gccPHATHistory, tdoaHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories, 
                 tdoaParametersQueue, tdoaParametersAck, togglePlayQueue, togglePlayAck, toggleSeparationQueue, toggleSeparationAck,
                 processFramesEvent, processFramesDoneEvent, terminateEvent):
        super(GCCNMFProcess, self).__init__()

        self.oladProcessor = oladProcessor
        self.gccNMFProcessor = GCCNMFProcessor(sampleRate, windowSize, numTimePerChunk, dictionariesW, dictionaryType, dictionarySize, numHUpdates, microphoneSeparationInMetres,
                                               localizationEnabled, localizationWindowSize, gccPHATHistory, tdoaHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories)
        
        self.tdoaParametersQueue = tdoaParametersQueue
        self.tdoaParametersAck = tdoaParametersAck
        self.togglePlayQueue = togglePlayQueue
        self.togglePlayAck = togglePlayAck
        self.toggleSeparationQueue = toggleSeparationQueue
        self.toggleSeparationAck = toggleSeparationAck
        
        self.processFramesEvent = processFramesEvent
        self.processFramesDoneEvent = processFramesDoneEvent
        self.terminateEvent = terminateEvent
        
    def run(self):
        #os.nice(-20)
        while True:
            if self.terminateEvent.is_set():
                logging.info('GCCNMFProcessor: received terminate')
                return
            
            wait = True
            
            if not self.tdoaParametersQueue.empty():
                logging.debug('GCCNMFProcessor: received tdoaParams')
                self.processTDOAParametersQueue()
                logging.debug('AudioStreamProcessor: processed tdoaParams')
                self.tdoaParametersAck.set()
                logging.debug('AudioStreamProcessor: ack set')
                wait = False
            
            if not self.togglePlayQueue.empty():
                logging.debug('GCCNMFProcessor: received togglePlayParams')
                self.processTogglePlayQueue()
                logging.debug('GCCNMFProcessor: processed togglePlayParams')
                self.togglePlayAck.set()
                logging.debug('GCCNMFProcessor: ack set')
                wait = False
                
            if not self.toggleSeparationQueue.empty():
                logging.debug('GCCNMFProcessor: received toggleSeparationParams')
                self.processToggleSeparationQueue()
                logging.debug('GCCNMFProcessor: processed toggleSeparationParams')
                self.toggleSeparationAck.set()
                logging.debug('GCCNMFProcessor: ack set')
                wait = False
            
            if self.processFramesEvent.is_set():
                self.processFramesEvent.clear()
                #logging.info('GCCNMFProcessor: received processFramesEvent')
                self.oladProcessor.processFrames(self.gccNMFProcessor.processFrames)
                #logging.info('GCCNMFProcessor: setting processFramesDoneEvent')
                self.processFramesDoneEvent.set()
                #logging.info('GCCNMFProcessor: set processFramesDoneEvent')
                wait = False
            
            if wait:
                sleep(0.001)
    
    def processTDOAParametersQueue(self):
        parameters = self.tdoaParametersQueue.get()
        if 'targetTDOAIndexes' in parameters:
            targetTDOAIndexes = parameters['targetTDOAIndexes']
            logging.info( 'GCCNMFProcessor: setting targetTDOAIndexes: %s' % str(targetTDOAIndexes) )
            self.gccNMFProcessor.setTargetTDOAIndexes(targetTDOAIndexes)
        elif 'localizationEnabled' in parameters:
            localizationEnabled = parameters['localizationEnabled']
            localizationWindowSize = parameters['localizationWindowSize']
            logging.info( 'GCCNMFProcessor: setting localizationEnabled: %s, localizationWindowSize %s' % (str(localizationEnabled), str(localizationWindowSize)) )
            self.gccNMFProcessor.localizationEnabled = localizationEnabled
            self.gccNMFProcessor.localizationWindowSize = localizationWindowSize
        else:
            targetTDOAIndex = parameters['targetTDOAIndex']
            targetTDOAEpsilon = parameters['targetTDOAEpsilon']
            targetTDOABeta = parameters['targetTDOABeta']
            targetTDOANoiseFloor = parameters['targetTDOANoiseFloor']
            logging.info( 'GCCNMFProcessor: setting targetTDOAIndex: %.2f, targetTDOAEpsilon: %.2f, targetTDOABeta: %.2f, targetTDOANoiseFloor: %.2f' % \
                          (targetTDOAIndex, targetTDOAEpsilon, targetTDOABeta, targetTDOANoiseFloor)) 
            self.gccNMFProcessor.setTargetTDOARange(targetTDOAIndex, targetTDOAEpsilon, targetTDOABeta, targetTDOANoiseFloor)
             
    def processTogglePlayQueue(self):
        from theano.compile.sharedvalue import SharedVariable
        
        parameters = self.togglePlayQueue.get()
        parametersRequiringReset = ['microphoneSeparationInMetres', 'numTDOAs', 'numSources', 'targetMode',
                                    'dictionarySize', 'dictionaryType', 'gccPHATNLEnabled']

        resetGCCNMFProcessor = False
        for parameterName, parameterValue in parameters.items():
            if not hasattr(self.gccNMFProcessor, parameterName):
                logging.info('GCCNMFProcessor: setting %s: %s' % (parameterName, parameterValue))
                setattr(self.gccNMFProcessor, parameterName, parameterValue)
                resetGCCNMFProcessor |= parameterName in parametersRequiringReset
            else:
                currentParam = getattr(self.gccNMFProcessor, parameterName)
                if issubclass(type(currentParam), SharedVariable):
                    if currentParam.get_value() != parameterValue:
                        logging.info('GCCNMFProcessor: setting %s: %s (shared)' % (parameterName, parameterValue))
                        currentParam.set_value(parameterValue)
                    else:
                        logging.info('GCCNMFProcessor: %s unchanged: %s (shared)' % (parameterName, parameterValue))
                else:
                    if currentParam != parameterValue:
                        logging.info('GCCNMFProcessor: setting %s: %s' % (parameterName, parameterValue))
                        setattr(self.gccNMFProcessor, parameterName, parameterValue)
                    else:
                        logging.info('GCCNMFProcessor: %s unchanged: %s' % (parameterName, parameterValue))
                resetGCCNMFProcessor |= parameterName in parametersRequiringReset

        if resetGCCNMFProcessor:
            self.gccNMFProcessor.reset()
    
    def processToggleSeparationQueue(self):
        parameters = self.toggleSeparationQueue.get()

        if 'separationEnabled' in parameters:
            separationEnabled = parameters['separationEnabled']
            logging.info( 'GCCNMFProcessor: setting separationEnabled: %s' % str(separationEnabled) )
            self.gccNMFProcessor.separationEnabled = separationEnabled
    
class GCCNMFProcessor(object):
    def __init__(self, sampleRate, windowSize, numTimePerChunk, dictionariesW, dictionaryType, dictionarySize, numHUpdates, microphoneSeparationInMetres,
                 localizationEnabled, localizationWindowSize, gccPHATHistory=None, tdoaHistory=None, inputSpectrogramHistory=None, outputSpectrogramHistory=None, coefficientMaskHistories=None):
        super(GCCNMFProcessor, self).__init__()
        
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.numTimePerChunk = numTimePerChunk
        self.dictionariesW = dictionariesW
        self.dictionaryType = dictionaryType
        self.dictionarySize = dictionarySize
        self.microphoneSeparationInMetres = microphoneSeparationInMetres
        
        self.gccPHATHistory = gccPHATHistory
        self.tdoaHistory = tdoaHistory
        self.inputSpectrogramHistory = inputSpectrogramHistory
        self.outputSpectrogramHistory = outputSpectrogramHistory
        self.coefficientMaskHistories = coefficientMaskHistories
        
        self.windowFunction = np.sqrt( np.hamming(self.windowSize).astype(np.float32) )[:, np.newaxis]
        self.synthesisWindowFunction = self.windowFunction
        
        self.numTDOAs = None
        self.separationEnabled = True
        self.localizationEnabled = localizationEnabled
        self.localizationWindowSize = localizationWindowSize
        self.targetMode = TARGET_MODE_WINDOW_FUNCTION
        
        from theano import shared
        self.targetTDOAIndex = shared( np.float32(10.0) )
        self.targetTDOAEpsilon = shared( np.float32(2.0) )
        self.targetTDOABeta = shared( np.float32(1.0) )
        self.targetTDOANoiseFloor = shared( np.float32(0.0) )
        
    def processFrames(self, windowedSamples):
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
        
        if self.inputSpectrogramHistory:
            self.inputSpectrogramHistory.set( -np.mean(np.abs(self.complexMixtureSpectrogram), axis=0) ** (1/3.0) )
        if self.gccPHATHistory:
            self.gccPHATHistory.set( np.nanmean(realGCC, axis=0).T )
        if self.tdoaHistory:
            if self.localizationEnabled:
                gccPHATHistory = self.gccPHATHistory.getUnraveledArray()
                tdoaIndex = np.argmax( np.nanmean(gccPHATHistory[:, -self.localizationWindowSize:], axis=-1) )
                #tdoaIndex = (self.targetTDOAIndex.get_value() + 1) % self.numTDOAs
                #tdoaIndex = np.random.randint(0, self.numTDOAs+1)
                self.targetTDOAIndex.set_value(tdoaIndex)
            self.tdoaHistory.set( np.array( [[self.targetTDOAIndex.get_value()]] ) )
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