#!/usr/bin/env python

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
import ctypes
import numpy as np

from multiprocessing import Event, Queue, Array, freeze_support

from gccNMF.defs import DEFAULT_AUDIO_FILE, DEFAULT_CONFIG_FILE
from gccNMF.realtime.utils import SharedMemoryCircularBuffer, OverlapAddProcessor
from gccNMF.realtime.config import getGCCNMFConfigParams, parseArguments
from gccNMF.realtime.audioProcessor import PyAudioStreamProcessor as AudioStreamProcessor
from gccNMF.realtime.gccNMFProcessor import GCCNMFProcess

class RealtimeGCCNMF(object):
    def __init__(self, audioPath=DEFAULT_AUDIO_FILE, configPath=DEFAULT_CONFIG_FILE):
        params = getGCCNMFConfigParams(audioPath, configPath)
        
        logging.info('RealtimeGCCNMF: Starting with audio path: %s' % params.audioPath)
        
        self.initQueuesAndEvents()
        self.initSharedArrays(params)
        self.initHistoryBuffers(params)
        self.initProcesses(params)
        
        self.run(params)
    
    def initQueuesAndEvents(self):
        self.togglePlayAudioProcessQueue = Queue()
        self.togglePlayAudioProcessAck = Event()
        self.togglePlayGCCNMFProcessQueue = Queue()
        self.togglePlayGCCNMFProcessAck = Event()
        self.toggleSeparationGCCNMFProcessQueue = Queue()
        self.toggleSeparationGCCNMFProcessAck = Event()
        self.tdoaParamsGCCNMFProcessQueue = Queue()
        self.tdoaParamsGCCNMFProcessAck = Event()
        
        self.processFramesEvent = Event()
        self.processFramesDoneEvent = Event()
        self.terminateEvent = Event()
    
    def initSharedArrays(self, params):    
        inputFramesArray = Array(ctypes.c_double, params.numChannels*params.blockSize)
        self.inputFrames = np.frombuffer(inputFramesArray.get_obj()).reshape( (params.numChannels, -1) )
        outputFramesArray = Array(ctypes.c_double, params.numChannels*params.blockSize)
        self.outputFrames = np.frombuffer(outputFramesArray.get_obj()).reshape( (params.numChannels, -1) )
        
    def initHistoryBuffers(self, params):
        self.gccPHATHistory = SharedMemoryCircularBuffer( (params.numTDOAs, params.numTDOAHistory) )
        self.tdoaHistory = SharedMemoryCircularBuffer( (1, params.numTDOAHistory) )
        self.inputSpectrogramHistory = SharedMemoryCircularBuffer( (params.numFreq, params.numSpectrogramHistory) )
        self.outputSpectrogramHistory = SharedMemoryCircularBuffer( (params.numFreq, params.numSpectrogramHistory) )
        self.coefficientMaskHistories = {}
        for size in params.dictionarySizes:
            self.coefficientMaskHistories[size] = SharedMemoryCircularBuffer( (size, params.numSpectrogramHistory) )
        
    def initProcesses(self, params):
        self.audioProcess = AudioStreamProcessor(params.numChannels, params.sampleRate, params.windowSize, params.hopSize, params.blockSize, params.deviceIndex,
                                                 self.togglePlayAudioProcessQueue, self.togglePlayAudioProcessAck,
                                                 self.inputFrames, self.outputFrames, self.processFramesEvent, self.processFramesDoneEvent, self.terminateEvent)
        self.oladProcessor = OverlapAddProcessor(params.numChannels, params.windowSize, params.hopSize, params.blockSize, params.windowsPerBlock, self.inputFrames, self.outputFrames)
        self.gccNMFProcess = GCCNMFProcess(self.oladProcessor, params.sampleRate, params.windowSize, params.windowsPerBlock, params.dictionariesW, params.dictionaryType, params.dictionarySize, params.numHUpdates, params.microphoneSeparationInMetres, params.localizationEnabled, params.localizationWindowSize,
                                           self.gccPHATHistory, self.tdoaHistory, self.inputSpectrogramHistory, self.outputSpectrogramHistory, self.coefficientMaskHistories,
                                           self.tdoaParamsGCCNMFProcessQueue, self.tdoaParamsGCCNMFProcessAck, self.togglePlayGCCNMFProcessQueue, self.togglePlayGCCNMFProcessAck, self.toggleSeparationGCCNMFProcessQueue, self.toggleSeparationGCCNMFProcessAck,
                                           self.processFramesEvent, self.processFramesDoneEvent, self.terminateEvent)
        self.audioProcess.start()
        self.gccNMFProcess.start()
    
    def run(self, params):
        try:
            from pyqtgraph.Qt import QtGui
            from gccNMF.realtime.gccNMFInterface import RealtimeGCCNMFInterfaceWindow
            
            app = QtGui.QApplication([])
            gccNMFInterfaceWindow = RealtimeGCCNMFInterfaceWindow(params.audioPath, params.numTDOAs, params.gccPHATNLAlpha, params.gccPHATNLEnabled, params.dictionariesW, params.dictionarySize,
                                                                  params.dictionarySizes, params.dictionaryType, params.numHUpdates, params.localizationEnabled, params.localizationWindowSize,
                                                                  self.gccPHATHistory, self.tdoaHistory, self.inputSpectrogramHistory, self.outputSpectrogramHistory, self.coefficientMaskHistories,
                                                                  self.togglePlayAudioProcessQueue, self.togglePlayAudioProcessAck,
                                                                  self.togglePlayGCCNMFProcessQueue, self.togglePlayGCCNMFProcessAck,
                                                                  self.tdoaParamsGCCNMFProcessQueue, self.tdoaParamsGCCNMFProcessAck,
                                                                  self.toggleSeparationGCCNMFProcessQueue, self.toggleSeparationGCCNMFProcessAck)
            app.exec_()
            logging.info('Window closed')
            self.terminateEvent.set()
    
            self.audioProcess.join()
            logging.info('Audio process joined')
            
            self.gccNMFProcess.join()
            logging.info('GCCNMF process joined')
        
        finally:
            self.audioProcess.terminate()
            self.gccNMFProcess.terminate()
    
class RealtimeGCCNMFNoGUI(RealtimeGCCNMF):
    def __init__(self, audioPath=DEFAULT_AUDIO_FILE, configPath=DEFAULT_CONFIG_FILE):
        super(RealtimeGCCNMFNoGUI, self).__init__(audioPath, configPath)
    
    def initHistoryBuffers(self, params):
        self.gccPHATHistory = None
        self.inputSpectrogramHistory = None
        self.outputSpectrogramHistory = None
        self.coefficientMaskHistories = None
    
    def queueParams(self, queue, ack, params, label='params'):
        ack.clear()
        logging.debug('HeadlessGCCNMF: putting %s' % label)
        queue.put(params)
        logging.debug('HeadlessGCCNMF: put %s' % label)
        ack.wait()
        logging.debug('HeadlessGCCNMF: ack received')
        
    def initParams(self, params):
        self.queueParams(self.tdoaParamsGCCNMFProcessQueue,
                         self.tdoaParamsGCCNMFProcessAck,
                         {'targetTDOAIndex': 9.60,
                          'targetTDOAEpsilon': params.targetTDOAEpsilon,
                          'targetTDOABeta': params.targetTDOABeta,
                          'targetTDOANoiseFloor': params.targetTDOANoiseFloor},
                         'gccNMFProcessTDOAParameters (region)')

        self.queueParams(self.togglePlayGCCNMFProcessQueue,
                         self.togglePlayGCCNMFProcessAck,
                         {'numTDOAs': params.numTDOAs,
                          'dictionarySize': params.dictionarySize,
                          'microphoneSeparationInMetres': params.microphoneSeparationInMetres},
                         'gccNMFProcessTogglePlayParameters')

        self.queueParams(self.toggleSeparationGCCNMFProcessQueue,
                         self.toggleSeparationGCCNMFProcessAck,
                         {'separationEnabled': True},
                         'separationEnabledParameters')

        self.queueParams(self.togglePlayAudioProcessQueue,
                         self.togglePlayAudioProcessAck,
                         {'fileName': params.audioPath,
                          'start': ''},
                         'audioProcessParameters')
            
    def run(self, params):
        self.initParams(params)
        
        try:
            self.audioProcess.join()
            logging.info('Audio process joined')
            
            self.gccNMFProcess.join()
            logging.info('GCCNMF process joined')
        finally:
            self.audioProcess.terminate()
            self.gccNMFProcess.terminate()
        logging.info('Done.')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    freeze_support() # for multiprocessing on Windows
    
    args = parseArguments()
    if not args.no_gui:
        RealtimeGCCNMF(args.input, args.config)
    else:
        RealtimeGCCNMFNoGUI(args.input, args.config)