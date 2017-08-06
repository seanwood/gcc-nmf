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
import numpy as np
from multiprocessing import Process
from time import sleep
import time as tm

from gccNMF.wavfile import pcm2float, float2pcm

class PyAudioStreamProcessor(Process):
    def __init__(self, numChannels, sampleRate, windowSize, hopSize, blockSize, deviceIndex,
                 togglePlayQueue, togglePlayAck, inputFrames, outputFrames, processFramesEvent, processFramesDoneEvent, terminateEvent):
        super(PyAudioStreamProcessor, self).__init__()

        self.numChannels = numChannels
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.blockSize = blockSize
        
        self.togglePlayQueue = togglePlayQueue
        self.togglePlayAck = togglePlayAck
        self.inputFrames = inputFrames
        self.outputFrames = outputFrames
        self.processFramesEvent = processFramesEvent
        self.processFramesDoneEvent = processFramesDoneEvent
        self.terminateEvent = terminateEvent
        self.deviceIndex = deviceIndex
        
        self.numBlocksPerBuffer = 8
        
        self.processingTimes = []
        self.underflowCounter = 0
        
        self.fileName = None
        self.audioStream = None
        self.pyaudio = None
        
        self.fileNameChanged = False
        
    def run(self):
        #os.nice(-20)
        
        lastPrintTime = tm.time()
        
        while True:
            currentTime = tm.time()            
            
            if self.terminateEvent.is_set():
                logging.debug('AudioStreamProcessor: received terminate')
                if self.audioStream:
                    self.audioStream.close()
                    logging.debug('AudioStreamProcessor: stream stopped')
                return
            
            if not self.togglePlayQueue.empty():
                parameters = self.togglePlayQueue.get()
                logging.debug('AudioStreamProcessor: received togglePlayParams')
                
                fileName = parameters['fileName']
                if fileName != self.fileName:
                    self.fileName = fileName
                    self.fileNameChanged = True
                    if self.active():
                        self.reset()
        
                if 'stop' in parameters: self.stopStream()
                elif 'start' in parameters: self.startStream()        
                
                logging.debug('AudioStreamProcessor: processed togglePlayParams')
                self.togglePlayAck.set()
                logging.debug('AudioStreamProcessor: ack set')
            elif currentTime - lastPrintTime >= 2:
                if len(self.processingTimes) != 0:
                    logging.info( 'Processing times (min/max/avg): %f, %f, %f' % (np.min(self.processingTimes), np.max(self.processingTimes), np.mean(self.processingTimes)) )
                    lastPrintTime = currentTime
                    del self.processingTimes[:]
            else:
                sleep(0.1)
    
    def filePlayerCallback(self, in_data, numFrames, time_info, status):
        startTime = tm.time()
        
        if self.sampleIndex+numFrames >= self.numFrames:
            self.sampleIndex = 0
        
        inputBuffer = self.samples[self.sampleIndex*self.bytesPerFrameAllChannels:(self.sampleIndex+numFrames)*self.bytesPerFrameAllChannels]
        inputIntArray = np.frombuffer(inputBuffer, dtype='<i2')
        self.inputFrames[:] = pcm2float(inputIntArray).reshape(-1, self.numChannels).T
        self.sampleIndex += numFrames
        
        #logging.info('AudioStreamProcessor: setting processFramesEvent')
        self.processFramesDoneEvent.clear()
        self.processFramesEvent.set()
        #logging.info('AudioStreamProcessor: waiting for processFramesDoneEvent')
        self.processFramesDoneEvent.wait()
        #logging.info('AudioStreamProcessor: done waiting for processFramesDoneEvent')
        
        outputIntArray = float2pcm(self.outputFrames.T.flatten())
        try:
            outputBuffer = np.getbuffer(outputIntArray)
        except:
            outputBuffer = outputIntArray.tobytes()
        
        self.processingTimes.append(tm.time() - startTime)
        
        return outputBuffer, self.paContinue
    
    def active(self):
        if not self.audioStream:
            return False
        else:
            return self.audioStream.is_active()

    def startStream(self):
        if not self.audioStream or self.fileNameChanged:
            logging.info('AudioStreamProcessor: creating stream...')
            self.fileNameChanged = False
            self.reset()
        logging.info('AudioStreamProcessor: starting stream')
        self.audioStream.start_stream()
        
    def stopStream(self):
        if self.audioStream:
            logging.info('AudioStreamProcessor: stopping stream')
            self.audioStream.stop_stream()
            
    def reset(self):
        if self.audioStream:
            logging.info('AudioStreamProcessor: aborting stream')
            self.audioStream.close()
        self.createAudioStream()
        
    def togglePlay(self):
        self.stopStream() if self.active() else self.startStream()

    def logProcessingTimes(self):
        if len(self.processingTimes) == 0:
            return
        
        with self.processingTimesLock:
            minProcessingTime = np.min(self.processingTimes)
            maxProcessingTime = np.max(self.processingTimes)
            meanProcessingTime = np.mean(self.processingTimes)
            stdProcessingTime = np.std(self.processingTimes)
            
            minTimeToProcess = np.min(self.timesToProcess)
            maxTimeToProcess = np.max(self.timesToProcess)
            meanTimeToProcess = np.mean(self.timesToProcess)
            stdTimeToProcess = np.std(self.timesToProcess)
            del self.processingTimes[:]
        with self.underflowCounterLock:
            numUnderflows = self.underflowCounter
            self.underflowCounter = 0
        logging.info( 'Min/max/mean/std processing time: %f, %f, %f, %f. Num underflows: %d (min/max/meanTimeToProcess' % (minProcessingTime, maxProcessingTime, meanProcessingTime, stdProcessingTime, numUnderflows) )
        logging.info( 'min/max/mean/std time to process: %f, %f, %f, %f' % (minTimeToProcess, maxTimeToProcess, meanTimeToProcess, stdTimeToProcess) )
            
    def createAudioStream(self):
        import wave        
        import pyaudio
        
        if self.pyaudio is None:
            self.pyaudio = pyaudio.PyAudio()
        
        self.paContinue = pyaudio.paContinue
        
        waveFile = wave.open(self.fileName, 'rb')
        self.numFrames = waveFile.getnframes()
        self.samples = waveFile.readframes(self.numFrames)
        self.sampleRate = waveFile.getframerate()
        self.bytesPerFrame = waveFile.getsampwidth()
        self.bytesPerFrameAllChannels = self.bytesPerFrame * self.numChannels
        self.format = self.pyaudio.get_format_from_width(self.bytesPerFrame)
        waveFile.close()
        self.waveFile = wave.open(self.fileName, 'rb')
            
        self.sampleIndex = 0
        self.audioStream = self.pyaudio.open(format=self.format,
                                             channels=self.numChannels,
                                             rate=self.sampleRate,
                                             frames_per_buffer=self.blockSize,
                                             output=True,
                                             stream_callback=self.filePlayerCallback)