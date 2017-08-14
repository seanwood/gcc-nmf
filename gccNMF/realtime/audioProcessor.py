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
import os

from gccNMF.wavfile import pcm2float, float2pcm

class PyAudioStreamProcessor(Process):
    def __init__(self, params, numChannels, sampleRate, windowSize, hopSize, blockSize, deviceNameQuery,
                 playingFlag, paramsNamespace, inputFrames, outputFrames, processFramesEvent, processFramesDoneEvent, terminateEvent):
        super(PyAudioStreamProcessor, self).__init__()

        self.params = params

        self.numChannels = numChannels
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.blockSize = blockSize

        self.playingFlag = playingFlag
        self.paramsNamespace = paramsNamespace
        self.inputFrames = inputFrames
        self.outputFrames = outputFrames
        self.processFramesEvent = processFramesEvent
        self.processFramesDoneEvent = processFramesDoneEvent
        self.terminateEvent = terminateEvent
        self.deviceNameQuery = deviceNameQuery
        
        self.numBlocksPerBuffer = 8
        
        self.processingTimes = []
        self.underflowCounter = 0
        
        self.fileName = None
        self.audioStream = None
        self.pyaudio = None
        self.deviceIndex = None
        self.fileNameChanged = False
 
    def run(self):
        try:
            os.nice(-20)
        except OSError:
            pass
        
        lastPrintTime = tm.time()
        
        while True:
            if self.shouldTerminate():
                return
             
            self.updateFileName()
            self.updatePlaying()
            
            currentTime = tm.time()
            if currentTime - lastPrintTime >= 2 and len(self.processingTimes) != 0:
                logging.info( 'Processing times (min/max/avg): %f, %f, %f' % (np.min(self.processingTimes), np.max(self.processingTimes), np.mean(self.processingTimes)) )
                lastPrintTime = currentTime
                del self.processingTimes[:]
                
            sleep(0.1)
        
    def shouldTerminate(self):
        if self.terminateEvent.is_set():
            logging.debug('AudioStreamProcessor: received terminate')
            try:
                if self.audioStream:
                    self.audioStream.close()
                    logging.debug('AudioStreamProcessor: stream stopped')
            finally:
                return True
        return False
        
    def updateFileName(self):
        try:
            currentFileName = self.paramsNamespace.fileName
        except:
            return
            
        if currentFileName != self.fileName:
            logging.info('AudioStreamProcessor: setting file name: %s' % currentFileName) 
            self.fileName = currentFileName
            
            active = self.active()
            self.resetAudioStream()
            if active:
                self.startStream()

    def updatePlaying(self):
        if bool(self.playingFlag.value) and not self.active():
            self.startStream()
        elif not bool(self.playingFlag.value) and self.active():
            self.stopStream()
    
    def filePlayerCallback(self, in_data, numFrames, time_info, status):
        startTime = tm.time()
        
        if self.sampleIndex+numFrames >= self.numFrames:
            self.sampleIndex = 0
        
        self.inputFrames[:] = self.samples[:, self.sampleIndex:self.sampleIndex+numFrames]
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
            logging.info('AudioStreamProcessor: getbuffer failed... calling tobytes')
            outputBuffer = outputIntArray.tobytes()
        
        self.processingTimes.append(tm.time() - startTime)
        
        return outputBuffer, self.paContinue
    
    def active(self):
        if not self.audioStream:
            return False
        else:
            return self.audioStream.is_active()

    def startStream(self):
        if not self.audioStream:
            self.resetAudioStream()
        logging.info('AudioStreamProcessor: starting stream')
        self.audioStream.start_stream()
        
    def stopStream(self):
        if self.audioStream:
            logging.info('AudioStreamProcessor: stopping stream')
            self.audioStream.stop_stream()
    
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

    def getDeviceIndex(self, deviceQuery):
        foundIndex = None
        logging.info('Querying audio devices for %s:')
        for deviceIndex in range(self.pyaudio.get_device_count()):
            deviceName = self.pyaudio.get_device_info_by_index(deviceIndex)['name']
            if deviceQuery in deviceName and foundIndex is None:
                foundIndex = deviceIndex            
                logging.info( ' *  %s' % deviceName )
            else:
                logging.info( '    %s' % deviceName )
                
        if not foundIndex:
            logging.warn('Device not found %s' % self.deviceNameQuery)
            
        return foundIndex
        
    def resetAudioStream(self):
        import wave        
        import pyaudio
        
        if self.pyaudio is None:
            self.pyaudio = pyaudio.PyAudio()

        if self.audioStream:
            logging.info('AudioStreamProcessor: aborting stream')
            self.audioStream.close()
            self.audioStream = None
        
        self.paContinue = pyaudio.paContinue
        
        waveFile = wave.open(self.fileName, 'rb')
        self.numFrames = waveFile.getnframes()
        samplesString = waveFile.readframes(self.numFrames)
        self.sampleRate = waveFile.getframerate()
        self.bytesPerFrame = waveFile.getsampwidth()
        self.bytesPerFrameAllChannels = self.bytesPerFrame * self.numChannels
        self.format = self.pyaudio.get_format_from_width(self.bytesPerFrame)
        waveFile.close()
            
        samplesIntArray = np.frombuffer(samplesString, dtype='<i2')
        self.samples = pcm2float(samplesIntArray).reshape(-1, self.numChannels).T
        if self.params.normalizeInput:
            maxAbsAmplitude = np.max(np.abs(self.samples))
            self.samples /= (maxAbsAmplitude / self.params.normalizeInputMaxValue)
            logging.info('AudioStreamProcessor: normalizing input. Max abs was %f, now %f' % (maxAbsAmplitude, np.max(np.abs(self.samples))) )

        self.deviceIndex = self.getDeviceIndex(self.deviceNameQuery)
            
        self.sampleIndex = 0
        self.audioStream = self.pyaudio.open(format=self.format,
                                             channels=self.numChannels,
                                             rate=self.sampleRate,
                                             frames_per_buffer=self.blockSize,
                                             output=True,
                                             stream_callback=self.filePlayerCallback,
                                             output_device_index=self.deviceIndex)