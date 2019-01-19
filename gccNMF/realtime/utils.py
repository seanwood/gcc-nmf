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

import ctypes
from time import time
import numpy as np
from numpy import prod, frombuffer, concatenate, exp, abs
from multiprocessing import Array, Value
import logging

class SharedMemoryCircularBuffer():
    def __init__(self, shape, initValue=0):
        self.array = Array( ctypes.c_double, int(prod(shape)) )
        self.values = frombuffer(self.array.get_obj()).reshape(shape)
        self.values[:] = initValue
        
        self.numValues = self.values.shape[-1]
        
        self.index = Value(ctypes.c_int)    
        self.index.value = 0
        
    def set(self, newValues, index=None):
        index = self.index.value if index is None else index 
        numNewValues = newValues.shape[-1]
        if index + numNewValues < self.numValues:
            self.values[..., index:index+numNewValues] = newValues
            self.index.value = index + numNewValues
            return self.index.value
        else:
            numAtEnd = self.numValues - index
            numAtStart = numNewValues - numAtEnd
        
            self.values[..., index:] = newValues[..., :numAtEnd]
            self.values[..., :numAtStart] = newValues[..., numAtEnd:]
            self.index.value = numAtStart
            return self.index.value
    
    def get(self, index=None):
        index = (self.index.value-1)%self.numValues if index is None else (index % self.numValues)
        #print(self.numValues, self.values.shape)
        return self.values[..., index]

    def getUnraveledArray(self):
        return concatenate( [self.values[:, self.index.value:], self.values[:, :self.index.value]], axis=-1 ) 
        
    def size(self):
        return self.values.shape[-1]

class OverlapAddProcessor(object):
    def __init__(self, numChannels, windowSize, hopSize, blockSize, windowsPerBlock, inputFrames, outputFrames):
        super(OverlapAddProcessor, self).__init__()
        
        self.numChannels = numChannels
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.blockSize = blockSize
        self.windowsPerBlock = windowsPerBlock
        
        # IPC
        self.inputFrames = inputFrames
        self.outputFrames = outputFrames
        
        # Buffers        
        self.numBlocksPerBuffer = 8
        
        self.inputBufferIndex = 0
        self.inputBufferSize = self.blockSize * self.numBlocksPerBuffer
        self.inputBuffer = np.zeros( (self.numChannels, self.inputBufferSize), np.float32 ) 

        self.outputBufferIndex = 0
        self.outputBufferSize = self.blockSize * self.numBlocksPerBuffer
        self.outputBuffer = np.zeros( (self.numChannels, self.outputBufferSize), np.float32 )        
        
        self.windowedSamples = np.zeros( (self.numChannels, self.windowSize, self.windowsPerBlock), np.float32 )
    
    def processFrames(self, processFramesFunction):
        #startTime = time()
        self.inputBuffer[:, :-self.blockSize] = self.inputBuffer[:, self.blockSize:]
        self.inputBuffer[:, -self.blockSize:] = self.inputFrames
        
        self.outputBuffer[:, :-self.blockSize] = self.outputBuffer[:, self.blockSize:]
        self.outputBuffer[:, -self.blockSize:] = 0
        
        windowIndexes = np.arange(self.inputBufferSize - self.windowSize - (self.windowsPerBlock-1)*self.hopSize, self.inputBufferSize-self.windowSize +1, self.hopSize)
        for i, windowIndex in enumerate(windowIndexes):
            self.windowedSamples[..., i] = self.inputBuffer[:, windowIndex:windowIndex+self.windowSize]
        
        processedFrames = processFramesFunction(self.windowedSamples)
        
        for i, windowIndex in enumerate(windowIndexes):
            self.outputBuffer[:, windowIndex:windowIndex+self.windowSize] += processedFrames[..., i]
            
        self.outputFrames[:] = self.outputBuffer[:, -3*self.blockSize:-2*self.blockSize]
        #totalTime = time() - startTime
        #logging.info('processFrames took %f' % totalTime)