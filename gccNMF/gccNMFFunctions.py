'''
The MIT License (MIT)

Copyright (c) 2016 Sean UN Wood

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

from numpy.random import random, seed
from numpy import hanning, array, squeeze, arange, concatenate, sqrt, sum, dot, newaxis, linspace, \
    exp, outer, pi, einsum, argsort, mean, hsplit, zeros, empty, min, max, isnan, all, nanargmax, empty_like, \
    where, zeros_like, angle, arctan2, int16, float32, complex64, argmax, take
from scipy.io import wavfile
from scipy.signal import argrelmax
from os.path import basename, join

from gccNMF.librosaSTFT import stft, istft
import logging

SPEED_OF_SOUND_IN_METRES_PER_SECOND = 340.29

def getMixtureFileName(mixtureFileNamePrefix):
    return mixtureFileNamePrefix + '_mix.wav'

def getSourceEstimateFileName(mixtureFileNamePrefix, targetIndex):
    sourceEstimateFileName = mixtureFileNamePrefix + '_sim_%d.wav' % (targetIndex+1)
    return sourceEstimateFileName

def loadMixtureSignal(mixtureFileName):
    sampleRate, stereoSamples = wavfile.read(mixtureFileName)
    return stereoSamples.T / float32(2**16 / 2), sampleRate

def getMaxTDOA(microphoneSeparationInMetres):
    return microphoneSeparationInMetres / SPEED_OF_SOUND_IN_METRES_PER_SECOND

def getTDOAsInSeconds(microphoneSeparationInMetres, numTDOAs):
    maxTDOA = getMaxTDOA(microphoneSeparationInMetres)
    tdoasInSeconds = linspace(-maxTDOA, maxTDOA, numTDOAs)
    return tdoasInSeconds

def getFrequenciesInHz(sampleRate, numFrequencies):
    return linspace(0, sampleRate/2, numFrequencies)

def computeComplexMixtureSpectrogram(stereoSamples, windowSize, hopSize, windowFunction, fftSize=None):
    if fftSize is None:
        fftSize = windowSize

    complexMixtureSpectrograms = array( [stft(squeeze(stereoSamples[channelIndex]).copy(), windowSize, hopSize, fftSize, hanning, center=False)
                                         for channelIndex in arange(2)] )
    return complexMixtureSpectrograms

def performKLNMF(V, dictionarySize, numIterations, sparsityAlpha, epsilon=1e-16, seedValue=0):
    seed(seedValue)
    
    W = random( (V.shape[0], dictionarySize) ).astype(float32) + epsilon
    H = random( (dictionarySize, V.shape[1]) ).astype(float32) + epsilon

    for iterationIndex in range(numIterations):
        H *= dot( W.T, V / dot( W, H ) ) / ( sum(W, axis=0)[:, newaxis] + sparsityAlpha + epsilon )
        W *= dot( V / dot( W, H ), H.T ) / sum(H, axis=1)
        
        dictionaryAtomNorms = sqrt( sum(W**2, 0 ) )
        W /= dictionaryAtomNorms
        H *= dictionaryAtomNorms[:, newaxis]
        
    return W, H

def getAngularSpectrogram(spectralCoherenceV, frequenciesInHz, microphoneSeparationInMetres, numTDOAs):
    numFrequencies, numTime = spectralCoherenceV.shape
    
    tdoasInSeconds = getTDOAsInSeconds(microphoneSeparationInMetres, numTDOAs)
    expJOmega = exp( outer(frequenciesInHz, -(2j * pi) * tdoasInSeconds) )
    
    FREQ, TIME, TDOA = range(3)
    return sum( einsum( spectralCoherenceV, [FREQ, TIME], expJOmega, [FREQ, TDOA], [TDOA, FREQ, TIME] ).real, axis=1 )
    
def estimateTargetTDOAIndexesFromAngularSpectrum(angularSpectrum, microphoneSeparationInMetres, numTDOAs, numSources):
    peakIndexes = argrelmax(angularSpectrum)[0]
    tdoasInSeconds = getTDOAsInSeconds(microphoneSeparationInMetres, numTDOAs)
    
    if numSources:
        logging.info('numSources provided, taking first %d peaks' % numSources )
        sourcePeakIndexes = peakIndexes[ argsort(angularSpectrum[peakIndexes])[-numSources:] ]
        
        if len(sourcePeakIndexes) != numSources:
            logging.info('didn''t find enough peaks in ITDFunctions.estimateTargetTDOAIndexesFromAngularSpectrum... aborting' )
            os._exit(1)
    else:
        kMeans = KMeans(n_clusters=2, n_init=10)
        kMeans.fit(angularSpectrum[peakIndexes][:, newaxis])
        sourcesClusterIndex = argmax(kMeans.cluster_centers_)
        sourcePeakIndexes = peakIndexes[where(kMeans.labels_ == sourcesClusterIndex)].astype('int32')
        logging.info('numSources not provided, found %d sources' % len(sourcePeakIndexes) )
    
    # return sources ordered left to right
    sourcePeakIndexes = sorted(sourcePeakIndexes)
    
    logging.info( 'Found target TDOAs: %s' % str(sourcePeakIndexes) )
    return sourcePeakIndexes

def getTargetTDOAGCCNMFs(coherenceV, microphoneSeparationInMetres, numTDOAs, frequenciesInHz, targetTDOAIndexes, W, stereoH):
    numTargets = len(targetTDOAIndexes)
    
    hypothesisTDOAs = getTDOAsInSeconds(microphoneSeparationInMetres, numTDOAs)
    
    numFrequencies, numTime = coherenceV.shape
    numChannels, numAtom, numTime = stereoH.shape
    normalizedW = W #/ sqrt( sum(W**2, axis=1, keepdims=True) )
    
    expJOmegaTau = exp( outer(frequenciesInHz, -(2j * pi) * hypothesisTDOAs) )

    TIME, FREQ, TDOA, ATOM = range(4)
    targetTDOAGCCNMFs = empty( (numTargets, numAtom, numTime), float32 )
    for targetIndex, targetTDOAIndex in enumerate(targetTDOAIndexes):
        gccChunk = einsum( coherenceV, [FREQ, TIME], expJOmegaTau[:, targetTDOAIndex], [FREQ], [FREQ, TIME] )
        targetTDOAGCCNMFs[targetIndex] = einsum( normalizedW, [FREQ, ATOM], gccChunk, [FREQ, TIME], [ATOM, TIME] ).real
    
    return targetTDOAGCCNMFs
    
def getTargetCoefficientMasks(targetTDOAGCCNMFs, numTargets):
    nanArgMax = nanargmax(targetTDOAGCCNMFs, axis=0)
    
    targetCoefficientMasks = zeros_like(targetTDOAGCCNMFs)
    for targetIndex in range(numTargets):
        targetCoefficientMasks[targetIndex][where(nanArgMax==targetIndex)] = 1
    return targetCoefficientMasks
    
def getTargetSpectrogramEstimates(targetCoefficientMasks, complexMixtureSpectrogram, W, stereoH):
    numTargets = targetCoefficientMasks.shape[0]
    targetSpectrogramEstimates = zeros( (numTargets,) + complexMixtureSpectrogram.shape, complex64 )
    for targetIndex, targetCoefficientMask in enumerate(targetCoefficientMasks):
        for channelIndex, coefficients in enumerate(stereoH):
            targetSpectrogramEstimates[targetIndex, channelIndex] = dot(W, coefficients * targetCoefficientMask)
    return targetSpectrogramEstimates * exp( 1j * angle(complexMixtureSpectrogram) )

def getTargetSignalEstimates(targetSpectrogramEstimates, windowSize, hopSize, windowFunction):
    numTargets, numChannels, numFreq, numTime = targetSpectrogramEstimates.shape
    
    targetSignalEstimates = []
    for targetIndex in range(numTargets):
        currentSignalEstimates = []
        for channelIndex in range(numChannels):
            currentSignalEstimates.append( istft(targetSpectrogramEstimates[targetIndex, channelIndex], hopSize, windowSize, windowFunction) )
        targetSignalEstimates.append(currentSignalEstimates)
    return array(targetSignalEstimates)

def saveTargetSignalEstimates(targetSignalEstimates, sampleRate, mixtureFileNamePrefix):
    numTargets = targetSignalEstimates.shape[0]
    
    targetSignalEstimates = (targetSignalEstimates * float32(2**16 / 2)).astype(int16)
    for targetIndex in range(numTargets):
        sourceEstimateFileName = getSourceEstimateFileName(mixtureFileNamePrefix, targetIndex)
        wavfile.write( sourceEstimateFileName, sampleRate, targetSignalEstimates[targetIndex].T )
