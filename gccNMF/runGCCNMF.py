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

from gccNMFFunctions import *
from gccNMFPlotting import *

def runGCCNMF(mixtureFilePrefix, windowSize, hopSize, numTDOAs, microphoneSeparationInMetres, numTargets=None, windowFunction=hanning):
    maxTDOA = microphoneSeparationInMetres / SPEED_OF_SOUND_IN_METRES_PER_SECOND
    tdoasInSeconds = linspace(-maxTDOA, maxTDOA, numTDOAs).astype(float32)
    
    mixtureFileName = getMixtureFileName(mixtureFilePrefix)
    stereoSamples, sampleRate = loadMixtureSignal(mixtureFileName)
    complexMixtureSpectrogram = computeComplexMixtureSpectrogram(stereoSamples, windowSize, hopSize, windowFunction)
    numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
    frequenciesInHz = linspace(0, sampleRate / 2.0, numFrequencies)
    
    V = concatenate( abs(complexMixtureSpectrogram), axis=-1 )
    W, H = performKLNMF(V, dictionarySize=128, numIterations=100, sparsityAlpha=0)
    stereoH = array( hsplit(H, numChannels) )
    
    spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])
    angularSpectrogram = getAngularSpectrogram(spectralCoherenceV, frequenciesInHz, microphoneSeparationInMetres, numTDOAs)
    meanAngularSpectrum = mean(angularSpectrogram, axis=-1) 
    targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum(meanAngularSpectrum, microphoneSeparationInMetres, numTDOAs, numTargets)

    targetTDOAGCCNMFs = getTargetTDOAGCCNMFs(spectralCoherenceV, microphoneSeparationInMetres, numTDOAs, frequenciesInHz, targetTDOAIndexes, W, stereoH)
    targetCoefficientMasks = getTargetCoefficientMasks(targetTDOAGCCNMFs, numTargets)
    targetSpectrogramEstimates = getTargetSpectrogramEstimates(targetCoefficientMasks, complexMixtureSpectrogram, W, stereoH)
    targetSignalEstimates = getTargetSignalEstimates(targetSpectrogramEstimates, windowSize, hopSize, windowFunction)
    
    saveTargetSignalEstimates(targetSignalEstimates, sampleRate, mixtureFileNamePrefix)

if __name__ == '__main__':
    # Preprocessing params
    windowSize = 1024
    fftSize = windowSize
    hopSize = 128
    windowFunction = hanning
    
    # TDOA params
    numTDOAs = 128
    
    # NMF params
    dictionarySize = 128
    numIterations = 100
    sparsityAlpha = 0
    
    # Input params    
    mixtureFileNamePrefix = '../data/dev1_female3_liverec_130ms_1m'
    microphoneSeparationInMetres = 1.0
    numSources = 3
    
    runGCCNMF( mixtureFileNamePrefix, windowSize, hopSize, numTDOAs,
               microphoneSeparationInMetres, numSources, windowFunction )