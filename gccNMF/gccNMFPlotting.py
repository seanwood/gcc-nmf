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

from gccNMF.gccNMFFunctions import *
from matplotlib.pyplot import *
from numpy import arange, max, abs, linspace, angle

import matplotlib.pyplot as pyplot

def imshow(x, **args):
    pyplot.imshow(x, aspect='auto', origin='lower', interpolation='nearest', **args)

def plotMixtureSignal(stereoSamples, sampleRate):
    numChannels, numSamples = stereoSamples.shape
    sampleTimesInSeconds = arange(numSamples) / float(sampleRate)
    maxValue = max(abs(stereoSamples)) * 1.1
    
    subplot(211)
    plot(sampleTimesInSeconds, stereoSamples[0])
    axis('tight')
    ylim((-maxValue, maxValue))
    title('Left channel')
    subplot(212)
    plot(sampleTimesInSeconds, stereoSamples[1])
    axis('tight')
    title('Right channel')
    ylim((-maxValue, maxValue))

def plotSpectrogram(spectrogram, maxValue, durationInSeconds, frequenciesInkHz, titleLabel):
    imshow((spectrogram / maxValue) ** (1/3.0),
           extent=[0, durationInSeconds, frequenciesInkHz[0], frequenciesInkHz[-1]],
           cmap=cm.binary)
    title(titleLabel)
    ylabel('Frequency (kHz)')
    xlabel('Time (s)')

def plotMixtureSpectrograms(complexMixtureSpectrogram, frequenciesInkHz, durationInSeconds):
    numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
    
    magnitudeSpectrograms = abs(complexMixtureSpectrogram)
    maxSpectrogramValue = max(magnitudeSpectrograms)
    
    phaseSpectrograms = angle(complexMixtureSpectrogram)
    
    subplot(211)
    plotSpectrogram(magnitudeSpectrograms[0], maxSpectrogramValue, durationInSeconds, frequenciesInkHz, 'Left Magnitude Spectrogram')
    subplot(212)
    plotSpectrogram(magnitudeSpectrograms[1], maxSpectrogramValue, durationInSeconds, frequenciesInkHz, 'Right Magnitude Spectrogram')
    xlabel('Time (s)')
    
    tight_layout()
    
def plotGCCPHATLocalization(spectralCoherenceV, angularSpectrogram, meanAngularSpectrum, targetTDOAIndexes, microphoneSeparationInMetres, numTDOAs, durationInSeconds):
    maxTDOA = getMaxTDOA(microphoneSeparationInMetres)
    tdoasInSeconds = getTDOAsInSeconds(microphoneSeparationInMetres, numTDOAs)
    
    subplot(121)
    rectifiedAngularSpectrogram = angularSpectrogram.copy()
    rectifiedAngularSpectrogram[rectifiedAngularSpectrogram < 0] = 0
    imshow( rectifiedAngularSpectrogram, extent=[0, durationInSeconds, -maxTDOA, maxTDOA], cmap=cm.binary)
    ylabel('TDOA (s)')
    xlabel('Time (s)')
    title('GCC-PHAT Angular Spectrogram')
    
    subplot(122)
    plot(tdoasInSeconds, meanAngularSpectrum, 'k')
    axis('tight')
    yMin, yMax = ylim()
    marginSize = (yMax-yMin) * 0.1
    yMin -= marginSize
    yMax += marginSize
    
    for targetTDOAIndex in targetTDOAIndexes:
        plot( (tdoasInSeconds[targetTDOAIndex], tdoasInSeconds[targetTDOAIndex]), (yMin, yMax), c='b', linestyle='--' )
    ylim( (yMin, yMax) )
    
    xlabel('TDOA (s)')
    title('Mean GCC-PHAT Angular Spectrum')
   
def plotRows(xValues, rowValues, color, scalingFactor):
    numRows = rowValues.shape[0]
    maxValue = max(rowValues)
    for rowIndex in range(numRows):
        plot(xValues, rowValues[rowIndex] / maxValue * scalingFactor + rowIndex, color)
    axis('tight')
    ylim( (0, numRows) )
    
def plotCols(yValues, colValues, color, scalingFactor):
    numCols = colValues.shape[1]
    maxValue = max(colValues)
    for colIndex in range(numCols):
        plot(yValues, colValues[:, colIndex] / maxValue * scalingFactor + colIndex, color)
    axis('tight')
    ylim( (0, numCols))

def plotNMFDecomposition(V, W, H, freqsInkHz, durationInSeconds, numAtomsToPlot, wRootFactor = 1/3.0, hRootFactor = 1/1.2):
    numFreq, dictionarySize = W.shape
    dictionarySize, numWindows = H.shape
    stftTimesInSeconds = linspace(0, durationInSeconds*2, numWindows)
    
    plotColor = 'k'
    scalingFactor = .9
    
    subplot(311)
    imshow( V ** (1/3.0), extent=[0, durationInSeconds*2, freqsInkHz[0], freqsInkHz[-1]], cmap=cm.binary )
    plot( (durationInSeconds, durationInSeconds), (freqsInkHz[0], freqsInkHz[-1]), '--' )
    title('Input V (left and right spectrograms concatenated in time)')
    ylabel('Frequency (kHz)')
    xlabel('Time (s)')
    
    subplot(323)
    imshow( W.T ** wRootFactor, extent=[freqsInkHz[0], freqsInkHz[-1], 0, dictionarySize], cmap=cm.binary )
    title('Dictionary W')
    ylabel('Atom index')
    xlabel('Frequency (kHz)')
    
    # plot a subset of dictionary atoms in detail: columns of W
    subplot(324)
    title('Subset of dictionary atoms (detail)')
    plotCols(freqsInkHz, W[:, :numAtomsToPlot] ** wRootFactor, plotColor, scalingFactor)
    ylabel('Atom index')
    xlabel('Frequency (kHz)')
    
    subplot(325)
    imshow( H ** hRootFactor, extent=[0, durationInSeconds*2, 0, dictionarySize], cmap=cm.binary )
    plot( (durationInSeconds, durationInSeconds), (0, dictionarySize), '--' )
    ylim(0, dictionarySize-1)
    title('Coefficients H (left and right coefficients concatenated in time)')
    ylabel('Atom index')
    xlabel('Time (s)')
    
    # plot a subset of dictionary atoms in detail (left): rows of H
    subplot(326)
    title('Subset of left and right atom coefficients H (detail)')
    plotRows(stftTimesInSeconds, H[:numAtomsToPlot] ** hRootFactor, plotColor, scalingFactor)
    plot( (durationInSeconds, durationInSeconds), (0, numAtomsToPlot), '--' )
    ylabel('Atom index')
    xlabel('Time (s)')
    
    tight_layout()
    
def plotCoefficientMasks(targetCoefficientMasks, stereoH, durationInSeconds):
    hRootFactor = 1/2.0

    numTargets, numAtoms, numWindows = targetCoefficientMasks.shape
    stftTimesInSeconds = linspace(0, durationInSeconds, numWindows)

    for targetIndex, targetCoefficientMask in enumerate(targetCoefficientMasks):
        subplot2grid( (3, numTargets), (0, targetIndex) )
        imshow(targetCoefficientMask, cmap=cm.binary,
               extent=[stftTimesInSeconds[0], stftTimesInSeconds[-1], 0, numAtoms]) 
        title('Source %d coefficient mask' % (targetIndex+1))
        if targetIndex == 0:
            ylabel('Atom index')
        
        subplot2grid( (3, numTargets), (1, targetIndex) )
        imshow( (targetCoefficientMask * stereoH[0]) ** hRootFactor, cmap=cm.binary,
                extent=[stftTimesInSeconds[0], stftTimesInSeconds[-1], 0, numAtoms]) 
        title('Source %d masked coefficients (left)' % (targetIndex+1))
        if targetIndex == 0:
            ylabel('Atom index')
        
        subplot2grid( (3, numTargets), (2, targetIndex) )
        imshow( (targetCoefficientMask * stereoH[1]) ** hRootFactor, cmap=cm.binary,
                extent=[stftTimesInSeconds[0], stftTimesInSeconds[-1], 0, numAtoms]) 
        title('Source %d masked coefficients (right)' % (targetIndex+1))
        if targetIndex == 0:
            ylabel('Atom index')
        xlabel('Time (s)')
    
    tight_layout()

def plotTargetSpectrogramEstimates(targetSpectrogramEstimates, durationInSeconds, frequenciesInkHz):
    magnitudeSpectrogramEstimates = abs(targetSpectrogramEstimates)
    
    numTargets, numChannels, numFrequencies, numTime = targetSpectrogramEstimates.shape
    channelLabels = ['left', 'right']
    for targetIndex, spectrogramEstimates in enumerate( magnitudeSpectrogramEstimates ):
        for channelIndex, spectrogramEstimate in enumerate(spectrogramEstimates):
            subplot2grid( (numTargets, numChannels), (targetIndex, channelIndex) )
            imshow( spectrogramEstimate ** (1/3.0), extent=[0, durationInSeconds, frequenciesInkHz[0], frequenciesInkHz[-1]],
                    cmap=cm.binary )
            title('Source %d, %s' % (targetIndex+1, channelLabels[channelIndex]))
            
            if channelIndex == 0:
                ylabel('Frequency (kHz)')
            if targetIndex == numTargets-1:
                xlabel('Time (s)')
            else:
                xticks([])
                
def plotTargetSignalEstimates(targetSignalEstimates, sampleRate):
    numTargets, numChannels, numSamples = targetSignalEstimates.shape 
    recSampleTimesInSeconds = arange(numSamples) / float(sampleRate)
    
    maxValue = max(abs(targetSignalEstimates)) * 1.05
    
    channelLabels = ['left', 'right']
    for targetIndex, signalEstimates in enumerate(targetSignalEstimates):
        for channelIndex, signalEstimate in enumerate(signalEstimates):
            subplot2grid( (numTargets, numChannels), (targetIndex, channelIndex) )
            plot(recSampleTimesInSeconds, signalEstimate)
            title('Source %d, %s' % (targetIndex+1, channelLabels[channelIndex]))
            
            ylim( (-maxValue, maxValue) )
            
            if channelIndex == 0:
                ylabel('Frequency (kHz)')
            if targetIndex == numTargets-1:
                xlabel('Time (s)')
    
    tight_layout()

def plotTargetSignalEstimate(targetSignalEstimate, sampleRate, titleString):
    numChannels, numSamples = targetSignalEstimate.shape 
    recSampleTimesInSeconds = arange(numSamples) / float(sampleRate)
    
    maxValue = max(abs(targetSignalEstimate)) * 1.05
    
    channelLabels = ['left', 'right']
    for channelIndex, signalEstimate in enumerate(targetSignalEstimate):
        subplot(1, numChannels, channelIndex+1)
        plot(recSampleTimesInSeconds, signalEstimate)
        title('%s, %s' % (titleString, channelLabels[channelIndex]))
        
        ylim( (-maxValue, maxValue) )
        if channelIndex == 0:
            ylabel('Frequency (kHz)')
        xlabel('Time (s)')
    
def describeMixtureSignal(stereoSamples, sampleRate):
    numChannels, numSamples = stereoSamples.shape
    
    print('Input mixture signal:')
    print('\tsampleRate: %d samples/sec' % sampleRate)
    print('\tnumChannels: %d' % numChannels)
    print('\tnumSamples: %d' % numSamples)
    print('\tdtype: %s' % str(stereoSamples.dtype))
    print('\tduration: %.2f seconds' % (numSamples / float(sampleRate)) )
    
def describeMixtureSpectrograms(windowSize, hopSize, windowFunction, complexMixtureSpectrogram):
    print('STFT:')
    print('\twindowSize: %d' % windowSize)
    print('\thopSize: %d' % hopSize)
    print('\twindowFunction: %s' % str(windowFunction))
    print('\tcomplexMixtureSpectrogram.shape = (numChannels, numFreq, numWindows): (%d, %d, %d)' % complexMixtureSpectrogram.shape )
    print('\tcomplexMixtureSpectrogram.dtype = %s' % (complexMixtureSpectrogram.dtype) )

def describeNMFDecomposition(V, W, H):
    numFreq, numTime = V.shape
    numFreq, numAtom = W.shape
    numAtom, numTime = H.shape
    
    print('Input V:\n    V.shape = (numFreq, numWindows): (%d, %d)' % (numFreq, numTime) )
    print('    V.dtype = %s' % V.dtype )
    print('Dictionary W:\n    W.shape = (numFreq, numAtoms): (%d, %d)' % (numFreq, numAtom) )
    print('    W.dtype = %s' % W.dtype )
    print('Coefficients H:\n    H.shape = (numAtoms, numWindows): (%d, %d)' % (numAtom, numTime) )
    print('    H.dtype = %s' % H.dtype )
