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
from os import listdir
from os.path import join, isdir
from collections import OrderedDict

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from gccNMF.realtime.gccNMFProcessor import TARGET_MODE_BOXCAR, TARGET_MODE_MULTIPLE, TARGET_MODE_WINDOW_FUNCTION

BUTTON_WIDTH = 50
        
class RealtimeGCCNMFInterfaceWindow(QtGui.QMainWindow):
    def __init__(self, audioPath, numTDOAs, gccPHATNLAlpha, gccPHATNLEnabled, dictionariesW, dictionarySize, dictionarySizes, dictionaryType, numHUpdates, localizationEnabled, localizationWindowSize,
                 gccPHATHistory, tdoaHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories,
                 togglePlayAudioProcessQueue, togglePlayAudioProcessAck,
                 togglePlayGCCNMFProcessQueue, togglePlayGCCNMFProcessAck,
                 tdoaParamsGCCNMFProcessQueue, tdoaParamsGCCNMFProcessAck,
                 toggleSeparationGCCNMFProcessQueue, toggleSeparationGCCNMFProcessAck):
        super(RealtimeGCCNMFInterfaceWindow, self).__init__()
        
        self.audioPath = audioPath
        logging.info('Loading interface with audio path: %s' % self.audioPath)
        self.initAudioFiles()
        
        self.numTDOAs = numTDOAs
        self.tdoaIndexes = np.arange(numTDOAs)
        self.dictionariesW = getVisualizedDictionariesW(dictionariesW)
        self.dictionaryTypes = self.dictionariesW.keys()
        
        self.dictionarySize = dictionarySize
        self.dictionarySizes = dictionarySizes
        self.dictionaryType = dictionaryType
        self.numHUpdates = numHUpdates
        self.targetTDOAIndex = self.numTDOAs / 2.0
        self.targetTDOAEpsilon = self.numTDOAs / 10.0
        self.gccPHATNLAlpha = gccPHATNLAlpha
        self.gccPHATNLEnabled = gccPHATNLEnabled
        self.localizationEnabled = localizationEnabled
        self.localizationWindowSize = localizationWindowSize
        
        self.gccPHATPlotTimer = QtCore.QTimer()
        self.gccPHATPlotTimer.timeout.connect(self.updateGCCPHATPlot)
        
        self.gccPHATHistory = gccPHATHistory
        self.gccPHATHistorySize = gccPHATHistory.size()
        self.tdoaHistory = tdoaHistory
        self.inputSpectrogramHistory = inputSpectrogramHistory
        self.outputSpectrogramHistory = outputSpectrogramHistory
        self.coefficientMaskHistories = coefficientMaskHistories
        
        self.togglePlayAudioProcessQueue = togglePlayAudioProcessQueue
        self.togglePlayAudioProcessAck = togglePlayAudioProcessAck
        self.togglePlayGCCNMFProcessQueue = togglePlayGCCNMFProcessQueue
        self.togglePlayGCCNMFProcessAck = togglePlayGCCNMFProcessAck
        self.tdoaParamsGCCNMFProcessQueue = tdoaParamsGCCNMFProcessQueue
        self.tdoaParamsGCCNMFProcessAck = tdoaParamsGCCNMFProcessAck
        self.toggleSeparationGCCNMFProcessQueue = toggleSeparationGCCNMFProcessQueue
        self.toggleSeparationGCCNMFProcessAck = toggleSeparationGCCNMFProcessAck
        
        self.playIconString = 'Play'
        self.pauseIconString = 'Pause'
        self.separationOffIconString = 'Disabled'
        self.separationOnIconString = 'Enabled'
        '''self.playIconString = u'\u23F5'
        self.pauseIconString = u'\u23F8'
        self.separationOffIconString = u'\u21F6 | \u21F6'
        self.separationOnIconString = u'\u21F6 | \u2192'''
        
        self.targetModeIconStrings = {TARGET_MODE_BOXCAR: u'\u168B',
                                      TARGET_MODE_MULTIPLE: u'\u168D',
                                      TARGET_MODE_WINDOW_FUNCTION: u'\u1109'}
        self.rollingImages = True
        
        self.initWindow()
        self.initControlWidgets()
        self.initVisualizationWidgets()
        self.initWindowLayout()
        
        self.localizationStateChanged()

        #self.show()
        self.showMaximized()
    
    def keyPressEvent(self, event):
        key = event.key()
        
        if key == QtCore.Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        if key == QtCore.Qt.Key_W and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.close()
        elif key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter or key == QtCore.Qt.Key_Space:
            self.togglePlay()
        elif key == QtCore.Qt.Key_F and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            if self.isFullScreen():
                self.showNormal()
                #self.showMaximized()
            else:
                self.showFullScreen()
        elif key == QtCore.Qt.Key_I and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.toggleInfoViews()
        elif key == QtCore.Qt.Key_1 and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.inputSpectrogramWidget.setVisible(self.inputSpectrogramWidget.isHidden())
        elif key == QtCore.Qt.Key_2 and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.outputSpectrogramWidget.setVisible(self.outputSpectrogramWidget.isHidden())
        elif key == QtCore.Qt.Key_3 and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.dictionaryWidget.setVisible(self.dictionaryWidget.isHidden())
        elif key == QtCore.Qt.Key_4 and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.coefficientMaskWidget.setVisible(self.coefficientMaskWidget.isHidden())
        elif key == QtCore.Qt.Key_5 and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.gccPHATHistoryWidget.setVisible(self.gccPHATHistoryWidget.isHidden())
        elif key == QtCore.Qt.Key_0 and QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.rollingImages = not self.rollingImages
            
        super(QtGui.QMainWindow, self).keyPressEvent(event)
        
    def closeEvent(self, event):
        logging.info('RealtimeGCCNMFInterfaceWindow: closing...')
        self.gccPHATPlotTimer.stop()
    
    def initAudioFiles(self):
        if isdir(self.audioPath):
            audioDirectory = self.audioPath
            self.audioFilePaths = [join(audioDirectory, fileName) for fileName in listdir(audioDirectory) if fileName.endswith('wav')]
        elif self.audioPath.endswith('.wav'):
            self.audioFilePaths = [self.audioPath]
        else:
            raise IOError('Unable to find wav files at: %s' % self.audioPath)
        self.selectedFileIndex = 0
        
    def initWindow(self):
        self.setWindowTitle('Real-time GCC-NMF')
        
        self.mainWidget = QtGui.QWidget()
        self.setCentralWidget(self.mainWidget)
        self.backgroundColor = self.mainWidget.palette().color(QtGui.QPalette.Background)
        self.borderColor = 'k'
        self.mainWidget.setStyleSheet('QSplitter::handle {image: url(images/notExists.png); background-color: #D8D8D8}')
        
        self.mainLayout = QtGui.QGridLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.mainWidget.setAutoFillBackground(True)
        p = QtGui.QPalette(self.mainWidget.palette())
        p.setColor(self.mainWidget.backgroundRole(), QtCore.Qt.black)
        self.mainWidget.setPalette(p)
        
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(1)
        
    def initWindowLayout(self):
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        
        self.infoLabelWidgets = []
        def addWidgetWithLabel(widget, label, fromRow, fromColumn, rowSpan=1, columnSpan=1):
            labeledWidget = QtGui.QWidget()
            widgetLayout = QtGui.QVBoxLayout()
            widgetLayout.setContentsMargins(0, 0, 0, 0)
            widgetLayout.setSpacing(1)
            labeledWidget.setLayout(widgetLayout)
            
            labelWidget = QtGui.QLabel(label)
            labelWidget.setContentsMargins(0, 3, 0, 1)
            labelWidget.setAutoFillBackground(True)
            labelWidget.setAlignment(QtCore.Qt.AlignCenter)
            #labelWidget.setStyleSheet('QLabel { border-top-width: 10px }')       
            self.infoLabelWidgets.append(labelWidget)
            
            widgetLayout.addWidget(labelWidget)
            widgetLayout.addWidget(widget)
            labeledWidget.setSizePolicy(sizePolicy)
            self.mainLayout.addWidget(labeledWidget, fromRow, fromColumn, rowSpan, columnSpan)
        
        addWidgetWithLabel(self.inputSpectrogramWidget, 'Input Spectrogram', 0, 1)
        addWidgetWithLabel(self.outputSpectrogramWidget, 'Output Spectrogram', 1, 1)
        addWidgetWithLabel(self.controlsWidget, 'GCC-NMF Masking Function', 0, 2)
        addWidgetWithLabel(self.gccPHATHistoryWidget, 'GCC PHAT Angular Spectrogram', 0, 3)
        addWidgetWithLabel(self.dictionaryWidget, 'NMF Dictionary', 1, 2)
        addWidgetWithLabel(self.coefficientMaskWidget, 'NMF Dictionary Mask', 1, 3)
        #for widget in self.infoLabelWidgets:
        #    widget.hide()
        #map(lambda widget: widget.hide(), self.infoLabelWidgets)
        
    def initControlWidgets(self):
        self.initMaskFunctionControls()
        self.initMaskFunctionPlot()
        self.initNMFControls()
        self.initLocalizationControls()
        self.initUIControls()
         
        controlWidgetsLayout = QtGui.QVBoxLayout()
        controlWidgetsLayout.addWidget(self.gccPHATPlotWidget)
        controlWidgetsLayout.addLayout(self.maskFunctionControlslayout)
        self.addSeparator(controlWidgetsLayout)
        controlWidgetsLayout.addLayout(self.nmfControlsLayout)
        controlWidgetsLayout.addLayout(self.localizationControlsLayout)
        self.addSeparator(controlWidgetsLayout)
        controlWidgetsLayout.addWidget(self.uiConrolsWidget)
        
        self.controlsWidget = QtGui.QWidget()
        self.controlsWidget.setLayout(controlWidgetsLayout)
        self.controlsWidget.setAutoFillBackground(True)
    
    def initMaskFunctionControls(self):
        self.maskFunctionControlslayout = QtGui.QHBoxLayout()
        labelsLayout = QtGui.QVBoxLayout()
        slidersLayout = QtGui.QVBoxLayout()
        self.maskFunctionControlslayout.addLayout(labelsLayout)
        self.maskFunctionControlslayout.addLayout(slidersLayout)
        def addSlider(label, changedFunction, minimum, maximum, value):
            labelWidget = QtGui.QLabel(label)
            labelsLayout.addWidget(labelWidget)
            slider = QtGui.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)
            slider.setValue(value)
            slider.sliderReleased.connect(changedFunction)
            slidersLayout.addWidget(slider)
            return slider, labelWidget
        
        self.targetModeWindowTDOASlider, self.targetModeWindowTDOALabel = addSlider('Center:', self.tdoaRegionChanged, 0, 100, 50)
        self.targetModeWindowWidthSlider, _ = addSlider('Width:', self.tdoaRegionChanged, 1, 101, 50)
        self.targetModeWindowBetaSlider, _ = addSlider('Shape:', self.tdoaRegionChanged, 0, 100, 50)
        self.targetModeWindowNoiseFloorSlider, _ = addSlider('Floor:', self.tdoaRegionChanged, 0, 100, 0)
        
    def initMaskFunctionPlot(self):
        self.gccPHATPlotWidget = self.createGraphicsLayoutWidget(self.backgroundColor, contentMargins=(6, 12, 18, 10))
        self.gccPHATPlotItem = self.gccPHATPlotWidget.addPlot()
        self.gccPHATPlotItem.getViewBox().setBackgroundColor((255, 255, 255, 150))
        self.gccPHATPlot = self.gccPHATPlotItem.plot()
        self.gccPHATPlot.setPen((0, 0, 0))
        self.gccPHATPlotItem.hideAxis('left')
        self.gccPHATPlotItem.hideAxis('bottom')
        self.gccPHATPlotItem.hideButtons()
        self.gccPHATPlotItem.setXRange(0, self.numTDOAs - 1)
        
        self.targetTDOARegion = pg.LinearRegionItem([self.targetTDOAIndex - self.targetTDOAEpsilon, self.targetTDOAIndex + self.targetTDOAEpsilon],
                                                    bounds=[0, self.numTDOAs - 1], movable=True)
        self.targetTDOARegion.sigRegionChangeFinished.connect(self.tdoaRegionChanged)
        
        self.targetWindowFunctionPen = pg.mkPen((0, 0, 204, 255), width=2)  # , style=QtCore.Qt.DashLine)
        self.targetWindowFunctionPlot = TargetWindowFunctionPlot(self.targetTDOARegion, self.targetModeWindowTDOASlider, self.targetModeWindowBetaSlider, self.targetModeWindowNoiseFloorSlider, self.targetModeWindowWidthSlider, self.numTDOAs, pen=self.targetWindowFunctionPen)
        self.gccPHATPlotItem.addItem(self.targetWindowFunctionPlot)
        self.targetWindowFunctionPlot.updateData()

    def initNMFControls(self):
        self.nmfControlsLayout = QtGui.QHBoxLayout()
        self.nmfControlsLayout.addStretch(1)
        self.nmfControlsLayout.addWidget(QtGui.QLabel('Dictionary Size:'))
        self.dictionarySizeDropDown = QtGui.QComboBox()
        for dictionarySize in self.dictionarySizes:
            self.dictionarySizeDropDown.addItem( str(dictionarySize) )
        self.dictionarySizeDropDown.setMaximumWidth(75)
        self.dictionarySizeDropDown.setCurrentIndex(self.dictionarySizes.index(self.dictionarySize))
        self.dictionarySizeDropDown.currentIndexChanged.connect(self.dictionarySizeChanged)
        self.nmfControlsLayout.addWidget(self.dictionarySizeDropDown)
        self.nmfControlsLayout.addStretch(1)
        
        self.nmfControlsLayout.addWidget(QtGui.QLabel('Num Updates:'))
        self.numHUpdatesSpinBox = QtGui.QSpinBox()
        self.nmfControlsLayout.addWidget(self.numHUpdatesSpinBox)
        self.nmfControlsLayout.addStretch(1)
    
    def initLocalizationControls(self):
        self.localizationControlsLayout = QtGui.QHBoxLayout()
        self.localizationControlsLayout.addStretch(3)
        self.localizationCheckBox = QtGui.QCheckBox('Enable Localization')
        self.localizationCheckBox.setChecked(self.localizationEnabled)
        self.localizationCheckBox.stateChanged.connect(self.localizationStateChanged)
        self.localizationControlsLayout.addWidget(self.localizationCheckBox)

        self.localizationControlsLayout.addStretch(1)
        self.localizationWindowSizeLabel = QtGui.QLabel('Sliding Window Size:')
        self.localizationControlsLayout.addWidget(self.localizationWindowSizeLabel)
        self.localziaitonWindowSizeSpinBox = QtGui.QSpinBox()
        self.localziaitonWindowSizeSpinBox.setMinimum(1)
        self.localziaitonWindowSizeSpinBox.setMaximum(128)
        self.localziaitonWindowSizeSpinBox.setValue(self.localizationWindowSize)
        self.localziaitonWindowSizeSpinBox.valueChanged.connect(self.localizationParamsChanged)
        self.localizationControlsLayout.addWidget(self.localziaitonWindowSizeSpinBox)
        self.localizationControlsLayout.addStretch(3)

    def initUIControls(self):
        self.uiConrolsWidget = QtGui.QWidget()
        buttonBarWidgetLayout = QtGui.QHBoxLayout(spacing=0)
        buttonBarWidgetLayout.setContentsMargins(0, 0, 0, 0)
        buttonBarWidgetLayout.setSpacing(0)
        self.uiConrolsWidget.setLayout(buttonBarWidgetLayout)
        
        def addButton(label, widget=None, function=None):
            button = QtGui.QPushButton(label)
            if function is None:
                button.clicked.connect(lambda: widget.setVisible(widget.isHidden()))
            else:
                button.clicked.connect(function)
            button.setStyleSheet('QPushButton {'
                                 'border-color: black;'
                                 'border-width: 5px;}')
            buttonBarWidgetLayout.addWidget(button)
            return button

        addButton('Info', function=self.toggleInfoViews)
        self.toggleSeparationButton = addButton(self.separationOnIconString, function=self.toggleSeparation)
        self.playPauseButton = addButton(self.playIconString, function=self.togglePlay)

    def initVisualizationWidgets(self):
        self.inputSpectrogramWidget = self.createGraphicsLayoutWidget(self.backgroundColor)
        inputSpectrogramViewBox = self.inputSpectrogramWidget.addViewBox()
        self.inputSpectrogramHistoryImageItem = pg.ImageItem(self.inputSpectrogramHistory.values)  # , border=self.borderColor)
        inputSpectrogramViewBox.addItem(self.inputSpectrogramHistoryImageItem)
        inputSpectrogramViewBox.setRange(xRange=(0, self.inputSpectrogramHistory.values.shape[1]), yRange=(0, self.inputSpectrogramHistory.values.shape[0]), padding=0)
        
        self.outputSpectrogramWidget = self.createGraphicsLayoutWidget(self.backgroundColor)
        outputSpectrogramViewBox = self.outputSpectrogramWidget.addViewBox()
        self.outputSpectrogramHistoryImageItem = pg.ImageItem(self.outputSpectrogramHistory.values)  # , border=self.borderColor)
        outputSpectrogramViewBox.addItem(self.outputSpectrogramHistoryImageItem)
        outputSpectrogramViewBox.setRange(xRange=(0, self.outputSpectrogramHistory.values.shape[1] - 1), yRange=(0, self.outputSpectrogramHistory.values.shape[0] - 1), padding=0)
        
        self.gccPHATHistoryWidget = self.createGraphicsLayoutWidget(self.backgroundColor)
        gccPHATHistoryViewBox = self.gccPHATHistoryWidget.addViewBox()  # invertY=True)
        self.gccPHATImageItem = pg.ImageItem(self.gccPHATHistory.values)  # , border=self.borderColor)
        gccPHATHistoryViewBox.addItem(self.gccPHATImageItem)
        gccPHATHistoryViewBox.setRange(xRange=(0, self.gccPHATHistory.values.shape[1] - 1), yRange=(0, self.gccPHATHistory.values.shape[0] - 1), padding=0)
        
        self.tdoaPlotDataItem = pg.PlotDataItem( pen=pg.mkPen((255, 0, 0, 255), width=4) )
        gccPHATHistoryViewBox.addItem(self.tdoaPlotDataItem)

        dictionarySize = self.dictionarySizes[self.dictionarySizeDropDown.currentIndex()]
        self.coefficientMaskWidget = self.createGraphicsLayoutWidget(self.backgroundColor)
        self.coefficientMaskViewBox = self.coefficientMaskWidget.addViewBox()
        self.coefficientMaskHistory = self.coefficientMaskHistories[dictionarySize]
        self.coefficientMaskHistoryImageItem = pg.ImageItem()  # , border=self.borderColor)
        self.coefficientMaskViewBox.addItem(self.coefficientMaskHistoryImageItem)
        
        self.dictionaryWidget = self.createGraphicsLayoutWidget(self.backgroundColor)
        self.dictionaryViewBox = self.dictionaryWidget.addViewBox()
        self.dictionaryImageItem = pg.ImageItem()  # 1 - visualizedDictionary)#, border=self.borderColor)
        self.dictionaryViewBox.addItem(self.dictionaryImageItem)
        self.dictionarySizeChanged(False)

    def addSeparator(self, layout, lineStyle=QtGui.QFrame.HLine):
        separator = QtGui.QFrame()
        separator.setFrameShape(lineStyle)
        separator.setFrameShadow(QtGui.QFrame.Sunken)
        layout.addWidget(separator)    
        
    def createGraphicsLayoutWidget(self, backgroundColor, border=None, contentMargins=(0, 0, 0, 0)):
        graphicsLayoutWidget = pg.GraphicsLayoutWidget(border=border)
        graphicsLayoutWidget.setBackground(backgroundColor)
        graphicsLayoutWidget.ci.layout.setContentsMargins(*contentMargins)
        graphicsLayoutWidget.ci.layout.setSpacing(0)
        return graphicsLayoutWidget
    
    def updateGCCPHATPlot(self):
        gccPHATValues = np.squeeze(np.mean(self.gccPHATHistory.values, axis=-1))
        gccPHATValues -= min(gccPHATValues)
        gccPHATValues /= max(gccPHATValues)
        self.gccPHATPlot.setData(y=gccPHATValues)
        if self.rollingImages:
            self.gccPHATImageItem.setImage(-self.gccPHATHistory.getUnraveledArray().T)
            self.tdoaPlotDataItem.setData(y=self.tdoaHistory.getUnraveledArray()[0])
            self.inputSpectrogramHistoryImageItem.setImage(self.inputSpectrogramHistory.getUnraveledArray().T)
            self.outputSpectrogramHistoryImageItem.setImage(self.outputSpectrogramHistory.getUnraveledArray().T)
            self.coefficientMaskHistoryImageItem.setImage(self.coefficientMaskHistory.getUnraveledArray().T, levels=[0, 1])
        else:
            self.gccPHATImageItem.setImage(-self.gccPHATHistory.values.T)
            self.tdoaPlotDataItem.setData(y=self.tdoaHistory.values[0])
            self.inputSpectrogramHistoryImageItem.setImage(self.inputSpectrogramHistory.values.T)
            self.outputSpectrogramHistoryImageItem.setImage(self.outputSpectrogramHistory.values.T)
            self.coefficientMaskHistoryImageItem.setImage(self.coefficientMaskHistory.values.T, levels=[0, 1])
        
        if self.localizationCheckBox.isChecked():
            sliderValue = self.tdoaHistory.get()[0] / (self.numTDOAs-1) * 100
            self.targetModeWindowTDOASlider.setValue(sliderValue)
        
    def toggleInfoViews(self):
        isHidden = self.infoLabelWidgets[0].isHidden()
        for view in self.infoLabelWidgets:
            view.setVisible(isHidden)
        #map(lambda view: view.setVisible(isHidden), self.infoLabelWidgets) 
        
    def togglePlay(self):
        playing = self.playPauseButton.text() == self.playIconString
        logging.info('GCCNMFInterface: setting playing: %s' % playing)
        
        self.playPauseButton.setText(self.pauseIconString if playing else self.playIconString)
        
        if playing:
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.tdoaRegionChanged()
            self.updateTogglePlayParamsGCCNMFProcess()
            self.updateTogglePlayParamsAudioProcess(playing)
            QtGui.QApplication.restoreOverrideCursor()
        else:
            self.updateTogglePlayParamsAudioProcess(playing)

        self.gccPHATPlotTimer.start(100) if playing else self.gccPHATPlotTimer.stop()
    
    def toggleSeparation(self):
        separationEnabled = self.toggleSeparationButton.text() == self.separationOffIconString
        logging.info('GCCNMFInterface: toggleSeparation(): now %s' % separationEnabled)
        
        self.toggleSeparationButton.setText(self.separationOnIconString if separationEnabled else self.separationOffIconString)
        self.queueParams(self.toggleSeparationGCCNMFProcessQueue,
                         self.toggleSeparationGCCNMFProcessAck,
                         {'separationEnabled': separationEnabled},
                         'separationEnabledParameters')
        
    def numHUpdatesChanged(self):
        numHUpdates = int(self.numHUpdatesTextBox.text())
        logging.info('GCCNMFInterface: setting numHUpdates: %d' % numHUpdates)
        
        self.queueParams(self.togglePlayGCCNMFProcessQueue,
                         self.togglePlayGCCNMFProcessAck,
                         {'numHUpdates': numHUpdates},
                         'gccNMFProcessTogglePlayParameters')

    def updateFileNameAudioProcess(self):
        self.queueParams(self.togglePlayAudioProcessQueue,
                         self.togglePlayAudioProcessAck,
                         {'fileName': self.selectedFilePath},
                         'audioProcessParameters')
      
    def updateTogglePlayParamsAudioProcess(self, playing):
        self.queueParams(self.togglePlayAudioProcessQueue,
                         self.togglePlayAudioProcessAck,
                         {'fileName': self.audioFilePaths[self.selectedFileIndex],
                          'start' if playing else 'stop': ''},
                         'audioProcessParameters')

    def updateTogglePlayParamsGCCNMFProcess(self):
        self.queueParams(self.togglePlayGCCNMFProcessQueue,
                         self.togglePlayGCCNMFProcessAck,
                         {'numTDOAs': self.numTDOAs,
                          'dictionarySize': int(self.dictionarySizeDropDown.currentText())},
                          'gccNMFProcessTogglePlayParameters')
        
    def tdoaRegionChanged(self):
        self.queueParams(self.tdoaParamsGCCNMFProcessQueue,
                         self.tdoaParamsGCCNMFProcessAck,
                         {'targetTDOAIndex': self.targetWindowFunctionPlot.getTDOA(),
                          'targetTDOAEpsilon': self.targetWindowFunctionPlot.getWindowWidth(),  # targetTDOAEpsilon,
                          'targetTDOABeta': self.targetWindowFunctionPlot.getBeta(),
                          'targetTDOANoiseFloor': self.targetWindowFunctionPlot.getNoiseFloor()},
                         'gccNMFProcessTDOAParameters (region)')
        self.targetWindowFunctionPlot.updateData()

    def localizationParamsChanged(self):
        self.queueParams(self.tdoaParamsGCCNMFProcessQueue,
                         self.tdoaParamsGCCNMFProcessAck,
                         {'localizationEnabled': self.localizationCheckBox.isChecked(),
                          'localizationWindowSize': int(self.localziaitonWindowSizeSpinBox.value())},
                         'gccNMFProcessTDOAParameters (localization)')
        
    def dictionarySizeChanged(self, changeGCCNMFProcessor=True):
        self.dictionarySize = self.dictionarySizes[self.dictionarySizeDropDown.currentIndex()]
        logging.info('GCCNMFInterface: setting dictionarySize: %d' % self.dictionarySize)
        
        visualizedDictionary = self.dictionariesW[self.dictionaryType][self.dictionarySize]
        self.dictionaryImageItem.setImage(visualizedDictionary)
        self.dictionaryViewBox.setXRange(0, visualizedDictionary.shape[0] - 1, padding=0)
        self.dictionaryViewBox.setYRange(0, visualizedDictionary.shape[1] - 1, padding=0)
        
        self.coefficientMaskHistory = self.coefficientMaskHistories[self.dictionarySize]
        self.coefficientMaskViewBox.setXRange(0, self.coefficientMaskHistory.values.shape[1] - 1, padding=0)
        self.coefficientMaskViewBox.setYRange(0, self.coefficientMaskHistory.values.shape[0] - 1, padding=0)
        
        if changeGCCNMFProcessor:
            self.queueParams(self.togglePlayGCCNMFProcessQueue,
                             self.togglePlayGCCNMFProcessAck,
                             {'dictionarySize': self.dictionarySize},
                             'gccNMFProcessTogglePlayParameters')

    def dictionaryTypeChanged(self):
        dictionaryType = self.dictionaryTypes[self.dictionaryTypeDropDown.currentIndex()]
        logging.info('GCCNMFInterface: setting dictionarySize: %s' % dictionaryType)
        
        self.queueParams(self.togglePlayGCCNMFProcessQueue,
                         self.togglePlayGCCNMFProcessAck,
                         {'dictionaryType': dictionaryType},
                         'gccNMFProcessTogglePlayParameters')

    def localizationStateChanged(self):
        onlineLocalizationEnabled = self.localizationCheckBox.isChecked()
        self.targetModeWindowTDOASlider.setEnabled(not onlineLocalizationEnabled)
        self.targetModeWindowTDOALabel.setEnabled(not onlineLocalizationEnabled)
        self.localziaitonWindowSizeSpinBox.setEnabled(onlineLocalizationEnabled)
        self.localizationWindowSizeLabel.setEnabled(onlineLocalizationEnabled)

        self.localizationParamsChanged()
    
    def queueParams(self, queue, ack, params, label='params'):
        ack.clear()
        logging.debug('GCCNMFInterface: putting %s' % label)
        queue.put(params)
        logging.debug('GCCNMFInterface: put %s' % label)
        ack.wait()
        logging.debug('GCCNMFInterface: ack received')
        
def generalizedGaussian(x, alpha, beta, mu):
    return np.exp( - (np.abs(x-mu) / alpha) ** beta )
        
class TargetWindowFunctionPlot(pg.PlotDataItem):
    def __init__(self, tdoaRegionItem, targetModeWindowTDOASlider, targetModeWindowBetaSlider, targetModeWindowNoiseFloorSlider, targetModeWindowWidthSlider, numTDOAs, *args, **kwargs):
        super(TargetWindowFunctionPlot, self).__init__(*args, **kwargs)
        
        self.tdoaRegionItem = tdoaRegionItem
        self.targetModeWindowTDOASlider = targetModeWindowTDOASlider
        self.targetModeWindowBetaSlider = targetModeWindowBetaSlider
        self.targetModeWindowNoiseFloorSlider = targetModeWindowNoiseFloorSlider
        self.targetModeWindowWidthSlider = targetModeWindowWidthSlider

        self.targetModeWindowTDOASlider.valueChanged.connect(self.updateData)
        self.targetModeWindowBetaSlider.valueChanged.connect(self.updateData)
        self.targetModeWindowNoiseFloorSlider.valueChanged.connect(self.updateData)
        self.targetModeWindowWidthSlider.valueChanged.connect(self.updateData)
        self.numTDOAs = numTDOAs
        self.tdoas = np.arange(self.numTDOAs).astype(np.float32)
        
    def updateData(self):
        mu = self.getTDOA()
        alpha = self.getWindowWidth()
        beta = self.getBeta()
        noiseFloor = self.getNoiseFloor()
        data = generalizedGaussian(self.tdoas, alpha, beta, mu)
        data -= min(data)
        data = data / max(data) * (1 - noiseFloor) + noiseFloor 
        self.setData(self.tdoas, data)
    
    def getBeta(self):
        lnBeta = self.targetModeWindowBetaSlider.value() / 100.0
        lnBeta *= 10.0
        lnBeta -= 5.0
        beta = np.exp(lnBeta)
        return beta
    
    def getNoiseFloor(self):
        noiseFloorValue = self.targetModeWindowNoiseFloorSlider.value() / 100.0
        return noiseFloorValue
    
    def getWindowWidth(self):
        windowWidth = self.targetModeWindowWidthSlider.value() / 100.0 * self.numTDOAs
        return windowWidth
    
    def getTDOA(self):
        tdoa = self.targetModeWindowTDOASlider.value() / 100.0 * self.numTDOAs
        return tdoa
    
def getVisualizedDictionariesW(dictionariesW):
    visualizedDictionariesW = OrderedDict()
    for dictionaryType, dictionaries in dictionariesW.items():
        currentDictionaries = OrderedDict()
        for dictionarySize, dictionary in dictionaries.items():
            visualizedDictionary = dictionary.copy()
            visualizedDictionary /= np.max(visualizedDictionary)
            visualizedDictionary **= (1 / 3.0)
            visualizedDictionary = 1 - visualizedDictionary
            currentDictionaries[dictionarySize] = visualizedDictionary
        visualizedDictionariesW[dictionaryType] = currentDictionaries
    return visualizedDictionariesW
