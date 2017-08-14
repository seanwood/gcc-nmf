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
from os.path import join, isdir, abspath
from collections import OrderedDict
import platform

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from gccNMF.realtime.gccNMFProcessor import TARGET_MODE_BOXCAR, TARGET_MODE_MULTIPLE, TARGET_MODE_WINDOW_FUNCTION

CONTINUOUS_SLIDERS = True
INFO_HIDDEN_ON_STARTUP = True

# Icons from https://icons8.com
RESOURCES_DIR = join( abspath( join( __file__, '../resources' ) ))
PLAY_ON_ICON_PATH = join(RESOURCES_DIR, 'play-on.png')
PLAY_OFF_ICON_PATH = join(RESOURCES_DIR, 'play-off.png')
FORWARD_ICON_PATH = join(RESOURCES_DIR, 'forward-button.png')
BACK_ICON_PATH = join(RESOURCES_DIR, 'back-button.png')
SEPARATION_ON_ICON_PATH = join(RESOURCES_DIR, 'separation-on.png')
SEPARATION_OFF_ICON_PATH = join(RESOURCES_DIR, 'separation-off.png')
INFO_ON_ICON_PATH = join(RESOURCES_DIR, 'info-on.png')
INFO_OFF_ICON_PATH = join(RESOURCES_DIR, 'info-off.png')
VOLUME_ICON_PATH = join(RESOURCES_DIR, 'volume-icon.png')
        
class RealtimeGCCNMFInterfaceWindow(QtGui.QMainWindow):
    def __init__(self, params, audioPath, numTDOAs, gccPHATNLAlpha, gccPHATNLEnabled, dictionariesW, dictionarySize, dictionarySizes, dictionaryType, numHUpdates,
                 gccPHATHistory, inputSpectrogramHistory, outputSpectrogramHistory, coefficientMaskHistories,
                 audioPlayingFlag, paramsNamespace, gccNMFParams, gccNMFDirtyParamNames):
        super(RealtimeGCCNMFInterfaceWindow, self).__init__()
        
        self.params = params
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
        
        self.gccPHATPlotTimer = QtCore.QTimer()
        self.gccPHATPlotTimer.timeout.connect(self.updateGCCPHATPlot)
        
        self.gccPHATHistory = gccPHATHistory
        self.gccPHATHistorySize = gccPHATHistory.size()
        self.inputSpectrogramHistory = inputSpectrogramHistory
        self.outputSpectrogramHistory = outputSpectrogramHistory
        self.coefficientMaskHistories = coefficientMaskHistories
        
        self.audioPlayingFlag = audioPlayingFlag
        self.paramsNamespace = paramsNamespace
        self.gccNMFParams = gccNMFParams
        self.gccNMFDirtyParamNames = gccNMFDirtyParamNames
        
        self.targetModeIconStrings = {TARGET_MODE_BOXCAR: u'\u168B',
                                      TARGET_MODE_MULTIPLE: u'\u168D',
                                      TARGET_MODE_WINDOW_FUNCTION: u'\u1109'}
        self.rollingImages = True
        
        self.initWindow()
        self.initControlWidgets()
        self.initVisualizationWidgets()
        self.initWindowLayout()
        
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        
        if params.startupWindowMode == 'normal':
            self.show()
        elif params.startupWindowMode == 'maximized':
            self.showMaximized()
        elif params.startupWindowMode == 'fullscreen':
            delay = params.linuxFullscreenDelay
            if platform.system().lower() == 'linux' and delay != 0:
                self.show()
                QtCore.QTimer.singleShot(delay, self.showFullScreen)
            else:
                self.showFullScreen()
                self.raise_()

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
            self.playPauseButton.setChecked( not self.playPauseButton.isChecked() )
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
        
        fontSize = self.getControlFontSize()
        self.infoLabelWidgets = []
        def addWidgetWithLabel(widget, label, fromRow, fromColumn, rowSpan=1, columnSpan=1):
            labeledWidget = QtGui.QWidget()
            widgetLayout = QtGui.QVBoxLayout()
            widgetLayout.setContentsMargins(0, 0, 0, 0)
            widgetLayout.setSpacing(1)
            labeledWidget.setLayout(widgetLayout)
            
            labelWidget = QtGui.QLabel(label)
            labelWidget.setStyleSheet('font: %dpt;' % fontSize)
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
        if INFO_HIDDEN_ON_STARTUP:
            #map(lambda widget: widget.hide(), self.infoLabelWidgets)
            for widget in self.infoLabelWidgets:
                widget.hide()
        
    def initControlWidgets(self):
        self.initMaskFunctionControls()
        self.initMaskFunctionPlot()
        self.initNMFControls()
        self.initUIControls()
         
        mainControlsLayout = QtGui.QGridLayout()
        mainControlsLayout.setContentsMargins(0, 0, 0, 0)
        mainControlsLayout.setSpacing(0)
        mainControlsLayout.addWidget(self.gccPHATPlotWidget, 0, 0, 1, 1) #fromRow, fromColumn, rowSpan, columnSpan
        
        controlsWidget = QtGui.QWidget()
        controlWidgetsLayout = QtGui.QVBoxLayout()
        controlsWidget.setLayout(controlWidgetsLayout)
        controlWidgetsLayout.addLayout(self.maskFunctionControlslayout)
        controlWidgetsLayout.addWidget(self.uiConrolsWidget)
        mainControlsLayout.addWidget(controlsWidget, 1, 0, 1, 1) #fromRow, fromColumn, rowSpan, columnSpan
        
        self.controlsWidget = QtGui.QWidget()
        self.controlsWidget.setLayout(mainControlsLayout)
        self.controlsWidget.setAutoFillBackground(True)
    
    def initMaskFunctionControls(self):
        self.maskFunctionControlslayout = QtGui.QHBoxLayout()
        labelsLayout = QtGui.QVBoxLayout()
        slidersLayout = QtGui.QVBoxLayout()
        self.maskFunctionControlslayout.addLayout(labelsLayout)
        self.maskFunctionControlslayout.addLayout(slidersLayout)
        
        sliderHeight = self.getControlHeight()
        fontSize = self.getControlFontSize()
        def addSlider(label, minimum, maximum, value, iconPath=None):
            labelWidget = QtGui.QLabel(label)
            labelWidget.setStyleSheet('font:%dpt;' % fontSize)
            if iconPath:
                labelWidget.setPixmap( QtGui.QPixmap(iconPath) )
            labelsLayout.addWidget(labelWidget)

            slider = QtGui.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)
            slider.setValue(value)
            slider.setStyleSheet("QSlider::groove:horizontal { "
                      "border: 1px solid #999999; "
                      "height: %dpx; "
                      "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4); "
                      "margin: 2px 0; "
                      "} "
                      "QSlider::handle:horizontal { "
                      "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f); "
                      "border: 1px solid #5c5c5c; "
                      "width: 70px; "
                      "margin: -2px 0px; "
                      "} " % sliderHeight )
            slider.setMinimumSize(50, sliderHeight+5)
            slidersLayout.addWidget(slider)
            return slider

        self.targetModeWindowTDOASlider = addSlider('Center', 0, 100, 50)
        self.targetModeWindowWidthSlider = addSlider('Width', 1, 101, 50)
        self.targetModeWindowBetaSlider = addSlider('Shape', 25, 100, 50)
        self.targetModeWindowNoiseFloorSlider = addSlider('Floor', 0, 100, 0)
        self.audioPlaybackGainSlider = addSlider('', 0, 100, 100, iconPath=VOLUME_ICON_PATH)
        
    def initMaskFunctionPlot(self):
        self.gccPHATPlotWidget = self.createGraphicsLayoutWidget(self.backgroundColor, contentMargins=(0, 0, 0, 0))
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
        #self.targetTDOARegion.sigRegionChangeFinished.connect(self.tdoaRegionChanged)
        
        self.targetWindowFunctionPen = pg.mkPen((0, 0, 204, 255), width=2)  # , style=QtCore.Qt.DashLine)
        self.targetWindowFunctionPlot = TargetWindowFunctionPlot(self.targetTDOARegion, self.targetModeWindowTDOASlider, self.targetModeWindowBetaSlider, self.targetModeWindowNoiseFloorSlider, self.targetModeWindowWidthSlider, self.numTDOAs, pen=self.targetWindowFunctionPen)
        self.gccPHATPlotItem.addItem(self.targetWindowFunctionPlot)
        self.targetWindowFunctionPlot.updateData()
        
        def buildVariableChangedFunction(variableName, getValue):
            def variableChanged():
                logging.info('GCCNMFProcessor: %s changed, now: %s' % (variableName, getValue()))
                setattr(self.gccNMFParams, variableName, getValue())
                self.gccNMFDirtyParamNames.append(variableName)
            return variableChanged
        
        if CONTINUOUS_SLIDERS:
            self.targetModeWindowTDOASlider.valueChanged.connect( buildVariableChangedFunction('targetTDOAIndex', self.targetWindowFunctionPlot.getTDOA) )
            self.targetModeWindowWidthSlider.valueChanged.connect( buildVariableChangedFunction('targetTDOAEpsilon', self.targetWindowFunctionPlot.getWindowWidth) )
            self.targetModeWindowBetaSlider.valueChanged.connect( buildVariableChangedFunction('targetTDOABeta', self.targetWindowFunctionPlot.getBeta) )
            self.targetModeWindowNoiseFloorSlider.valueChanged.connect( buildVariableChangedFunction('targetTDOANoiseFloor', self.targetWindowFunctionPlot.getNoiseFloor) )
            self.audioPlaybackGainSlider.valueChanged.connect( buildVariableChangedFunction('audioPlaybackGain', self.getAudioPlaybackGain) )
        else:
            self.targetModeWindowTDOASlider.sliderReleased.connect( buildVariableChangedFunction('targetTDOAIndex', self.targetWindowFunctionPlot.getTDOA) )
            self.targetModeWindowWidthSlider.sliderReleased.connect( buildVariableChangedFunction('targetTDOAEpsilon', self.targetWindowFunctionPlot.getWindowWidth) )
            self.targetModeWindowBetaSlider.sliderReleased.connect( buildVariableChangedFunction('targetTDOABeta', self.targetWindowFunctionPlot.getBeta) )
            self.targetModeWindowNoiseFloorSlider.sliderReleased.connect( buildVariableChangedFunction('targetTDOANoiseFloor', self.targetWindowFunctionPlot.getNoiseFloor) )
            self.audioPlaybackGainSlider.sliderReleased.connect( buildVariableChangedFunction('audioPlaybackGain', self.getAudioPlaybackGain) )

    def initNMFControls(self):
        controlHeight = self.getControlHeight()
        fontSize = self.getControlFontSize()
        
        self.nmfControlsLayout = QtGui.QHBoxLayout()
        #self.nmfControlsLayout.addStretch(1)
        labelWidget = QtGui.QLabel('NMF\nAtoms')
        labelWidget.setStyleSheet('font:%dpt;' % fontSize)
        self.nmfControlsLayout.addWidget(labelWidget)
        
        self.dictionarySizeDropDown = QtGui.QComboBox()
        self.dictionarySizeDropDown.setStyleSheet("QComboBox {font-size: %dpt;}" % fontSize)
        self.dictionarySizeDropDown.setMinimumWidth(100)
        
        for dictionarySize in self.dictionarySizes:
            self.dictionarySizeDropDown.addItem( str(dictionarySize) )
        self.dictionarySizeDropDown.setMaximumWidth(75)
        self.dictionarySizeDropDown.setMinimumHeight(65)
        self.dictionarySizeDropDown.setCurrentIndex(self.dictionarySizes.index(self.dictionarySize))
        self.dictionarySizeDropDown.currentIndexChanged.connect(self.dictionarySizeChanged)
        self.nmfControlsLayout.addSpacing(10)
        #self.nmfControlsLayout.addStretch(1)
        self.nmfControlsLayout.addWidget(self.dictionarySizeDropDown)
        
        #self.nmfControlsLayout.addWidget(QtGui.QLabel('Num Updates:'))
        #self.numHUpdatesSpinBox = QtGui.QSpinBox()
        #self.nmfControlsLayout.addWidget(self.numHUpdatesSpinBox)
        #self.nmfControlsLayout.addStretch(1)
    
    def initUIControls(self):
        self.uiConrolsWidget = QtGui.QWidget()
        buttonBarWidgetLayout = QtGui.QHBoxLayout(spacing=0)
        buttonBarWidgetLayout.setContentsMargins(0, 0, 0, 0)
        buttonBarWidgetLayout.setSpacing(0)
        self.uiConrolsWidget.setLayout(buttonBarWidgetLayout)
        
        buttonHeight = self.getControlHeight()
        fontSize = self.getControlFontSize()
        def addButton(label=None, onIconPath=None, offIconPath=None, iconSize=None, widget=None, function=None):
            button = QtGui.QPushButton(label) if label else QtGui.QPushButton() 
            
            if function is None:
                button.clicked.connect(lambda: widget.setVisible(widget.isHidden()))
            else:
                button.clicked.connect(function)
            button.setStyleSheet('QPushButton {'
                                 'border-color: black;'
                                 'border-width: 5px;'
                                 'font: %dpt;}' % fontSize)
            if onIconPath and offIconPath:
                icon = QtGui.QIcon()
                icon.addPixmap( QtGui.QPixmap(onIconPath), QtGui.QIcon.Normal, QtGui.QIcon.Off )
                icon.addPixmap( QtGui.QPixmap(offIconPath), QtGui.QIcon.Normal, QtGui.QIcon.On )
                button.setIcon(icon)
                if iconSize:
                    button.setIconSize( QtCore.QSize(iconSize,iconSize) )
                button.setCheckable(True)
            elif onIconPath:
                icon = QtGui.QIcon(onIconPath)
                button.setIcon(icon)
                if iconSize:
                    button.setIconSize( QtCore.QSize(iconSize,iconSize) )
                
            button.setMinimumHeight(buttonHeight)
            buttonBarWidgetLayout.addWidget(button)
            return button
            
        buttonBarWidgetLayout.addLayout(self.nmfControlsLayout)
        
        buttonBarWidgetLayout.addStretch(2)
        self.toggleSeparationButton = addButton(onIconPath=SEPARATION_ON_ICON_PATH, offIconPath=SEPARATION_OFF_ICON_PATH, function=self.toggleSeparation, iconSize=80)
        self.playPauseButton = addButton(onIconPath=PLAY_ON_ICON_PATH, offIconPath=PLAY_OFF_ICON_PATH, function=self.togglePlay, iconSize=80)
        
        buttonBarWidgetLayout.addStretch(1)
        self.backButton = addButton(onIconPath=BACK_ICON_PATH, function=self.previousFile, iconSize=60)
        self.forwardButton = addButton(onIconPath=FORWARD_ICON_PATH, function=self.nextFile, iconSize=60)
        
        buttonBarWidgetLayout.addStretch(1)
        self.infoButton = addButton(onIconPath=INFO_ON_ICON_PATH, offIconPath=INFO_OFF_ICON_PATH, function=self.toggleInfoViews, iconSize=60)

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

    def getAudioPlaybackGain(self):
        value = self.audioPlaybackGainSlider.value() / 100.0
        return value

    def getControlHeight(self):
        app = QtGui.QApplication.instance()
        screenHeight = app.desktop().screenGeometry().height()
        controlHeight = int(screenHeight * 0.05)
        return controlHeight

    def getControlFontSize(self):
        app = QtGui.QApplication.instance()
        screenHeight = app.desktop().screenGeometry().height()
        fontSize = int(screenHeight * 0.0225)
        return fontSize

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
            self.inputSpectrogramHistoryImageItem.setImage(self.inputSpectrogramHistory.getUnraveledArray().T)
            self.outputSpectrogramHistoryImageItem.setImage(self.outputSpectrogramHistory.getUnraveledArray().T)
            self.coefficientMaskHistoryImageItem.setImage(self.coefficientMaskHistory.getUnraveledArray().T, levels=[0, 1])
        else:
            self.gccPHATImageItem.setImage(-self.gccPHATHistory.values.T)
            self.inputSpectrogramHistoryImageItem.setImage(self.inputSpectrogramHistory.values.T)
            self.outputSpectrogramHistoryImageItem.setImage(self.outputSpectrogramHistory.values.T)
            self.coefficientMaskHistoryImageItem.setImage(self.coefficientMaskHistory.values.T, levels=[0, 1])
        
    def toggleInfoViews(self):
        infoEnabled = self.infoButton.isChecked()
        for view in self.infoLabelWidgets:
            view.setVisible(infoEnabled)
        #map(lambda view: view.setVisible(isHidden), self.infoLabelWidgets) 
        
    def togglePlay(self):
        playing = self.playPauseButton.isChecked() #self.playPauseButton.text() == self.playIconString
        logging.info('GCCNMFInterface: setting playing: %s' % playing)
        
        if playing:
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.paramsNamespace.fileName = self.audioFilePaths[self.selectedFileIndex]
            self.audioPlayingFlag.value = playing
            QtGui.QApplication.restoreOverrideCursor()
        else:
            self.audioPlayingFlag.value = playing
        
        self.gccPHATPlotTimer.start(25) if playing else self.gccPHATPlotTimer.stop()
    
    def toggleSeparation(self):
        separationEnabled = self.toggleSeparationButton.isChecked() #self.toggleSeparationButton.text() == self.separationOffIconString
        logging.info('GCCNMFInterface: toggleSeparation(): now %s' % separationEnabled)
        #self.toggleSeparationButton.setText(self.separationOnIconString if separationEnabled else self.separationOffIconString)
        
        self.gccNMFParams.separationEnabled = separationEnabled
        self.gccNMFDirtyParamNames.append('separationEnabled')
    
    def nextFile(self):
        logging.info('GCCNMFInterface: nextFile()' )
    
    def previousFile(self):
        logging.info('GCCNMFInterface: previousFile()' )
    
    def dictionarySizeChanged(self, changeGCCNMFProcessor=True):
        self.dictionarySize = self.dictionarySizes[self.dictionarySizeDropDown.currentIndex()]
        logging.info('GCCNMFInterface: setting dictionarySize: %d' % self.dictionarySize)
        
        visualizedDictionary = self.dictionariesW[self.dictionaryType][self.dictionarySize]
        self.dictionaryImageItem.setImage(visualizedDictionary)
        self.dictionaryViewBox.setXRange(0, visualizedDictionary.shape[0], padding=0)
        self.dictionaryViewBox.setYRange(0, visualizedDictionary.shape[1], padding=0)
        
        self.coefficientMaskHistory = self.coefficientMaskHistories[self.dictionarySize]
        self.coefficientMaskViewBox.setXRange(0, self.coefficientMaskHistory.values.shape[1], padding=0)
        self.coefficientMaskViewBox.setYRange(0, self.coefficientMaskHistory.values.shape[0], padding=0)
        
        if changeGCCNMFProcessor:
            self.gccNMFParams.dictionarySize = self.dictionarySize
            self.gccNMFDirtyParamNames.append('dictionarySize')    
    
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
        #self.tdoas = np.arange(self.numTDOAs*5).astype(np.float32)
        self.tdoas = np.linspace(0, self.numTDOAs, self.numTDOAs*5).astype(np.float32)
        
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
