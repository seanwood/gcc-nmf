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
from collections import namedtuple
import ast
import argparse
try:
    import configparser  # Python 3.x
except ImportError:
    import ConfigParser as configparser # Python 2.x
    
from gccNMF.defs import DEFAULT_AUDIO_FILE, DEFAULT_AUDIO_DIR, DEFAULT_CONFIG_FILE
from gccNMF.realtime.gccNMFPretraining import getDictionariesW

INT_OPTIONS = ['numTDOAs', 'numTDOAHistory', 'numSpectrogramHistory', 'numChannels',
               'windowSize', 'hopSize', 'blockSize', 'dictionarySize', 'numHUpdates',
               'fullscreenDelay']
FLOAT_OPTIONS = ['gccPHATNLAlpha', 'microphoneSeparationInMetres', 'normalizeInputMaxValue']
BOOL_OPTIONS = ['gccPHATNLEnabled', 'normalizeInput']
STRING_OPTIONS = ['dictionaryType', 'audioPath', 'deviceNameQuery', 'startupWindowMode']

def getDefaultConfig():
    configParser = configparser.ConfigParser(allow_no_value=True)
    configParser.optionxform = str
    config = {}
    config['TDOA'] = {'numTDOAs': '64',
                      'numTDOAHistory': '128',
                      'numSpectrogramHistory': '128',
                      'gccPHATNLAlpha': '2.0',
                      'gccPHATNLEnabled': 'False',
                      'microphoneSeparationInMetres': '0.1',
                      'targetTDOAEpsilon': '5.0',
                      'targetTDOABeta': '2.0',
                      'targetTDOANoiseFloor': '0.0'}
    
    config['Audio'] = {'numChannels': '2',
                       'sampleRate': '16000',
                       'normalizeInput': 'True',
                       'normalizeInputMaxValue': '0.99',
                       'deviceNameQuery': 'None',
                       'audioPath': DEFAULT_AUDIO_DIR}
    
    config['STFT'] = {'windowSize': '1024',
                      'hopSize': '512',
                      'blockSize': '1024'}
    
    config['NMF'] = {'dictionarySize': '50',
                     'dictionarySizes': '[1, 2, 3, 4, 5, 10, 25, 50, 100, 200, 300, 400, 500]',
                     'dictionaryType': 'Pretrained',
                     'numHUpdates': '0'}
    
    config['UI'] = {'startupWindowMode': 'fullscreen',
                    'linuxFullscreenDelay': '1000'}
    
    try:
        for key, value in config.items():
            configParser[key] = value
    except:
        for sectionKey, sectionValue in config.items():
            configParser.add_section(sectionKey)
            for key, value in sectionValue.items():
                configParser.set(sectionKey, key, value)
    return configParser

def getDictFromConfig(config):
    logging.info('GCCNMFConfig: loading configuration params...')
    dictionary = {}
    for section in config.sections():
        logging.info(section)
        dictionary[section] = {}
        for option in config.options(section):
            if option in INT_OPTIONS:
                dictionary[option] = config.getint(section, option)
            elif option in FLOAT_OPTIONS:
                dictionary[option] = config.getfloat(section, option)
            elif option in BOOL_OPTIONS:
                dictionary[option] = config.getboolean(section, option)
            elif option in STRING_OPTIONS:
                dictionary[option] = config.get(section, option)
            else:
                dictionary[option] = ast.literal_eval( config.get(section, option) )
            logging.info('    %s: %s' % (option, str(dictionary[option])) )
    return dictionary

def getGCCNMFConfig(configPath):
    raise ValueError('configPath is None')

def getGCCNMFConfigParams(audioPath=DEFAULT_AUDIO_DIR, configPath=DEFAULT_CONFIG_FILE):
    try:
        config = getGCCNMFConfig(configPath)
    except:
        config = getDefaultConfig()
        
    parametersDict = getDictFromConfig(config)
    parametersDict['audioPath'] = audioPath
    parametersDict['numFreq'] = parametersDict['windowSize'] // 2 + 1
    parametersDict['windowsPerBlock'] = parametersDict['blockSize'] // parametersDict['hopSize']
    parametersDict['dictionariesW'] = getDictionariesW(parametersDict['windowSize'], parametersDict['dictionarySizes'], ordered=True)
    
    params = namedtuple('ParamsDict', parametersDict.keys())(**parametersDict)
    return params

def parseArguments():
    parser = argparse.ArgumentParser(description='Real-time GCC-NMF Speech Enhancement')
    parser.add_argument('-i','--input', help='input wav file path', default=DEFAULT_AUDIO_DIR, required=False)
    parser.add_argument('-c','--config', help='config file path', default=DEFAULT_CONFIG_FILE, required=False)
    parser.add_argument('--no-gui', help='no user interface mode', action='store_true')
    return parser.parse_args()