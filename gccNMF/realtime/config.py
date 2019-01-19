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
    
from gccNMF.defs import DEFAULT_AUDIO_FILE, DEFAULT_CONFIG_FILE
from gccNMF.realtime.gccNMFPretraining import getDictionariesW

INT_OPTIONS = ['numTDOAs', 'numTDOAHistory', 'numSpectrogramHistory', 'numChannels',
               'windowSize', 'hopSize', 'blockSize', 'dictionarySize', 'numHUpdates',
               'localizationWindowSize']
FLOAT_OPTIONS = ['gccPHATNLAlpha', 'microphoneSeparationInMetres']
BOOL_OPTIONS = ['gccPHATNLEnabled', 'localizationEnabled']
STRING_OPTIONS = ['dictionaryType', 'audioPath']

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
                      'targetTDOANoiseFloor': '0.0',
                      'localizationEnabled': 'True',
                      'localizationWindowSize': '6'}
    
    config['Audio'] = {'numChannels': '2',
                       'sampleRate': '16000',
                       'deviceIndex': 'None'}
    
    config['STFT'] = {'windowSize': '1024',
                      'hopSize': '512',
                      'blockSize': '512'}
    
    config['NMF'] = {'dictionarySize': '64',
                     'dictionarySizes': '[64, 128, 256, 512, 1024]',
                     'dictionaryType': 'Pretrained',
                     'numHUpdates': '0'}
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

def getGCCNMFConfigParams(audioPath=DEFAULT_AUDIO_FILE, configPath=DEFAULT_CONFIG_FILE):
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
    parser.add_argument('-i','--input', help='input wav file path', default=DEFAULT_AUDIO_FILE, required=False)
    parser.add_argument('-c','--config', help='config file path', default=DEFAULT_CONFIG_FILE, required=False)
    parser.add_argument('--no-gui', help='no user interface mode', action='store_true')
    return parser.parse_args()