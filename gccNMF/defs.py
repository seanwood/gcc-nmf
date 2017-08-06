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

from os import environ
from os.path import join, abspath

def getVariableWithDefault(environmentVariable, defaultValue):
    try:
        return environ[environmentVariable]
    except:
        return defaultValue

ROOT_DIR = abspath( join( __file__, '../..' ) )
DATA_DIR = getVariableWithDefault('GCCNMF_DATA_DIR', join(ROOT_DIR, 'data'))
DEFAULT_AUDIO_FILE = join(DATA_DIR, 'dev_Sq1_Co_A_mix.wav')
DEFAULT_CONFIG_FILE = join(ROOT_DIR, 'gccNMF.cfg')

SPEED_OF_SOUND_IN_METRES_PER_SECOND = 340.29