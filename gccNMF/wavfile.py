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

import numpy as np
import contextlib
from scipy.io import wavfile
import logging

CLIP_PROTECTION_MAX_SAMPLE_VALUE = 0.99

def wavread(filePath):
    sampleRate, samples_pcm = wavfile.read(filePath)
    samples_float32 = pcm2float(samples_pcm)
    return samples_float32.T, sampleRate
    
def wavwrite(samples_float32, filePath, sampleRate, clipProtection=True):
    maxAbsValue = np.max(np.abs(samples_float32)) 
    if maxAbsValue >= 1:
        if clipProtection:
            logging.warning('wavwrite: max abs signal value exceeds 1, rescaling to %2f' % CLIP_PROTECTION_MAX_SAMPLE_VALUE)
            samples_float32 = samples_float32 / maxAbsValue * CLIP_PROTECTION_MAX_SAMPLE_VALUE
        else:
            raise ValueError('wavwrite: max abs signal value exceeds 1')
    samples_pcm = float2pcm( samples_float32.astype(np.float32) )
    wavfile.write( filePath, sampleRate, samples_pcm.T )
    
"""
Helper functions for working with audio files in NumPy.

Taken from:
https://raw.githubusercontent.com/mgeier/python-audio/master/audio-files/utility.py
"""

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.

    Returns
    -------
    numpy.ndarray
        Normalized floating point data.

    See Also
    --------
    float2pcm, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.

    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.

    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html

    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.

    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.

    See Also
    --------
    pcm2float, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def pcm24to32(data, channels=1, normalize=True):
    """Convert 24-bit PCM data to 32-bit.

    Parameters
    ----------
    data : buffer
        A buffer object where each group of 3 bytes represents one
        little-endian 24-bit value.
    channels : int, optional
        Number of channels, by default 1.
    normalize : bool, optional
        If ``True`` (the default) the additional zero-byte is added as
        least significant byte, effectively multiplying each value by
        256, which leads to the maximum 24-bit value being mapped to the
        maximum 32-bit value.  If ``False``, the zero-byte is added as
        most significant byte and the values are not changed.

    Returns
    -------
    numpy.ndarray
        The content of *data* converted to an *int32* array, where each
        value was padded with zero-bits in the least significant byte
        (``normalize=True``) or in the most significant byte
        (``normalize=False``).

    """
    if len(data) % 3 != 0:
        raise ValueError('Size of data must be a multiple of 3 bytes')

    out = np.zeros(len(data) // 3, dtype='<i4')
    out.shape = -1, channels
    temp = out.view('uint8').reshape(-1, 4)
    if normalize:
        # write to last 3 columns, leave LSB at zero
        columns = slice(1, None)
    else:
        # write to first 3 columns, leave MSB at zero
        columns = slice(None, -1)
    temp[:, columns] = np.frombuffer(data, dtype='uint8').reshape(-1, 3)
    return out


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for temporarily setting NumPy print options.

    See http://stackoverflow.com/a/2891805/500098

    """
    original = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield
    finally:
        np.set_printoptions(**original)
