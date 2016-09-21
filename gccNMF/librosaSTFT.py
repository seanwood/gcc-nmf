'''
Copyright (c) 2016, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
'''

# Sean Wood: taken from https://github.com/librosa/librosa

import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.fftpack as fft
import scipy
import scipy.signal
import six

MAX_MEM_BLOCK = 2**8 * 2**10

def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix


    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length `n_fft`.


    See Also
    --------
    istft : Inverse STFT

    ifgram : Instantaneous frequency spectrogram


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> D
    array([[  2.576e-03 -0.000e+00j,   4.327e-02 -0.000e+00j, ...,
              3.189e-04 -0.000e+00j,  -5.961e-06 -0.000e+00j],
           [  2.441e-03 +2.884e-19j,   5.145e-02 -5.076e-03j, ...,
             -3.885e-04 -7.253e-05j,   7.334e-05 +3.868e-04j],
          ..., 
           [ -7.120e-06 -1.029e-19j,  -1.951e-09 -3.568e-06j, ...,
             -4.912e-07 -1.487e-07j,   4.438e-06 -1.448e-05j],
           [  7.136e-06 -0.000e+00j,   3.561e-06 -0.000e+00j, ...,
             -5.144e-07 -0.000e+00j,  -1.514e-05 -0.000e+00j]], dtype=complex64)


    Use left-aligned frames, instead of centered frames


    >>> D_left = librosa.stft(y, center=False)


    Use a shorter hop length


    >>> D_short = librosa.stft(y, hop_length=64)


    Display a spectrogram


    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()

    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hann(win_length, sym=False)

    elif six.callable(window):
        # User supplied a window function
        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        if fft_window.size != n_fft:
            raise ParameterError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        valid_audio(y)
        y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix

def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """
    Inverse short-time Fourier transform.

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window      : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window * 2/3
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length `n_fft`

    See Also
    --------
    stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window.
        # 2/3 scaling is to make stft(istft(.)) identity for 25% hop
        ifft_window = scipy.signal.hann(win_length, sym=False) * (2.0 / 3)

    elif six.callable(window):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ParameterError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    ifft_window = pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    y = np.zeros(n_fft + hop_length * (n_frames - 1), dtype=dtype)

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    if center:
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y

class LibrosaError(Exception):
    '''The root librosa exception class'''
    pass


class ParameterError(LibrosaError):
    '''Exception class for mal-formed inputs'''
    pass

def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, size - n - lpad)

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                                     'at least input size ({:d})').format(size,
                                                                          n))

    return np.pad(data, lengths, **kwargs)

def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.

    This implementation uses low-level stride manipulation to avoid
    redundant copies of the time series data.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    frame_length : int > 0 [scalar]
        Length of the frame in samples

    hop_length : int > 0 [scalar]
        Number of samples to hop between frames

    Returns
    -------
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
        An array of frames sampled from `y`:
        `y_frames[i, j] == y[j * hop_length + i]`

    Raises
    ------
    ParameterError
        If `y` is not contiguous in memory, framing is invalid.
        See `np.ascontiguous()` for details.

        If `hop_length < 1`, frames cannot advance.

    Examples
    --------
    Extract 2048-sample frames from `y` with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.util.frame(y, frame_length=2048, hop_length=64)
    array([[ -9.216e-06,   7.710e-06, ...,  -2.117e-06,  -4.362e-07],
           [  2.518e-06,  -6.294e-06, ...,  -1.775e-05,  -6.365e-06],
           ..., 
           [ -7.429e-04,   5.173e-03, ...,   1.105e-05,  -5.074e-06],
           [  2.169e-03,   4.867e-03, ...,   3.666e-06,  -5.571e-06]], dtype=float32)

    '''

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    valid_audio(y)

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    if n_frames < 1:
        raise ParameterError('Buffer is too short (n={:d})'
                                    ' for frame_length={:d}'.format(len(y),
                                                                    frame_length))
    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames

def valid_audio(y, mono=False):
    '''Validate whether a variable contains valid, mono audio data.


    Parameters
    ----------
    y : np.ndarray
      The input data to validate

    mono : bool
      Whether or not to force monophonic audio

    Returns
    -------
    valid : bool
        True if all tests pass

    Raises
    ------
    ParameterError
        If `y` fails to meet the following criteria:
            - `type(y)` is `np.ndarray`
            - `mono == True` and `y.ndim` is not 1
            - `mono == False` and `y.ndim` is not 1 or 2
            - `np.isfinite(y).all()` is not True

    Examples
    --------
    >>> # Only allow monophonic signals
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.util.valid_audio(y)
    True

    >>> # If we want to allow stereo signals
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
    >>> librosa.util.valid_audio(y, mono=False)
    True
    '''

    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                                    'ndim={:d}, shape={}'.format(y.ndim,
                                                                 y.shape))
    elif y.ndim > 2:
        raise ParameterError('Invalid shape for audio: '
                                    'ndim={:d}, shape={}'.format(y.ndim,
                                                                 y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True