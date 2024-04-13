"""
Collection of digital filters and filter front ends for filters defined in the *scipy*
package

In the filtering module of *signal_processing*  tool box three base filters have been
implemented:

1. Ideal block filter (*bandpass_block_filter*)
2. Kaiser filter (*kaiser_bandpass_filter*)
3. Butterworth filter (*butter_bandpass_filter*).

Notes
-----

* All filters need at least a low *or* a high cut-off frequency as an input argument.
* In case only a low cut-off frequency is passed, the filters behave as a high-pass
  filter, as only the frequencies below f_cut_low are suppress
* In case only a high cut-off frequency is passed, the filters behave as a low
  pass-filter, as only the frequencies above the f_cut_high are suppressed.
* In case both the low and high cut-off frequency are given, the filters are
  band-pass filters.
* The cut-off frequencies and the sample frequency are given in Hz.
* The *kaiser_bandpass_filter* and *butter_bandpass_filter* are in fact front-ends to
  the *scipy* filters.
  In order to use the scipy filter quite some steps are required to calculate
  the filter coefficients.
  The implementations in this *signal_processing* package take care of that and then
  calls the *scipy* filters.
* The Ideal block filter *bandpass_block_filter* is a literal copy of the
  Matlab *band_pass2* filter
* On top of the base filters an extra front end is defined, *filter_signal*,
  which receives the same input arguments as all three filtered mentioned above,
  plus an argument *filter_type*. With this function we can pick which filter we
  want to use: *kaiser*, *block* or
  *butterworth*. The default choice is *block*.
"""

import logging

import numpy as np
from numpy import pi
from scipy.ndimage import shift
from scipy.signal import butter, filtfilt, firwin, kaiserord, lfilter

BUTTER_DEFAULT_ORDER = 2

logger = logging.getLogger(__name__)


def matrix_fft(
    om_spec=None,
    spec=None,
    eps=None,
    om_DFT=None,
    DFT=None,
    om_RAO=None,
    RAO=None,
    t=None,
    TT=None,
    s=4,
):
    """Matrix FFT routine obtained from matlab

    Parameters
    ----------
    om_spec : array_like
        frequencies of spectrum
    spec :  array_like
        spectrum density
    eps : float
        eps must always be in degrees
    om_DFT : array_like
        frequencies of DFT
    DFT : int
        DFT components
    om_RAO : array_like
        array with frequencies
    RAO : float
        rao array
    t : array_like
        array with time values
    TT : array_like
        array with values
    s : int
        Switch to determine the mode of operation, must be integer between 1 and
        5

    Returns
    -------
    :class:list
        List with arrays. The amount of arrays returned depends on the mode used

    Notes
    -----
    * FFT routine copied from HMS matfft by RvD. Modified input argument
      treatment: named input arguments are used Major difference is that this
      routine is only for 1-D signals, in the matlab code the array may contain
      more than one 1-D signals.
    * In case more than one signal needs to be filtered, in the python version
      you need to loop over the signals. Since the filter is linear it does not
      influence the calculation time
    * Depending on the option s, the following input argument are used an may
      not be None::

       1.   spec 2 DFT    use: [om_DFT,DFT]       = matfft(om_spec,spec,eps,1)
       2.   apply RAO     use: [om_DFT,DFT]       = matfft(om_DFT, DFT,om_RAO,
                                                                       RAO,2)
       3.   DFT 2 TT      use: [t     ,TT]        = matfft(om_DFT, DFT,3)
       4.   TT 2 DFT      use: [om_DFT,DFT]       = matfft(t,TT,4)
       5.   DFT 2 spec    use: [om_spec,spec,eps] = matfft(om_DFT, DFT,5)
            eps always in degrees!!!
    * The original code was followed as close as possible which means the
      variable naming does not follow the PIP standards
    """

    # check if the correct arguments are passed
    if s == 1:
        if om_spec is None or spec is None or eps is None:
            raise ValueError("om_spec, spec, and eps arguments must be given")
    elif s == 2:
        if om_DFT is None or DFT is None or om_RAO is None or RAO is None:
            raise ValueError("om_DFT, DFT, om_RAO, RAO arguments must be given")
    elif s in (3, 5):
        if om_DFT is None or DFT is None:
            raise ValueError("om_DFT, and DFT arguments must be given")
    elif s == 4:
        if t is None or TT is None:
            raise ValueError("t, and TT arguments must be given")
    else:
        raise ValueError("Option s must be integer between 1 and 5")

    if s == 3:
        om = om_DFT
        N = om.size
        dw = om[1]
        dt = 2 * pi / dw / N
        t = np.linspace(0, (N - 1) * dt, N)
        ns = DFT.size
        DFTL = np.zeros(N, dtype=complex)
        DFTL[:ns] = DFT[:]
        if N > ns:
            # zero padding if omega is larger than DFT
            DFTL[ns:] = complex(0, 0)
        if N % 2 == 0:
            n_fft = N - 1
        else:
            n_fft = N
        TT = np.real(np.fft.ifft(DFTL, n=n_fft))
        if n_fft != N:
            # in case we have used the add n_fft with a even N, append a zero at
            # the end to correct
            # the length
            TT = np.append(TT, np.array([0]))
        output = [t, TT]
    elif s == 4:

        dt = t[1] - t[0]

        if max(np.abs(np.diff(t) - dt)) > dt / 100.0:
            raise ValueError("time range must be equidistant")

        if TT.shape[0] != t.size:
            raise ValueError("first dim. of TT must equal length(t)")

        dt = (t[-1] - t[0]) / (t.size - 1)
        N = t.size  # number of timesteps
        dw = 2 * pi / N / dt  # frequency step [rad/s]
        om = np.linspace(0, (N - 1) * dw, N, endpoint=True)

        if N % 2 == 0:
            n_fft = N - 1
        else:
            n_fft = N
        DFT = np.fft.fft(TT, n=n_fft)
        if n_fft != N:
            # in case we have used the add n_fft with a even N, append a zero
            # at the end to correct the lenght
            DFT = np.append(DFT, np.array([0 + 0 * 1j]))

        if N % 2 == 0:
            logger.debug(f"In even section {N}")
            # some magic
            DFT[: int(N / 2) + 1] = 2 * DFT[: int(N / 2) + 1]
            DFT[0] /= 2.0
            DFT[int(N / 2) + 1] /= 2.0
            DFT[int(N / 2) + 1 :] = 0
        else:
            logger.debug(f"In odd section {N}")
            DFT[1 : int((N - 1) / 2)] = 2 * DFT[1 : int((N - 1) / 2)]
            DFT[int((N - 1) / 2) + 1 :] = 0

        output = [om, DFT]

    else:
        raise ValueError("Only s=3 and s=4 are implemented")

    return output


def bandpass_block_filter(T, X, wfiltlo=None, wfiltup=None):
    """
    Ideal band pass filter

    Parameters
    ----------
        T: arry_like
            time base 'T' (seconds !!) of size N OR delta time already.
        X: array_like
            signal 'X' of size N. Note that I took out the M dimensional signals
        wfiltlo: float, optional
            Filter frequency 'wfiltlo': lower boundary (high pass filter) in
            rad/s. In case wfillo is not given, zero is imposed, i.e. the band
            pass filter turns into a low pass filter. The high cut-off frequency
            must be given in that case
        wfiltup: float, optional
            filter frequency 'wfiltup': upper boundary (low pass filter) in
            rad/s. In case wfilup is not given, the Nyquist frequency is imposed
            , i.e. the band pass filter turns into a high pass filter.
            The low cut-off frequency must be given in that case
    Returns
    -------
    ndarray
        1-D array with the band passed filtered signal

    Raises
    ------
    ValueError:
        * In case no cut off frequencies are defined
        * In case that wfiltlo >= wfiltup

    Examples
    --------

    Define as signal with some noise

    >>> x = np.linspace(0, 10, num=500, endpoint=True)
    >>> fs = 1 / (x[1] - x[0])
    >>> y_orig = np.sin(2 * pi * x / 2.5)
    >>> y_noise = y_orig + np.random.random_sample(x.size)

    Filter the signal with the ideal block filter to recover the peak period at 2.5 s.
    Set the low and high cut-off frequencies at 3 s and 1 s, respectively

    >>> f_low = 2 * pi / 3.0  # T =  3.0 s. f_low is the frequency in rad/s
    >>> f_hig = 2 * pi / 1.0  # T =  1.0 s. f_hig is the frequency in rad/s
    >>> y_recov = bandpass_block_filter(x, y_noise, wfiltlo=f_low, wfiltup=f_hig)

    The recovered signal *y_recov* now contains the filter signal.

    Notes
    -----
    * Explicity extend the input signal with one value in case we have an odd
      signal. This is to avoid a bug in the fft making the function extremely
      slow for an odd  amount of points.
    * For the rest the original matlab code is followed as close as possible,
      which means that the variable naming does not follow the PIP standards
    """

    N = X.size

    if N % 2 == 1:
        # we have an odd signal, make sure that is is even, otherwise the filter
        # becomes extremely slow
        X = np.append(X, X[-1])
        N = X.size
        extended_array = True
    else:
        extended_array = False

    # put the ValueExceptions on the top
    if wfiltlo is None and wfiltup is None:
        raise ValueError(
            "At least one cut-off frequcny needs to be given. Please define "
            "either wfiltlo or wfiltup"
        )
    if wfiltlo is not None and wfiltup is not None and abs(wfiltlo) >= abs(wfiltup):
        raise ValueError(
            "Low pass filter frequency must be < high pass filter frequency."
        )

    try:
        T = T - T[0]
        if T.size != X.size:
            raise ValueError("Length of array X must be equal to length of array T")
        dt = (T[-1] - T[0]) / (N - 1)
    except (TypeError, IndexError):
        # if the reference to T[0] fails, it is assumed that dt is passed
        # through T already
        dt = T

    wNyq = pi / dt  # Nyquist frequency = 1/2 sampling frequency
    if wfiltup is None or (wfiltup is not None and wfiltup > wNyq):
        wfiltup = wNyq
        logger.warning(
            "High pass filter frequency > Nyquist frequency\n"
            "High pass filter frequency set to Nyquist frequency = pi/dT"
        )
    if wfiltlo is None:
        wfiltlo = 0

    # New code Feb 2007, using matfft and applying RAO
    # equidistant time (3 times original time trace)
    TT = np.linspace(0, 3 * (N - 1) * dt, 3 * (N - 1) + 1, endpoint=True)

    XX = np.zeros(3 * X.size - 2)
    XX[: N - 1] = -X[-1:0:-1] + 2 * X[0]
    XX[N - 1 : 2 * N - 1] = X[:]
    XX[2 * N - 1 : 3 * N - 1] = -X[-1:0:-1] + 2 * X[-1]

    if wfiltup < 0 or wfiltlo < 0:
        # lijnspiegeling
        XX[: N - 1] = -X[1::-1]
        XX[2 * N] = X[:]
        XX[2 * N :] = X[-1::-1]

    # note: left side of FFT contains twice energy, right side = zero
    om_DFT, DFT = matrix_fft(t=TT, TT=XX, s=4)
    RAO = np.where((om_DFT > abs(wfiltup)) | (om_DFT < abs(wfiltlo)), 0, 1)
    DFTr = DFT * RAO

    TT, YY = matrix_fft(om_DFT=om_DFT, DFT=DFTr, s=3)

    Y = YY[int(np.floor(int(YY.size / 3.0))) : int(np.ceil(2.0 * YY.size / 3.0))]

    if extended_array:
        # we added a zero at the beginning to make the signal even. Now remove
        # the last point again to keep the size of the output signal the same
        # as the input
        Y = Y[:-1]

    return Y


def band_pass_block(omega, fs=1, lowcut=None, highcut=None):
    """A perfect block filter with unity response

    Parameters
    ----------
        omega: class:`numpy.ndarray`
            Frequencies in rad/s
        fs: float
            Sample frequency in Hz
        lowcut: float
            Cut-off low frequency in Hz
        highcut: float
            Cut-off high frequency in Hz

    Returns
    -------
    :class:`numpy.ndarray`

        1-D array with 1 in between lowcut, highcut and 0 elsewhere

    Examples
    --------

    Create a array with some radial frequencies running from 0 to 2.5 rad/s

    >>> frequencies = np.linspace(0, 2.5, 10)

    Calculate the block response

    >>> band_pass_block(omega=frequencies, fs=1.0, lowcut=0.1, highcut=0.2)
    array([ 0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])

    """

    h_block = np.ones(omega.shape)
    h_zero = np.zeros(omega.shape)
    if lowcut is not None:
        h_block = np.where(omega * fs / (2 * np.pi) <= lowcut, h_zero, h_block)
    if highcut is not None:
        h_block = np.where(omega * fs / (2 * np.pi) > highcut, h_zero, h_block)

    return h_block


def kaiser_bandpass_coefficients(lowcut, highcut, fs, f_width_edge=None, ripple_db=200):
    """Calculates the Kaiser band pass filter coefficients

    Parameters
    ----------
    lowcut : float
        The low cut-off frequency [Hz]
    highcut : float
        The high cut-off frequency [Hz]
    fs: float
        Sample frequency [Hz]
    f_width_edge : float, optional
        Give the distance in Hz for the edges of the filter. If None, take 1 Hz.
        Default = None
    ripple_db:   float, option.
        The desired attenuation in the stop band, in dB. Default = 200 db

    Returns
    -------
    tuple :
        (*band_pass_coefficients*, *n_delay*)

        - *band_pass_coefficients*: list with the band pass coefficients
        - *n_delay*: the number of delay samples


    """

    f_nyq = fs / 2.0

    if highcut is not None and highcut >= f_nyq:
        logger.info(
            "The high cut frequency of {} is higher than Nyquist {}".format(
                highcut, f_nyq
            )
        )
        highcut = None

    if lowcut is not None and lowcut == 0:
        lowcut = None

    if lowcut is None and highcut is None:
        raise AssertionError(
            "Both low and high cut off are set to none (of set outside the "
            "range 0~f_nyq. Please make sure that you have proper boundaries"
        )

    # turn edge width from Hz to fraction of Nyquist maximum frequency
    if f_width_edge is not None:
        width = f_width_edge / f_nyq
    else:
        width = 1.0 / f_nyq

    # Compute the order and Kaiser parameter for the FIR filter.
    n_kaiser, beta = kaiserord(ripple_db, width)

    # Use firwin with a Kaiser window to create a high pass FIR filter.
    # This is done by modifying the lp to hp taps
    if lowcut is not None:
        # low cut is defined so create the high-pass filter coefficient here.
        # Do this by taking
        # the low-pass filter and apply a spectral inversion
        hp_taps = firwin(n_kaiser, lowcut / f_nyq, window=("kaiser", beta))
        # spectral inversion: multiply with -1 and add one to zero frequency
        hp_taps *= -1
        hp_taps[int(hp_taps.size / 2)] += 1
    else:
        # low cut is None, which means that we only do a low-pass filter
        hp_taps = None

    if highcut is not None:
        # Use firwin with Kaiser to create the low pas filter up the highest
        # frequency
        lp_taps = firwin(n_kaiser, highcut / f_nyq, window=("kaiser", beta))
        if hp_taps is not None:
            # in case the high-pass filter is defined as well, remove unity at
            # zero frequency again
            lp_taps[int(hp_taps.size / 2)] -= 1
    else:
        lp_taps = None

    if lowcut is not None and highcut is not None:
        # combine the low pass and high pass coefficients into band pass
        # coefficients
        taps = [sum(pair) for pair in zip(hp_taps, lp_taps)]
    elif highcut is None:
        taps = hp_taps
    elif lowcut is None:
        taps = lp_taps
    else:
        raise AssertionError(
            "Kaiser filter is used but both low and high cutoff frequency is " "none"
        )

    return taps, n_kaiser


def kaiser_bandpass_filter(
    data,
    lowcut=None,
    highcut=None,
    fs=1,
    f_width_edge=0.01,
    ripple_db=100.0,
    cval=None,
    replace_cval_with_unfiltered_signal=False,
    replace_cval_with_nan=False,
    use_filtfilt=True,
):
    """
    Filter the data with the FIR kaiser

    Parameters
    ----------

    data: ndarray
        Input signal 1D
    lowcut: float or None, optional
        Lower frequency in Hz. If lowcut frequency is not given, the filter
        turns into a low-pass filter with only a high cut-off frequency defined.
        The highcut must be defined in that case
    highcut: float or None, optional
        higher frequency in Hz. If highcut freqyenc is not given the filter
        turns into a high-pass filter with only the low cut-off frequency
        defined.  The lowcut frequency must be defined in that case
    fs: float, optional
        Sample frequency Hz. Default = 1.0 Hz
    f_width_edge: float, optional
        width of the edges. Default = 0.01
    ripple_db:  float, optional
        Supresion of the stop band in dB. Default = 100
    cval: float, optional
        Constant value just for the shift. Default = 0
    replace_cval_with_unfiltered_signal: bool, optional
        You may choose to replace the zero introduced due to the shift.
        Default = False
    replace_cval_with_nan:  bool, optional
        Replace the cval introduced by the shift correction with nans
    use_filtfilt:  bool, optional
        If true use the filtfilt which takes out the phase shift. Default = True

    Returns
    -------
    ndarray:
        Filtered signal

    Raises
    ------
    ValueError:
        * In case no cut off frequency is defined
        * In case the low cut-off frequency is higher than or equal to the high
          cut-off frequency

    Notes
    -----
    * If both the low and high cut-off frequencies *lowcut* and *highcut* are
      defined, the kaiser filter is a band pass filter
    * In case only the low cut-off frequency *lowcut* is defined, all
      frequencies higher than lowcut will pass (i.e. the filter is a high-pass
      filter)
    * In case only the high cut-off frequency *highcut* is defined, all
      frequencies lower than highcut will pass (i.e. the filter is a low-pass
      filter)

    """

    if highcut is None and lowcut is None:
        raise ValueError(
            "At least one cut-off frequency, *highcut* or *lowcut* must be " "given"
        )

    if highcut is not None and lowcut is not None and lowcut >= highcut:
        raise ValueError(
            "The low cut-off frequency must be smaller  than the high cut-off "
            "frequency"
        )

    if replace_cval_with_unfiltered_signal and replace_cval_with_nan:
        raise AssertionError(
            "Both flags replace_cval_with_unfiltered_signal and "
            "replace_cval_with_nan are."
            "while only one is allowed"
        )

    if highcut is not None and highcut > fs / 2:
        logger.info(
            "Clipping the high cut off frequency to the Nyquist "
            "frequency {} - {}"
            "".format(highcut, fs / 2)
        )
        highcut = fs / 2

    if lowcut is not None and lowcut == 0:
        lowcut = 0.001 * fs

    taps, n_delay = kaiser_bandpass_coefficients(
        lowcut, highcut, fs, f_width_edge, ripple_db=ripple_db
    )

    n_shift = int(n_delay / 2)

    if use_filtfilt:
        y_fir = filtfilt(taps, [1.0], data)
    else:
        y_fir = lfilter(taps, [1.0], data)

    if cval is not None:
        # is cval is given as a value, correct the phase shift
        y_fir = shift(y_fir, -n_shift, cval=cval)

        if replace_cval_with_unfiltered_signal:
            # the last part from n_shift contains the value cval; replace it
            # with the unfiltered signal
            delta_y = data[n_shift] - y_fir[n_shift]
            y_fir[-n_shift:] = data[-n_shift:] + delta_y
        elif replace_cval_with_unfiltered_signal:
            y_fir[-n_shift:] = np.nan

    return y_fir


def butter_bandpass_coefficients(f_cut_low, f_cut_high, fs, order=BUTTER_DEFAULT_ORDER):
    """Return the filter coefficients of a Butterworth bandpass filter

    Parameters
    ----------
    f_cut_low :
        Frequency low in Hz: suppress all frequencies below this value
    f_cut_high :
        Frequency high in Hz: suppress all frequencies above this value
    fs :
        the sample frequencies in Hz
    order :
        order of the filter: higher order is sharper drop off edge at the cutoff
        frequencies (order n has 20n dB/decaded drop off) (Default value = 2)

    Returns
    -------
    tuple:
        (b, a) with the band pass filter coefficients

    Notes
    -----
    * This function is a front-end to the *butter* function from the
      *scipy.signal* module.
    * The *butter* function requires the low and/or high cut-off frequency to be
      expressed as a fraction of the Nyquist frequency defined as 0.5 * f_sample
    * This function allows to get the Butterworth filter coefficients using the
      absolute high and low cut frequency as an input
    * This function is used from the *butterworth_filter* function, which is a
      front end to the digital Butterworth filter
    """
    nyq = 0.5 * fs
    low = f_cut_low / nyq
    high = f_cut_high / nyq
    return butter(order, [low, high], btype="bandpass")


def butter_highpass_coefficients(f_cut_low, fs, order=BUTTER_DEFAULT_ORDER):
    """Return the filter coefficients of a Butterworth high-pass filter

    Parameters
    ----------
    f_cut_low :
        suppress all frequencies below this value
    fs :
        the sample frequencies
    order :
        order of the filter: higher order is sharper drop off edge at the cutoff
        frequencies (order n has 20n dB/decade drop off) (Default value = 2)

    Returns
    -------
    tuple:
        (b, a) with the high pass filter coefficients

    Notes
    -----
    * The high-pass filter requires a low cut-off frequency: all frequencies
      below f_cut_low will be suppressed

    """
    nyq = 0.5 * fs
    low = f_cut_low / nyq
    return butter(order, low, btype="highpass")


def butter_lowpass_coefficients(f_cut_high, fs, order=BUTTER_DEFAULT_ORDER):
    """Return the filter coefficients of a Butterworth low-pass filter

    Parameters
    ----------
    f_cut_high :
        suppress all frequencies above this value
    fs :
        the sample frequencies
    order :
        order of the filter: higher order is sharper drop off edge at the cutoff
        frequencies (order n has 20n dB/decade drop off) (Default value = 2)

    Returns
    -------
    tuple:
        (b, a) with the low pass filter coefficients

    Notes
    -----
    * The low-pass filter requires a high cut-off frequency: all frequencies
      above f_cut_high will be suppressed
    """
    nyq = 0.5 * fs
    high = f_cut_high / nyq
    return butter(order, high, btype="lowpass")


def butterworth_filter(
    data,
    f_cut_low=None,
    f_cut_high=None,
    fs=1,
    order=BUTTER_DEFAULT_ORDER,
    use_filtfilt=True,
):
    """Apply the Butterworth filter (high-, low, or band-pass) to the signal
    'data'.

    Parameters
    ----------
    data : ndarray
        Array with the signal to be filtered
    f_cut_low : float or None, optional
        low frequency below which to suppress all signals (Default value = None)
    f_cut_high : float or None, optional
        high frequency above which to suppress all signals (Default value =None)
    fs : float, optional
        sample frequency (Default value = 1)
    order : int
        order of the filter (Default value = 2)
    use_filtfilt: bool, optional
        If true use the filtfilt filter from scipy to take out the phase shift.
        Default = True.
        If False, the lfilter is used instead, causing a phase shift.

    Returns
    -------
    ndarray:
        Filtered signal

    Raises
    ------
    ValueError:
        * In case no cut-off frequencies are defined
        * In case that wfiltlo >= wfiltup

    Examples
    --------

    First create a noise signal with 2 harmonics with a period of 2.5 s and 1.0 s

    >>> x = np.linspace(0, 10, num=500)
    >>> fs = 1 / (x[1] - x[0])
    >>> y_orig = np.sin(2 * pi * x / 2.5) + np.sin(2 * pi * x / 1.0)
    >>> y_noise = y_orig + np.random.random_sample(x.size)

    Filter the signal with the high pass butt block filter to recover the peak with a
    period of 1.0 s.
    Note that for the high-pass filter, we have to define the low-cut frequency
    f_low (as all the frequencies below f_low are suppressed)

    >>> f_low = 1 / 2.0 # T = 2.0 s. f_low is the frequency in Hz
    >>> y_recover = butterworth_filter(y_noise, f_cut_low=f_low, order=4, fs=fs)

    The recovered signal *y_recov* now contains the filtered signal.

    We can also apply a low-pass filter to remove the peak with a frequency of 1 Hz

    >>> f_high = 1 / 2.0  # T = 2.0 s. f_low is the frequency in Hz
    >>> y_recover = butterworth_filter(y_noise, f_cut_high=f_high, order=4, fs=fs)

    Perhaps better would be to use a band pass filter by both defining a high
    and low cut-off frequency. If we want to recover the peak with a period of
    1.0 seconds we can do

    >>> f_high = 1 / 0.5  # T = 0.5 s. Supress all frequencies above 2 Hz
    >>> f_low = 1 / 2.0  # T = 2.0 s. Suppress all frequency below 0.5 Hz
    >>> y_recover = butterworth_filter(y_noise, f_cut_high=f_high, f_cut_low=f_low,
    ...                                fs=fs)

    Notes
    -----
    * The *order* argument is used to control the sharpness of the filter a high order
      will result in a stronger suppression of frequencies outside the cut-off limits.
    * An *order* which is too high also leads to an overflow of the filter. Use
      trial-and-error to find an optimal *order*. Typically, *order* should be
      between 2 and 5
    * In case the filter type is set to *high*, only the low cut-off frequence
      f_cut_low needs to be defined: all the frequencies below f_cut_low will be
      suppressed (hence: high-pass)
    * In case the filter type is set to *low*, only the high cut-off frequence
      f_cut_high needs to be defined: all the frequencies above f_cut_high will
      be suppressed (hence: low-pass)
    """
    if f_cut_high is None and f_cut_low is None:
        raise ValueError(
            "At least one of the cut-off frequencies must be given "
            "(f_cut_high or f_cut_low)"
        )
    if f_cut_high is not None and f_cut_low is not None and f_cut_low >= f_cut_high:
        raise ValueError(
            "The low cut-off frequency must be smaller  than the high cut-off "
        )

    if f_cut_high is None:
        # only the low cut-off frequency is given, so we have a high-pass filter
        coefficients = butter_highpass_coefficients(f_cut_low, fs, order=order)
    elif f_cut_low is None:
        # only the high cut-off frequency is given, so we have a low-pass
        # filter
        coefficients = butter_lowpass_coefficients(f_cut_high, fs, order=order)
    else:
        # both the high and low cut off frequencies are given, so we have a
        # band pass filter
        coefficients = butter_bandpass_coefficients(
            f_cut_low, f_cut_high, fs, order=order
        )

    b, a = coefficients[0], coefficients[1]

    if use_filtfilt:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = lfilter(b, a, data)

    return data_filtered


def remove_phase_shift(signal, degrees=True):
    """Apply a phase shift to the angles of *signal* to get rid of phase-jumps at
    zero degrees

    Parameters
    ----------
    signal :
        the input array with the yaw angle in degrees or rad
    degrees :
        flag to indicate that the yaw is in degrees. Default is True

    Returns
    -------
    ndarray:
        array with shifted phase such that there are no phase-jumps at zero degrees

    Notes
    -----
    * The yaw angle sometimes has discontinuities when the phase angle goes from 0 to
      360. To avoid this, in this routine the range of the yaw is put as good as
      possible in the range between 0 and 360.
    * In case the MRU 6-DOF motions need to be filtered in order to obtain
      the accelerations, the yaw needs to be put into the range 0~360, otherwise the
      discontinuity at 0 degrees will lead to an artificial spike in the acceleration.
    * In case the signal passes the zero degree two times, this routine will fail.

    Examples
    --------

    Lets assume we have a yaw signal which passed the 0 degrees. If we want to
    get the derivative of
    the yaw, we need to put it into a range 0~360 such that we get rid of the
    discontinuity at 0.

    >>> yaw = np.hstack([np.linspace(350, 360, num=5, endpoint=False),
    ...                  np.linspace(0, 10, num=5)])
    >>> print(yaw)
    [ 350.   352.   354.   356.   358.     0.     2.5    5.     7.5   10. ]

    In case we want to calculate the first derivative for the yaw, we get a spike at 0

    >>> dyawdt = np.diff(yaw)
    >>> print(dyawdt)
    [   2.     2.     2.     2.  -358.     2.5    2.5    2.5    2.5]

    In order to get rid of this spike, first remove the phase shift

    >>> yaw2 = remove_phase_shift(yaw)
    >>> print(yaw2)
    [  0.    2.    4.    6.    8.   10.   12.5  15.   17.5  20. ]

    Now the derivative yields proper values with the spike of -358 degrees

    >>> dyaw2dt = np.diff(yaw2)
    >>> print(dyaw2dt)
    [ 2.   2.   2.   2.   2.   2.5  2.5  2.5  2.5]

    """

    if degrees:
        phase = np.deg2rad(signal)
    else:
        phase = signal

    # put phase into range -pi ~ pi
    new_signal = np.angle(np.exp(1j * phase))

    # find the minimum phase angle and set that angle to zero
    index_min = np.argmin(new_signal)
    min_phase = new_signal[index_min]

    # set the minimum phase to zero and move back to degrees if needed
    new_signal -= min_phase

    # convert the angle back to degrees is required
    if degrees:
        new_signal = np.rad2deg(new_signal)

    # start the phase signal at zero
    new_signal -= new_signal[0]

    return new_signal


def filter_signal(
    signal,
    filter_type="block",
    f_cut_low=None,
    f_cut_high=None,
    f_sampling=1,
    f_width_edge=0.05,
    ripple_db=100,
    order=BUTTER_DEFAULT_ORDER,
    constant_value=None,
    use_filtfilt=True,
):
    """Filter the signal of input *signal* base on the input argument

    Parameters
    ----------
    signal: ndarray
        Input 1D signal
    filter_type: {"mean", "butterworth", "kaiser", "block", "none"}
        Type of the filter to use. Default = block.
    f_cut_low: Quantity or float or None
        Low cut-off frequency in Hz. Default = None, which means that f_cut_high must
        be given and that we are dealing with a low-pass filter
    f_cut_high: Quantity or float or None
        High cut-off frequency in Hz. Default = None, which means that f_cut_low must
        be given and that we are dealing with a high-pass filter
    f_sampling: Quantity or float
        Sampling frequency. Default = 1 Hz
    f_width_edge: Quantity or float
        Sharpness of the stop band of the Kaiser filter. Only used when
        filter_type == "kaiser". A smaller f_width_edge approaches the ideal block
        filter better. Default = 0.05 Hz
    ripple_db: float, optional
        The desired attenuation in the stop band in dB. Only used when
        filter_type == "kaiser". Default = 100 dB
    order: int
        Order of the Butterworth filter. Only used when filter_type=="butterworth".
        Default = 2.
        Higher values causes to approach the ideal block band filter, but also a more
        unstable filter.
    constant_value: float
        Constant to use for the Kaiser filter when a phase shift correction is applied.
        Default = None, which means that the phase shift is not correct. Since we use
        filtfilt by default, this is not required. Only used when
        filter_type == "kaiser"
    use_filtfilt: bool
        If true use the filtfilt filter from scipy to take out the phase shift.
        Default = True.

    Returns
    -------
    ndarray:
        Filtered signal

    Raises
    ------
    ValueError:
        * In case no cut-off frequencies are defined
        * In case that wfiltlo >= wfiltup

    Notes
    -----
    * This function is a front end to the other filters in this module
    * The Nyquist frequency follows from the sample frequency as f_nyq = f_s / 2
    * At least one of the cut-off frequencies must be properly defined, i.e. in the
      range 0 ~ f_nyq
    * For f_cut_low == None (or 0), the filter is acting as a low-pass filter with its
      cut-off frequency given by f_cut_high. All frequencies above f_cut_high will be
      suppressed
    * For f_cut_high == None or f_cut_high >= f_nyquist, the filter is acting as a
      high-pass filter with its cut-off frequency given by f_cut_low.
      All frequencies below f_cut_low will be suppressed.
    * If both f_cut_low and f_cut_high are in the range 0 ~ f_nyquist, the filter is
      acting as a band pass filter
    """

    if f_cut_high is None and f_cut_low is None:
        raise ValueError(
            "Both the low and high cut off frequencies are None. At least set "
            "one cut off frequency"
        )
    if f_cut_low is not None and f_cut_high is not None and f_cut_low >= f_cut_high:
        raise ValueError(
            "The low cut-off frequency must be smaller  than the high cut-off "
        )

    # it is better to set the high and low cut to None if they are outside the
    # valid range 0 ~ f_nyq
    if f_cut_high is not None and f_cut_high >= f_sampling / 2:
        f_cut_high = None
    if f_cut_low is not None and f_cut_low == 0:
        f_cut_low = None

    if filter_type == "butterworth":
        filtered = butterworth_filter(
            data=signal,
            f_cut_low=f_cut_low,
            f_cut_high=f_cut_high,
            fs=f_sampling,
            order=order,
        )
    elif filter_type == "mean":
        filtered = signal - signal.mean()
    elif filter_type == "block":
        delta_t = 1.0 / f_sampling
        # The block filter does not allow a pure high or low-pass filter, only band.
        # Mimic pure high or low pass by settings the cut-off frequencies to 0 or the
        # Nyquist frequency
        if f_cut_low is None:
            # In this case, we want to model a low-pass filter (with the f_cut_high as
            # the maximum f)
            # set omega_cut to 0
            omega_cut_low = 0
        else:
            omega_cut_low = 2 * pi * f_cut_low
        if f_cut_high is None:
            # In this case, we want to model a high-pass filter (with the f_cut_low as
            # the minimum f) # set omega_cut to the Nyquist frequency f_n = f_s / 2
            omega_cut_high = 2 * pi * f_sampling / 2.0
        else:
            omega_cut_high = 2 * pi * f_cut_high
        filtered = bandpass_block_filter(
            delta_t, signal, wfiltlo=omega_cut_low, wfiltup=omega_cut_high
        )
    elif filter_type == "kaiser":
        filtered = kaiser_bandpass_filter(
            signal,
            lowcut=f_cut_low,
            highcut=f_cut_high,
            fs=f_sampling,
            f_width_edge=f_width_edge,
            ripple_db=ripple_db,
            cval=constant_value,
            use_filtfilt=use_filtfilt,
        )
    elif filter_type == "none":
        # No filtering at all. Pass the signal without changing.
        filtered = signal
    else:
        logger.warning("Filter not yet implemented")
        raise ValueError(f"Filter not recognised: {filter_type}")

    return filtered
