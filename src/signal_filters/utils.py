"""
Some utilities to support signal processing.
"""

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import peakutils
from numpy import pi

logger = logging.getLogger(__name__)


class SignalComponent:
    """
    Class to store the properties and data a signal signal component

    Parameters
    ----------
    key: str
       Name of the signal component
    type: {"sin", "cos", "random_normal"}
         Type of the the signal. Can be 'cos', 'sin' and 'random_normal' for now.
         Default = "cos"
    amplitude:  float, optional
        Amplitude of the signal. Default = 1.0
    frequency: float, optional
        Frequency of the periodic signals. Default = 1 Hz
    phase_shift: float, optional
        Phase shift of the periodic components. Default = 0
    """

    def __init__(self, key, type="cos", amplitude=1.0, frequency=1.0, phase_shift=0.0):

        self.key = key
        self.type = type
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.data = None

    def report(self):
        """
        Create a report of the current signal component properties
        """
        fms_str = "{:20s} : {:}"
        fms_flt = "{:20s} : {:.2g}"
        logger.info(fms_str.format("Key", self.key))
        logger.info(fms_str.format("Type", self.type))
        logger.info(fms_flt.format("Amplitude", self.amplitude))
        try:
            logger.info(fms_flt.format("Frequency", self.frequency))
        except TypeError:
            logger.info(fms_str.format("Frequency", self.frequency))
        logger.info(fms_flt.format("Phase", self.phase_shift))


class SignalGenerator:
    """
    Create a signal and generate the data consisting of pure sin components and noise

    Parameters
    ----------
    time_length: float, optional
        Length of the signal in seconds. Default = 100 s
    sample_frequency: float, optional
        Sample frequency in Hz. Default = 1 Hz
    create_data_frame: bool, optional
        if true, also create a pandas data frame to keep all the components

    Notes
    -----
    * A simple signal generator to generate a signal with given components

    Examples
    --------

    A signal consisting of 2 harmonic components and some noise can be created as
    follows

    >>> logger = create_logger(console_log_format_clean=True)
    >>> np.random.seed(0)  # makes sure to generate the same random numbers
    >>> signal = SignalGenerator(time_length=5, sample_frequency=10)

    The *SignalGenerator* object has been initialized but no signal components have been
    defined yet. We can add components by using the *add_components* method.

    >>> signal.add_component(key="f_cos", amplitude=1.5, period=2.5, phase_shift=0.0)

    The *add_component* method can be called several time to add more components. To add
    a sin component with phase shift 0.25 pi we can do

    >>> signal.add_component(key="f_sin", signal_type="sin", amplitude=0.3, period=0.5,
    ...                     phase_shift=0.25 * pi)

    And to add some extra noise do

    >>> signal.add_component(key="noise", signal_type="random_normal", amplitude=0.2)

    To invest which components have been adedd to the *report_components* method:

    >>> signal.report_components()
    # component: f_cos      --------------------------------------------------
    Key                  : f_cos
    Type                 : cos
    Amplitude            : 1.5
    Frequency            : 0.4
    Phase                : 0
    # component: f_sin      --------------------------------------------------
    Key                  : f_sin
    Type                 : sin
    Amplitude            : 0.3
    Frequency            : 2
    Phase                : 0.79
    # component: noise      --------------------------------------------------
    Key                  : noise
    Type                 : random_normal
    Amplitude            : 0.2
    Frequency            : None
    Phase                : 0

    Adding components does not generate the signal yet. For this use the *generate*
    method

    >>> signal.generate()

    After generate has been called the data given by the signal components has been
    generated and stored in the *data* attribute.

    >>> print(signal.data)
    [ 2.0649425   1.80020814  1.46327728  1.24532508  1.04105464  0.48020195
      0.55150542 -0.35827375 -0.95561921 -1.01021343 -0.97258474 -0.83650807
     -1.38289485 -1.76014355 -1.44208923 -0.93465859 -0.39001821 -0.72663093
     -0.51476493 -0.21283052  0.16505956  1.20176587  1.21940984  0.86972051
      1.77062852  1.4212589   1.7293284   1.23009291  1.10370228  0.9614148
      0.70664701  0.43712024 -0.50555946 -1.33113473 -1.16191556 -0.97012366
     -0.88130464 -1.29462642 -1.86194392 -1.59132243 -1.21110405 -0.97283761
     -1.02685331 -0.1872234  -0.14394181  0.58804267  0.82048308  1.20202067
      0.69537395  1.27412954  1.53303872]

    The data attribute is jut a numpy array which contains the summation of all
    individual components.

    The data is by default also stored as a pandas DataFrame in the *data_frame*
    attribute.

    >>> from tabulate import tabulate
    >>> print(tabulate(signal.data_frame.head(5), headers="keys", tablefmt="psql"))
    +------------+---------+---------+------------+-----------+
    |   Time [s] |   Total |   f_cos |      f_sin |     noise |
    |------------+---------+---------+------------+-----------|
    |        0   | 2.06494 | 1.5     |  0.212132  | 0.35281   |
    |        0.1 | 1.80021 | 1.45287 |  0.267302  | 0.0800314 |
    |        0.2 | 1.46328 | 1.31446 | -0.0469303 | 0.195748  |
    |        0.3 | 1.24533 | 1.09345 | -0.296307  | 0.448179  |
    |        0.4 | 1.04105 | 0.80374 | -0.136197  | 0.373512  |
    +------------+---------+---------+------------+-----------+

    The *Total* column of the data frame contains the same total signal as we have seen
    in the *data* attribute. However, all the sub component can be individually accessed
    in the data frame by the columns corresponding to the *key* name we have given to
    the components.
    """

    def __init__(self, time_length=100, sample_frequency=1.0, create_data_frame=True):

        self.time_length = time_length
        self.sample_frequency = sample_frequency
        self.create_data_frame = create_data_frame

        self.component = OrderedDict()
        self.number_of_samples = None
        self.time = None
        self.data = None
        self.data_frame = None

    def add_component(
        self,
        key,
        signal_type="cos",
        amplitude=1.0,
        frequency=None,
        period=None,
        phase_shift=0.0,
    ):
        """Add a new signal component to the list of signals

        Parameters
        ----------

        key: str,
            Name of this component
        signal_type: {"sin", "cos", "random_signal}
            Type of the signal. Default = *cos*
        amplitude: float, optional
            Amplitude of the signal
        frequency: float, optional
            Frequency of the current signal. Default = None
        period: float, optional
            Period of the current signal. Only frequency -or- period can be given.
            Default = None
        phase_shift: float, optional
            Phase shift of the current signal
        """

        if frequency is not None and period is not None:
            raise ValueError(
                "both frequency argument and period argument are given. Please pick one"
            )

        if period is not None:
            frequency = 1.0 / period

        self.component[key] = SignalComponent(
            key=key,
            type=signal_type,
            amplitude=amplitude,
            frequency=frequency,
            phase_shift=phase_shift,
        )

    def generate(self):
        """Generate the signal data based on the components added to this signal"""

        self.number_of_samples = int(self.time_length * self.sample_frequency + 1)
        self.time = np.linspace(0, self.time_length, self.number_of_samples)
        self.data = None

        for key, ss in self.component.items():
            if ss.type == "cos":
                signal = ss.amplitude * np.cos(
                    2 * pi * ss.frequency * self.time + ss.phase_shift
                )
            elif ss.type == "sin":
                signal = ss.amplitude * np.sin(
                    2 * pi * ss.frequency * self.time + ss.phase_shift
                )
            elif ss.type == "random_normal":
                signal = ss.amplitude * np.random.normal(0, 1, self.number_of_samples)
            else:
                raise ValueError(f"signal properties '{ss.type}' not implemented")

            # Per component store the data (note that we refer to the current component
            # with 'ss' make sure to use copy, otherwise you get a reference link
            ss.data = signal.copy()

            # add the component data to the total data
            if self.data is None:
                self.data = signal
            else:
                self.data += signal

        if self.create_data_frame:
            self.generate_data_frame()

    def report_components(self):
        """
        Make a report of all the signal components added to the SignalGenerator
        """
        for key, signal_properties in self.component.items():
            logger.info(
                "# component: {:10s} "
                "-------------------------------------------------- ".format(key)
            )
            signal_properties.report()

    def generate_data_frame(self, index_name="Time [s]"):
        """
        Turn the data into a pandas data frame. Need to create the data first

        Parameters
        ----------
        index_name : str, optional
            The name given to the index. Default = "Time [s]"
        """

        # start the data frame with the total signal
        self.data_frame = pd.DataFrame(
            index=self.time, data=self.data, columns=["Total"]
        )
        self.data_frame.index.name = index_name
        for key, ss in self.component.items():
            # add all the other components as well
            self.data_frame[key] = ss.data

    def info(self):
        """
        Create some output with information of the signal. Just use pandas for that
        """
        if self.data_frame is None:
            self.generate_data_frame()

        print(self.data_frame.describe())


def get_peaks(
    freq, psd, thres=0.01, min_dist=3, max_number_of_peaks=3, sort_peaks=True
):
    """
    Get the peaks in the power spectral density 'psd'

    Parameters
    ----------
    freq: ndarray
        frequencies belong to psd
    psd:  ndarray
        array with power spectral densities
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    max_number_of_peaks:  int, optional
        maximum number of peaks return. Default = 3
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    sort_peaks: bool
        If True, sort the peaks in the order of their height. Default = True

    Returns
    -------
    tuple
         (f_peak, psd_value_at_peak) Two ndarrays with peak value and psd values

    Examples
    --------

    First create a signal with three harmonical components

    >>> time_length = 1800 # 30 minutes sample
    >>> f_sample = 25 # 25 Hz
    >>> time = np.linspace(0, stop=time_length, num=time_length * f_sample)
    >>> amplitudes = np.array([1, 2, 0.5])  # three amplitudes of the peaks
    >>> f_peaks = np.array([0.03, 0.08, 1])  # three peak frequencies in Hz
    >>> omega_p = 2 * np.pi * f_peaks   # convert frequencies to rad/s
    >>> signal = np.zeros(time.shape)
    >>> for ii, om_p in enumerate(omega_p):
    ...    signal += amplitudes[ii] * np.sin(om_p * time)
    >>> signal += np.random.normal(omega_p.size, scale=0.4)

    The spectrum of this signal should contain three peaks. First calculate the spectrum

    >>> from scipy.signal import welch
    >>> freq, psd_signal = welch(signal, fs=f_sample, nperseg=4096)

    See if we can find the peaks in the spectrum with the get_peaks function

    >>> f_peaks_found, v_p = get_peaks(freq=freq, psd=psd_signal, sort_peaks=False)
    >>> for ii, fp in enumerate(f_peaks_found):
    ...    print("Peak {} at {:4.2f} Hz. Found peak {:4.2f} Hz".format(
                    ii, f_peaks[ii], fp))
    Peak 0 at 0.03 Hz. Found peak 0.03 Hz
    Peak 1 at 0.08 Hz. Found peak 0.08 Hz
    Peak 2 at 1.00 Hz. Found peak 1.00 Hz

    Indeed it can  be seen that the peaks are correctly found in the spectrum.

    We can also sort the peaks in order of their magnitude (which is the default
    behaviour

    >>> f_peaks_found, v_p = get_peaks(freq=freq, psd=psd_signal)
    >>> for ii, fp in enumerate(f_peaks_found):
    ...    print("Peak {} at {:4.2f} Hz with A ={:7.2f}".format(ii, fp, v_p[ii]))
    Peak 0 at 0.08 Hz with A = 215.22
    Peak 1 at 0.03 Hz with A =  54.12
    Peak 2 at 1.00 Hz with A =  13.23

    The highest peaks corresponds to the component with the largest amplitude indeed.
    This makes it easy to get the most dominant peak frequency as it always the first
    components

    >>> print("Most dominant frequency in spectrum is : {:.2f} Hz".format(
                f_peaks_found[0]))
    Most dominant frequency in spectrum is : 0.08 Hz

    Notes
    -----
    * This function is a front end to the *peak* function from the *peakutils* module.
    * The *peakutil.peak* function only returns the indices of the peaks in the array,
      whereas the *get_peaks* function returns the actual frequencies belonging to the
      peaks and also the peak values belonging to the peaks
    * Also, the peaks are sorted in height, such that the first peak is always the
      highest peak
    """

    peakind = peakutils.indexes(psd, thres=thres, min_dist=min_dist)
    logger.debug(f"Peaks: {peakind}")
    # get the frequencies and corresponding psd values of the peaks
    freq_peaks = freq[peakind[:max_number_of_peaks]]
    psd_peaks = psd[peakind[:max_number_of_peaks]]

    # Turn frequencies and psd values of the peaks in a dataframe so we can use the
    # sort routine to sort in the peaks in ascending psd order: the highest peaks is
    # set first
    peak_df = pd.DataFrame(psd_peaks, index=freq_peaks, columns=["psd"])

    if sort_peaks:
        peak_df.sort_values("psd", ascending=False, inplace=True)

    # turn the sorted peaks as numpy arrays
    return peak_df.index.values, peak_df["psd"].values
