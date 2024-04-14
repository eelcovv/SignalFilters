=============
SignalFilters
=============


    A collection of digital signal filter front end for  `SciPy`_


A collection of signal processing tools, utilities and class for signal processing

Description
===========

The signal processing tool box has the following topics

1. filters: Definition of three digital signal filters (all with low, high, -band-pass mode)
    - Ideal block filter
    - Butterworth filter
    - Kaiser filter
    - Phase shift removal
2. utils: Classes and function to support signal processing
    - SignalGenerator: class to generated signal with multiple harmonic components and noise for
      testing purposes
    - get_peaks: Extract the peaks from a power spectral density

Installation
============

*SignalFilters* can be installed via pip from `PyPi`_::

    pip install SignalFilters

Notes
-----

* The `SciPy`_ packages provides most signal processing tool, such as as a Power
  Spectral Density (PSF) estimator.
* The filters defined in this package are a front end to the Scipy filters, making it
   easier to use digital filters in your code.
* For peak finding either the `PeakUtils`_ or the `PyWafo`_ package is recommended.
* The function *get_peaks* is a front end to the *peakutils.peaks* function

Examples
========

Using digital filters is easy. First define a sine wave with a period of 10 seconds
with some noice

.. code-block:: python

    from  numpy import linspace, sin, random, pi
    from signal_filters.filters import filter_signal


    A_peak = 1.0            # Amplitude at 10 m
    a_noice = 0.2 * A       # Noice rms at 0.2 m
    T_peak = 10             # period of 10 seconds
    f_peak = 1 / T_peak     # peak frequency at 0.1 Hz
    total_time = 1000       # total sampling time of 1000 seconds
    f_sample = 10           # sample frequency at 10 Hz
    n_points = total_time * f_sample

    time = linspace(0, total_time, num=n_points, endpoint=False)
    y_original = sin(2 * pi * time / T_peak)
    y_noise = random.normal(scale=a_noice, size=y_original.size)
    y_total = y_original + y_noise

Next, we can filter this signal with a band pass filter with a cut-off at 0.08 Hz (low)
and at 0.12 Hz (high), such that the expected peak at 0.1 Hz is retrieved:

.. code-block:: python

    y_sine_filtered = filter_signal(y_total,
                                    f_cut_low=0.08,
                                    f_cut_high=.12,
                                    f_sampling=f_sample)

More examples can be found at example_filtering_ and example_filtering_rtd_.

.. _example_filtering:
    _static/example_filtering.html
.. _example_filtering_rtd:
    https://signalfilters.readthedocs.io/en/latest/_static/example_filtering.html

.. _PeakUtils:
   https://pypi.python.org/pypi/PeakUtils
.. _SciPy:
   https://www.scipy.org/
.. _PyWafo:
    https://github.com/wafo-project/pywafo

Note
====

This project has been set up using PyScaffold 4.5.0. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
