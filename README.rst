=============
SignalFilters
=============


    A collection of digital signal filter front end for scipy


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

Notes
-----

* The `SciPy`_ provides most signal processing tool, such as as power spectral density estimator
  *welch*, which uses an equivalent algorithm as the *specdens* fuction from the Matlab tool box
* The filters defined in this package are in fact front ends to the Scipy filters, however,
  in this package the filters have a more user-friendly interface.
* For peak finding either the `PeakUtils`_ or the `PyWafo`_ package is recommended.
* The function *get_peaks* is a front end to the *peakutils.peaks* function

Examples
========

* Wave Spectra Functions : :download:`../examples/example_filtering.html`

.. _PeakUtils:
   https://pypi.python.org/pypi/PeakUtils
.. _SciPy:
   https://www.scipy.org/
.. _PyWafo:
    https://github.com/wafo-project/pywafo

Note
====

This project has been set up using PyScaffold 2.5.6. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
