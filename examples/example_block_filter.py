"""
In this script it is demonstrated how to filter a noisy sine wave using the block filter
and how to analyse the spectra of the signals
"""

import matplotlib.pyplot as plt
from numpy import linspace, pi, random, sin
from scipy.signal import welch

from signal_filters import bandpass_block_filter

T = 2.5
A = 1.0
af = 0.3
x = linspace(0, 10, num=500, endpoint=True)
fs = 1 / (x[1] - x[0])
y_orig = A * sin(2 * pi * x / T)
y_noise = y_orig + af * random.normal(0, 1, x.size)

f_low = 2 * pi / 3.0
f_hig = 2 * pi / 1.0

y_recov = bandpass_block_filter(x, y_noise, wfiltlo=f_low, wfiltup=f_hig)

# plot time signal
plt.figure("Time series")
plt.plot(x, y_orig, label="original")
plt.plot(x, y_noise, label="noise")
plt.plot(x, y_recov, label="block")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("amplitude [-]")
plt.title("Demonstration block filter time series")

(
    ff,
    psd_y_orig,
) = welch(y_orig, fs, window="hann")
ff, psd_y_noise = welch(y_noise, fs, window="hann")
ff, psd_y_recov = welch(y_recov, fs, window="hann")

plt.figure("PSD")
plt.plot(ff, psd_y_orig, label="original")
plt.plot(ff, psd_y_noise, label="noise")
plt.plot(ff, psd_y_recov, label="block")
plt.semilogy()
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [s]")
plt.title("PSD [s]")
plt.title("Demonstration block filter power spectral densities")


ff2, psd_y_noise2 = welch(y_noise, fs, window="hann", nperseg=128, nfft=128)

var_noise = y_noise.var()
e_sum_1 = psd_y_noise.sum() * (ff[1] - ff[0])
e_sum_2 = psd_y_noise2.sum() * (ff2[1] - ff2[0])

plt.figure("PSD2")
plt.plot(ff, psd_y_noise, "-x", label="N=256")
plt.plot(ff2, psd_y_noise2, "-+", label="N=128")
plt.figtext(0.4, 0.8, f"E time series : {var_noise:.2f} ")
plt.figtext(0.4, 0.75, f"E PSD 256    : {e_sum_1:.2f} ")
plt.figtext(0.4, 0.70, f"E PSD 128    : {e_sum_2:.2f} ")
plt.semilogy()
plt.title("Comparison Energy PSD noise for 2 omega sample spacings")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [s]")
plt.legend()

plt.show()
