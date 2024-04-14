"""
The python implementation of the  block filter is tested
"""

import argparse
import logging

import funcy
import matplotlib.pyplot as plt
import numpy as np

# load the python implementation of bandpass2
from numpy import pi
from scipy.signal import freqz, welch

from signal_filters.filters import (
    band_pass_block,
    bandpass_block_filter,
    butter_bandpass_coefficients,
    butterworth_filter,
    kaiser_bandpass_coefficients,
    kaiser_bandpass_filter,
)
from signal_filters.utils import SignalGenerator

logger = logging.getLogger(__name__)


def filter_and_compare(
    show_butter=False,
    show_kaiser=False,
    show_block=False,
    butter_orders=None,
    f_sample=25.0,
    time_length=1800,
    f_kaiser_edge_width=0.1,
    f_low_cut=0.04,
    f_cut_high=0.25,
    timer_units="ms",
    timer_n_digits=0,
):
    """
    Create a noise signal and filter it with several filters and compare the plots

    Parameters
    ----------
    show_butter: bool, optional
        Show the filter signal using Butterworth filter
    show_kaiser: bool, optional
        Show the filter signal using Kaiser filter
    show_block: bool, optional
        Show the filter signal using Block filter
    butter_orders: list, optional
        List of order to use for the Butterworth filter
    f_sample: float, optional
        Sample frequency. Default = 25.0 Hz
    time_length: float, optional
        Length of the time signal which is generated. Default = 1000 s
    f_kaiser_edge_width: float, optional
    f_low_cut:
    f_cut_high
    timer_units
    timer_n_digits

    Returns
    -------

    """
    # Number of samples per fft block used for the Welch spectra at the end of the
    # script. Should be enough to capture the low-frequency peak
    # T_block = n_fft * t_sample = n_fft / f_sample ->
    #  f_min = 1 / T_block < f_low
    n_fft = 2**12
    t_block = n_fft / f_sample
    delta_f = 1 / t_block
    logger.info(
        "Using {} points in fft block with time length of {} s, leading to delta "
        "frequency of {} Hz".format(n_fft, t_block, delta_f)
    )

    # generate the signal
    signal = SignalGenerator(time_length=time_length, sample_frequency=f_sample)
    signal.add_component(key="f_low", amplitude=1.5, period=30.0, phase_shift=0.11)
    signal.add_component(key="f_high", amplitude=1.8, period=1.2, phase_shift=1.5)
    signal.add_component(key="f_zero", amplitude=1, period=14.0)
    signal.add_component(key="noise", amplitude=0.8, signal_type="random_normal")
    signal.report_components()
    signal.generate()
    signal.info()

    f_high_cut = f_low_cut
    f_low_cut = None

    # turn the frequencies from Hz to rad/s
    wfiltlow = 0.001  # 2 * np.pi * f_cut_low
    wfilthig = 2 * np.pi * f_high_cut

    if show_kaiser:
        # calculate the band pass filter coefficients with a bandwidth of 1 Hz.
        taps, n_delay = kaiser_bandpass_coefficients(
            f_low_cut,
            f_high_cut,
            f_sample,
            f_width_edge=f_kaiser_edge_width,
            ripple_db=100,
        )
        logging.debug(f"Kaiser delay {n_delay} with n_taps {len(taps)}")

    # plot the filter components
    plt.figure("Filter coefficients")
    # start with plotting the block filter
    om_block = np.linspace(0, pi, 2000, endpoint=False)
    h_block = band_pass_block(
        om_block, fs=f_sample, lowcut=f_low_cut, highcut=f_high_cut
    )
    plt.plot((f_sample * 0.5 / np.pi) * om_block, h_block, "-r", label="Block")

    plt.grid(True)
    plt.xlabel("Frequency/Freq_Nyq [-]")
    plt.xlim(0, 1)

    # On request, add the butter coefficients and kaiser coefficients
    if show_butter:
        # Plot the frequency response for a few different orders.
        for order in butter_orders:
            b, a = butter_bandpass_coefficients(
                f_low_cut, f_high_cut, f_sample, order=order[0]
            )
            omega, h = freqz(b, a, worN=2000)
            plt.plot(
                (f_sample * 0.5 / np.pi) * omega,
                abs(h),
                label=f"order = {order}",
            )

    if show_kaiser:
        w, h = freqz(taps, worN=2000)
        plt.plot(f_sample * 0.5 * w / np.pi, abs(h), "--g", label="FIR")

    # plt.legend(loc="outside")

    # set a reference to the signal data
    x = signal.data
    t = signal.time

    if show_butter:
        y_orders = list()
        for order in list(butter_orders):
            with funcy.print_durations(f"butter {order}"):
                y_block_butter = butterworth_filter(
                    x, f_low_cut, f_high_cut, f_sample, order=order[0]
                )
            y_orders.append(y_block_butter)

    if show_block:
        with funcy.print_durations("ideal bandpass"):
            dt = t[1] - t[0]
            y_block = bandpass_block_filter(dt, x, wfiltlow, wfilthig)

    if show_kaiser:
        # Use the kaiser filter
        with funcy.print_durations("kaiser bandpass"):
            y_fir = kaiser_bandpass_filter(
                x,
                lowcut=None,
                highcut=f_low_cut,
                fs=f_sample,
                f_width_edge=f_kaiser_edge_width,
                cval=None,
            )

    fig2, axis = plt.subplots(ncols=1, nrows=2, figsize=(10, 12))
    fig2.canvas.manager.set_window_title("Signal vs time")

    axis[0].plot(t, x, "--", label="Noisy signal", linewidth=1)
    axis[0].plot(t, signal.component["f_zero"].data, label="Target signal", linewidth=2)

    if show_butter:
        for cnt, order in enumerate(butter_orders):
            axis[0].plot(
                t,
                y_orders[cnt],
                "-",
                label=f"Butter O{order[0]:d}",
                linewidth=2,
            )

    if show_kaiser:
        axis[0].plot(t, y_fir, "-", label="FIR Kaiser", linewidth=2)

    if show_block:
        axis[0].plot(t, y_block, "-", label="Block", linewidth=2)

    axis[0].set_xlabel("time (seconds)")
    # axis[0].hlines([-a, a], 0, T, linestyles='--')
    # axis[0].grid(True)
    axis[0].set_xlim(200, 300.0)
    axis[0].set_title("Time series")
    axis[0].legend(loc="best")

    axis[1].set_xlabel("Frequence [Hz]")
    axis[1].set_ylabel("Spectral density [s]")
    axis[1].semilogy()
    axis[1].set_title("Power spectra densities")
    freq, s_x = welch(x, fs=f_sample, detrend=False, nfft=n_fft)
    axis[1].plot(freq, s_x, label="Noisy signal")
    freq, s_x0 = welch(
        signal.component["f_zero"].data, fs=f_sample, detrend=False, nfft=n_fft
    )
    axis[1].plot(freq, s_x0, label="Target signal")

    if show_butter:
        for cnt, order in enumerate(butter_orders):
            freq, s_y_order = welch(
                y_orders[cnt], fs=f_sample, detrend=False, nfft=n_fft
            )
            axis[1].plot(freq, s_y_order, "--", label=f"Butter O{order[0]:d}")

    if show_kaiser:
        freq, s_y_fir = welch(
            y_fir[~np.isnan(y_fir)], fs=f_sample, detrend=False, nfft=n_fft
        )
        axis[1].plot(freq, s_y_fir, "--", label="FIR Kaiser")

    if show_block:
        freq, s_y_block = welch(y_block, fs=f_sample, detrend=False, nfft=n_fft)
        axis[1].plot(freq, s_y_block, "-", label="Block Python", linewidth=2)

    axis[1].set_xlabel("Frequency [Hz]")
    axis[1].set_ylabel("PSD [s]")

    plt.show()


def parse_the_command_line_arguments():
    """
    Parse the command line to set some options

    Returns
    -------
    tuple (args, parser)
        The parsed arguments are stored in the args bjec

    """

    parser = argparse.ArgumentParser(
        description="Example of the signal processsin filters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # set the verbosity level command line arguments
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Be quiet: no output",
        action="store_const",
        dest="log_level",
        const=logging.WARNING,
    )
    parser.add_argument("--butter", help="Show the butter filter", action="store_true")
    parser.add_argument("--kaiser", help="Show the kaiser filter", action="store_true")
    parser.add_argument("--block", help="Show the block filter", action="store_true")
    parser.add_argument(
        "--order_butter",
        nargs="*",
        action="append",
        type=int,
        help="The orders to use for the the butter filter",
    )

    # parse the command line
    args = parser.parse_args()

    return args, parser


def main():
    """
    main function of this example
    """
    args, parser = parse_the_command_line_arguments()

    logger.debug("Calling the main function")
    filter_and_compare(
        show_butter=args.butter,
        show_block=args.block,
        show_kaiser=args.kaiser,
        butter_orders=args.order_butter,
    )


if __name__ == "__main__":
    main()
