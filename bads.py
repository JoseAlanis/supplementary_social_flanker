"""
=================
Find bad channels
=================

Methods for finding bad (e.g., noisy) channels in EEG data.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import warnings

import numpy as np
from scipy.stats import median_absolute_deviation as mad

from mne.io.base import BaseRaw


# main function which implements different methods
def find_bad_channels(inst, picks='eeg',
                      method='correlation',
                      mad_threshold=1,
                      std_threshold=1,
                      r_threshold=0.4,
                      percent_threshold=0.1,
                      time_step=1.0,
                      sfreq=None,
                      return_z_scores=False,
                      channels=None):

    # arguments to be passed to pick_types
    kwargs = {pick: True for pick in [picks]}

    # check that tha input data can be handled by the function
    if isinstance(inst, BaseRaw):
        # only keep data from desired channels
        inst = inst.copy().pick_types(**kwargs)
        dat = inst.get_data() * 1e6  # to microvolt
        channels = inst.ch_names
        sfreq = inst.info['sfreq']
    elif isinstance(inst, np.ndarray):
        if not channels:
            raise ValueError('If "inst" is not an instance of BaseRaw a list '
                             'of channel names must be provided')
        dat = inst
    else:
        raise ValueError('inst must be an instance of BaseRaw or a numpy array')

    # make sure method arguments are in a list
    if not isinstance(method, list):
        method = [method]

    # place holder for results
    bad_channels = dict()

    # 1) find channels with zero or near zero activity
    if 'flat' in method:
        # compute estimates of channel activity
        mad_flats = mad(dat, scale=1, axis=1) < mad_threshold
        std_flats = np.std(dat, axis=1) < std_threshold

        # flat channels identified
        flats = np.argwhere(np.logical_or(mad_flats, std_flats))
        flats = np.asarray([channels[int(flat)] for flat in flats])

        # warn user if too many channels were identified as flat
        if len(flats) > (len(channels) / 2):
            warnings.warn('Too many channels have been identified as "flat"! '
                          'Make sure the input values in "inst" are provided '
                          'on a volt scale. '
                          'Otherwise try choosing another (meaningful) '
                          'threshold for identification.')

        bad_channels.update(flat=flats)

    # 3) find bad channels by deviation (high variability in amplitude)
    if 'deviation' in method:

        # mean absolute deviation (MAD) scores for each channel
        mad_scores = \
            [mad(dat[i, :], scale=1) for i in range(dat.shape[0])]

        # compute robust z-scores for each channel
        rz_scores = \
            0.6745 * (mad_scores - np.nanmedian(mad_scores)) / mad(
                mad_scores,
                scale=1)

        # channels identified by deviation criterion
        bad_deviation = \
            [channels[i] for i in np.where(np.abs(rz_scores) > 5.0)[0]]

        bad_channels.update(deviation=np.asarray(bad_deviation))

        if return_z_scores:
            bad_channels.update(deviation_z_scores=rz_scores)

    # 3) find channels with low correlation to other channels
    if 'correlation' in method:

        # check that sampling frequency argument was provided
        if not sfreq:
            raise ValueError('If "inst" is not an instance of BaseRaw a '
                             'sampling frequency must be provided. Usually '
                             'the sampling frequency of the EEG recording in'
                             'question.')

        # based on the length of the provided data,
        # determine size and amount of time windows
        # for analyses
        corr_frames = time_step * sfreq
        corr_window = np.arange(0, corr_frames)
        n = corr_window.shape[0]
        corr_offsets = \
            np.arange(0, (dat.shape[1] - corr_frames), corr_frames)
        w_correlation = corr_offsets.shape[0]

        # placeholder for correlation coefficients
        channel_correlations = np.ones((w_correlation, len(channels)))

        # cut the data into windows
        x_bp_window = dat[: len(channels), : n * w_correlation]
        x_bp_window = x_bp_window.reshape(len(channels), n, w_correlation)

        # compute (pearson) correlation coefficient across channels
        # (for each channel and analysis time window)
        # take the absolute of the 98th percentile of the correlations with
        # the other channels as a measure of how well that channel is correlated
        # to other channels
        for k in range(w_correlation):
            eeg_portion = x_bp_window[:, :, k]
            window_correlation = np.corrcoef(eeg_portion)
            abs_corr = \
                np.abs((window_correlation - np.diag(np.diag(window_correlation))))  # noqa: E501
            channel_correlations[k, :] = np.percentile(abs_corr, 98, axis=0)

        # check which channels correlate badly with the other channels (i.e.,
        # are below correlation threshold) in a certain fraction of windows
        # (bad_time_threshold)
        thresholded_correlations = channel_correlations < r_threshold
        frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

        # find the corresponding channel names and return
        bad_idxs_bool = frac_bad_corr_windows > percent_threshold
        bad_idxs = np.argwhere(bad_idxs_bool)
        uncorrelated_channels = [channels[int(bad)] for bad in bad_idxs]

        bad_channels.update(correlation=np.asarray(uncorrelated_channels))  # noqa: E501

    return bad_channels