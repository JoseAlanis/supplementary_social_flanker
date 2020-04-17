"""
=============================================================
Extract segments of the data recorded during task performance
=============================================================
Segments that were recorded during the self-paced breaks (in between
experimental blocks) will be dropped.
Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.get_cmap('inferno')  # noqa

import numpy as np
import pandas as pd

from mne import Annotations, open_report
from mne.io import read_raw_fif

from scipy.stats import median_absolute_deviation as mad
from sklearn.preprocessing import normalize

# All parameters are defined in config.py
from config import fname, parser, n_jobs, sample_rate

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print('Initialise artefact detection for subject %s' % subject)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='raw_files',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

# Setting up band-pass filter from 0.1 - 40 Hz
#
# FIR filter parameters
# ---------------------
# Designing a one-pass, zero-phase, non-causal bandpass filter:
# - Windowed time-domain design (firwin) method
# - Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
# - Lower passband edge: 0.10
# - Lower transition bandwidth: 0.10 Hz (-6 dB cutoff frequency: 0.05 Hz)
# - Upper passband edge: 40.00 Hz
# - Upper transition bandwidth: 10.00 Hz (-6 dB cutoff frequency: 45.00 Hz)
# - Filter length: 8449 samples (33.004 sec)
raw.filter(l_freq=0.1, h_freq=40.,
           picks=['eeg', 'eog'],
           filter_length='auto',
           l_trans_bandwidth='auto',
           h_trans_bandwidth='auto',
           method='fir',
           phase='zero',
           fir_window='hamming',
           fir_design='firwin',
           n_jobs=n_jobs)

###############################################################################
# 2) Compute robust average reference

iterations = 0
noisy = []
max_iter = 4

raw_copy = raw.copy()
# eeg signal
eeg_signal = raw.get_data(picks='eeg')
# get robust estimate of central tendency (i.e., the median)
ref_signal = np.nanmedian(eeg_signal, axis=0)

eeg_temp = eeg_signal - ref_signal

while True:
    # find bad channels by deviation (high variability in amplitude)
    # mean absolute deviation (MAD) scores for each channel
    mad_scores = \
        [mad(eeg_temp[i, :], scale=1) for i in range(eeg_temp.shape[0])]

    # compute robust z-scores for each channel
    robust_z_scores_dev = \
        0.6745 * (mad_scores - np.nanmedian(mad_scores)) / mad(mad_scores,
                                                               scale=1)

    # channels identified by deviation criterion
    bad_deviation = \
        [raw_copy.ch_names[i]
         for i in np.where(np.abs(robust_z_scores_dev) > 5.0)[0]]

    noisy.extend(bad_deviation)

    if (iterations > 1 and
            (not bad_deviation or set(bad_deviation) == set(noisy))
            or
            iterations > max_iter):
        break

    if bad_deviation:
        raw_copy = raw.copy()
        raw_copy.info['bads'] = list(set(noisy))
        raw_copy.interpolate_bads(mode='accurate')

    eeg_signal = raw_copy.get_data(picks='eeg')
    ref_signal = np.nanmean(eeg_signal, axis=0)
    eeg_temp = eeg_signal - ref_signal

    if bad_deviation:
        print(bad_deviation)

    iterations = iterations + 1

###############################################################################
# 3) Find noisy channels

# remove robust reference
eeg_signal = raw.get_data(picks='eeg')
eeg_temp = eeg_signal - ref_signal

# mean absolute deviation (MAD) scores for each channel
mad_scores = \
    [mad(eeg_temp[i, :], scale=1) for i in range(eeg_temp.shape[0])]

# compute robust z-scores for each channel
robust_z_scores_dev = \
    0.6745 * (mad_scores - np.nanmedian(mad_scores)) / mad(mad_scores,
                                                           scale=1)
# plot results
z_colors = \
    normalize(np.abs(robust_z_scores_dev).reshape((1, robust_z_scores_dev.shape[0]))).ravel()  # noqa: E501

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
fig, ax = plt.subplots(figsize=(5, 15))
for i in range(robust_z_scores_dev.shape[0]):
    ax.axvline(x=5.0, ymin=-5.0, ymax=5.0,
               color='crimson', linestyle='dotted', linewidth=.8)
    ax.text(5.0, -2.0, 'crit. Z-score',  fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            color='crimson', bbox=props)
    ax.barh(i, np.abs(robust_z_scores_dev[i]), 0.9, color=cmap(z_colors[i]))
    ax.text(np.abs(robust_z_scores_dev[i]) + 0.25, i, raw.info['ch_names'][i],
            ha='center', va='center', fontsize=9)
ax.set_xlim(0, int(robust_z_scores_dev.max()+2))
plt.title('EEG channel deviation')
plt.xlabel('Abs. Z-Score')
plt.ylabel('Channels')
plt.close(fig)

# interpolate channels identified by deviation criterion
bad_channels = \
    [raw.ch_names[i] for i in
     np.where(np.abs(robust_z_scores_dev) > 5.0)[0]]
raw.info['bads'] = bad_channels
raw.interpolate_bads(mode='accurate')

###############################################################################
# 3) Reference eeg data to average of all eeg channels

raw.set_eeg_reference(ref_channels='average', projection=True)

###############################################################################
# 4) Find distorted segments in data
# channels to use in artefact detection procedure
eeg_channels = raw.copy().pick_types(eeg=True).ch_names

# ignore fronto-polar channels
picks = [raw.ch_names.index(channel)
         for channel in eeg_channels if channel not in
         {'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'}]

# use a copy of eeg data
raw_copy = raw.copy()
raw_copy.apply_proj()
data = raw_copy.get_data(eeg_channels)

# detect artifacts (i.e., absolute amplitude > 500 microV)
times = []
annotations_df = pd.DataFrame(times)
onsets = []
duration = []
annotated_channels = []
bad_chans = []

# loop through samples
for sample in range(0, data.shape[1]):
    if len(times) > 0:
        if sample <= (times[-1] + int(1 * raw_copy.info['sfreq'])):
            continue
    peak = []
    for channel in picks:
        peak.append(abs(data[channel][sample]))
    if max(peak) >= 300e-6:
        times.append(float(sample))
        annotated_channels.append(raw_copy.ch_names[picks[int(np.argmax(peak))]])  # noqa: E501
# if artifact found create annotations for raw data
if len(times) > 0:
    # get first time
    first_time = raw_copy.first_time
    # column names
    annot_infos = ['onset', 'duration', 'description']

    # save onsets
    onsets = np.asarray(times)
    # include one second before artifact onset
    onsets = ((onsets / raw_copy.info['sfreq']) + first_time) - 1
    # durations and labels
    duration = np.repeat(2, len(onsets))
    description = np.repeat('Bad', len(onsets))

    # get annotations in data
    artifacts = np.array((onsets, duration, description)).T
    # to pandas data frame
    artifacts = pd.DataFrame(artifacts,
                             columns=annot_infos)
    # annotations from data
    annotations = pd.DataFrame(raw_copy.annotations)
    annotations = annotations[annot_infos]

    # merge artifacts and previous annotations
    artifacts = artifacts.append(annotations, ignore_index=True)

    # create new annotation info
    annotations = Annotations(artifacts['onset'],
                              artifacts['duration'],
                              artifacts['description'],
                              orig_time=raw_copy.annotations.orig_time)
    # apply to raw data
    raw.set_annotations(annotations)

# save total annotated time
total_time = sum(duration)
# save frequency of annotation per channel
frequency_of_annotation = {x: annotated_channels.count(x) * 2
                           for x in annotated_channels}

# create plot with clean data
plot_artefacts = raw.plot(scalings=dict(eeg=50e-6, eog=50e-6),
                          n_channels=len(raw.info['ch_names']),
                          title='Robust reference applied Sub-%s' % subject,
                          show=False)


###############################################################################
# 5) Export data to .fif for further processing
# output path
output_path = fname.output(processing_step='repair_bads',
                           subject=subject,
                           file_type='raw.fif')

# sample down and save file
raw.resample(sfreq=sample_rate)
raw.save(output_path, overwrite=True)

###############################################################################
# 6) Create HTML report
bad_channels_identified = '<p>Channels_interpolated:.<br>'\
                          '%s <p>' \
                          % (', '.join([str(chan) for chan in bad_channels]))

with open_report(fname.report(subject=subject)[0]) as report:
    report.add_htmls_to_section(htmls=bad_channels_identified,
                                captions='Bad channels',
                                section='Artefact detection')
    report.add_figs_to_section(fig, 'Robust Z-Scores',
                               section='Artefact detection',
                               replace=True)
    report.add_figs_to_section(plot_artefacts, 'Clean data',
                               section='Artefact detection',
                               replace=True)
    report.save(fname.report(subject=subject)[1], overwrite=True,
                open_browser=False)
