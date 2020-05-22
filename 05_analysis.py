"""
==================================
Exploratory analysis of cue epochs
==================================

Compute descriptive statistics and exploratory analysis plots
for cue locked epochs.

Authors: José C. García Alanis <alanis.jcg@gmail.com>
         Philipp Lange         <philipp.lange0309@gmail.com>

License: BSD (3-clause)
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from mne import read_epochs, combine_evoked, grand_average
from mne.channels import make_1020_channel_selections
from mne.viz import plot_compare_evokeds

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat
from stats import within_subject_cis

incongruent_incorrect_neu = dict()
incongruent_correct_neu = dict()
incongruent_incorrect_erps_neu = dict()
incongruent_correct_erps_neu = dict()
incongruent_incorrect_pos = dict()
incongruent_correct_pos = dict()
incongruent_incorrect_erps_pos = dict()
incongruent_correct_erps_pos = dict()
incongruent_incorrect_neg = dict()
incongruent_correct_neg = dict()
incongruent_incorrect_erps_neg = dict()
incongruent_correct_erps_neg = dict()

baseline = (-0.800, -0.500)

###############################################################################
# 1) loop through subjects and compute ERPs for A and B cues
for subj in subjects:
    # log progress
    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Loading epochs for subject %s' % subj +
          LoggingFormat.END)

    # import the output from previous processing step
    input_file = fname.output(subject=subj,
                              processing_step='reaction_epochs',
                              file_type='epo.fif')
    target_epo = read_epochs(input_file, preload=True)

    # extract a and b epochs (only those with correct responses)
    # and apply baseline
    incongruent_incorrect_neu['subj_%s' % subj] = target_epo['block == 1']['incorrect_incongruent'].apply_baseline(baseline)
    incongruent_correct_neu['subj_%s' % subj] = target_epo['block == 1']['correct_incongruent'].apply_baseline(baseline)
    incongruent_incorrect_pos['subj_%s' % subj] = target_epo['block == 2']['incorrect_incongruent'].apply_baseline(baseline)
    incongruent_correct_pos['subj_%s' % subj] = target_epo['block == 2']['correct_incongruent'].apply_baseline(baseline)
    incongruent_incorrect_neg['subj_%s' % subj] = target_epo['block == 3']['incorrect_incongruent'].apply_baseline(baseline)
    incongruent_correct_neg['subj_%s' % subj] = target_epo['block == 3']['correct_incongruent'].apply_baseline(baseline)

    # compute ERP
    incongruent_incorrect_erps_neu['subj_%s' % subj] = incongruent_incorrect_neu['subj_%s' % subj].average()
    incongruent_correct_erps_neu['subj_%s' % subj] = incongruent_correct_neu['subj_%s' % subj].average()
    incongruent_incorrect_erps_pos['subj_%s' % subj] = incongruent_incorrect_pos['subj_%s' % subj].average()
    incongruent_correct_erps_pos['subj_%s' % subj] = incongruent_correct_pos['subj_%s' % subj].average()
    incongruent_incorrect_erps_neg['subj_%s' % subj] = incongruent_incorrect_neg['subj_%s' % subj].average()
    incongruent_correct_erps_neg['subj_%s' % subj] = incongruent_correct_neg['subj_%s' % subj].average()

# create evokeds dict
ga_incongruent_incorrect_neu = \
    grand_average(list(incongruent_incorrect_erps_neu.values()))
ga_incongruent_correct_neu = \
    grand_average(list(incongruent_correct_erps_neu.values()))
ga_incongruent_incorrect_pos = \
    grand_average(list(incongruent_incorrect_erps_pos.values()))
ga_incongruent_correct_pos = \
    grand_average(list(incongruent_correct_erps_pos.values()))
ga_incongruent_incorrect_neg = \
    grand_average(list(incongruent_incorrect_erps_neg.values()))
ga_incongruent_correct_neg = \
    grand_average(list(incongruent_correct_erps_neg.values()))


# create and plot difference ERP
joint_kwargs = \
    dict(times=[0.050, 0.200],
         ts_args=dict(time_unit='s'),
         topomap_args=dict(time_unit='s'))

combine_evoked([ga_incongruent_incorrect_neu, - ga_incongruent_correct_neu,
                ga_incongruent_incorrect_pos, - ga_incongruent_correct_pos],
               weights='equal').plot_joint(**joint_kwargs)

compare = plot_compare_evokeds(dict(neg_incorrect=ga_incongruent_incorrect_neg,
                                    neg_correct=ga_incongruent_correct_neg,
                                    pos_incorrect=ga_incongruent_incorrect_pos,
                                    pos_correct=ga_incongruent_correct_pos,
                                    solo_incorrect=ga_incongruent_incorrect_neu,
                                    solo_correct=ga_incongruent_correct_neu),
                               picks='FCz', invert_y=True,
                               ylim=dict(eeg=[-15, 5]))


ga_incongruent_incorrect_neu.plot_joint(picks='eeg', title='Neutral')
ga_incongruent_incorrect_neu.plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4],
                                          ch_type='eeg', title='neutral')

ga_incongruent_incorrect_pos.plot_joint(picks='eeg', title='Positiv')
ga_incongruent_incorrect_pos.plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4],
                                          ch_type='eeg', title='Positiv')

ga_incongruent_incorrect_neg.plot_joint(picks='eeg', title='Positiv')
ga_incongruent_incorrect_neg.plot_topomap(times=[0., 0.1, 0.2, 0.3, 0.4],
                                          ch_type='eeg', title='Positiv')
