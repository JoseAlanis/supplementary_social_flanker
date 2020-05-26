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
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from mne import read_epochs, combine_evoked, grand_average, epochs
from mne.channels import make_1020_channel_selections
from mne.viz import plot_compare_evokeds
import mne
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
for sub in subjects:
    # log progress
    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Loading epochs for subject %s' % sub +
          LoggingFormat.END)

    # import the output from previous processing step
    input_file = fname.output(subject=sub,
                              processing_step='reaction_epochs',
                              file_type='epo.fif')
    ern_epo = read_epochs(input_file, preload=True)


    df_epo = ern_epo.copy().apply_baseline(baseline)
    df_epo.crop(tmin=0, tmax=.1)
    df = df_epo.to_data_frame(picks='FCz', index=['epoch'])
    df = df[['time', 'FCz']]
    df = df.merge(df_epo.metadata, left_index=True, right_index=True)
    df.to_csv(
        '/Users/philipplange/PycharmProjects/social_flanker/ernsoc_data_bids/derivatives/results/dataframes/epoch_frames/epoch_subject%s' % sub +
        '.tsv')

# create evokeds dict

