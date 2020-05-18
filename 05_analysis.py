 289 lines (254 sloc) 11.2 KB
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

incongruent_incorrect_cues = dict()
incongruent_correct_cues= dict()
incongruent_incorrect_erps = dict()
incongruent_correct_erps  = dict()

baseline = (-0.300, -0.050)

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
    incongruent_incorrect_cues['subj_%s' % subj] = target_epo['incongruent_incorrect'].apply_baseline(baseline)
    incongruent_correct_cues['subj_%s' % subj] = target_epo['incongruent_correct'].apply_baseline(baseline)

    # compute ERP
    incongruent_incorrect_erps['subj_%s' % subj] = incongruent_incorrect_cues['subj_%s' % subj].average()
    incongruent_correct_erps['subj_%s' % subj] = incongruent_correct_cues['subj_%s' % subj].average()


ga_incongruent_incorrect = grand_average(list(incongruent_incorrect_erps.values()))
ga_incongruent_correct = grand_average(list(incongruent_correct_erps.values()))

# 3) plot global field power
gfp_times = {'t1': [0.07, 0.07],
             't2': [0.14, 0.10],
             't3': [0.24, 0.12],
             't4': [0.36, 0.24],
             't5': [0.60, 0.15],
             't6': [0.75, 0.25],
             't7': [2.00, 0.50]}

# create evokeds dict
evokeds = {'incon_incorrect': ga_incongruent_incorrect.copy().crop(tmin=-0.25),
           'incon_correct': ga_incongruent_correct.copy().crop(tmin=-0.25)}

# create and plot difference ERP
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
combine_evoked([ga_incongruent_incorrect, - ga_incongruent_correct],
               weights='equal').plot_joint(**joint_kwargs, picks=['FCz', 'Cz'])
