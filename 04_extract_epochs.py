import pandas as pd
import numpy as np

from mne import events_from_annotations, Epochs, open_report
from mne.io import read_raw_fif

# All parameters are defined in config.py
from config import fname, parser, LoggingFormat

args = parser.parse_args()
subject = args.subject
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='repaired_with_ica',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

# only keep EEG channels
raw.pick_types(eeg=True)

# 2) Get events from continuous EEG data

# create a dictionary with event IDs for standardised handling
ev_ids = {'245': 1,  # end of block
          '71': 2,  # onset of flanker stimuli
          '11': 3,  # target_C_L
          '12': 4,  # target_C_R
          '21': 5,  # target_I_L
          '22': 6,  # target_I_R
          '101': 7,  # left button pressed correctly
          '102': 8,  # right button pressed correctly
          '201': 9,  # left button pressed incorrectly
          '202': 10  # right button pressed incorrectly
          }

# extract events
events = events_from_annotations(raw, event_id=ev_ids, regexp=None)

###############################################################################
# 3) Recode events into respective conditions and add information about valid
# and invalid responses

# copy of events
new_evs = events[0].copy()

# global variables
trial = 0
sfreq = raw.info['sfreq']
block_end = new_evs[new_evs[:, 2] == 1, 0] / sfreq
# place holders for results
first_condition = []
probe_ids = []
reaction = []
rt = []

for event in range(len(new_evs[:, 2])):
    # --- 1st check: if next event is a congruent left stimulus ---
    if new_evs[event, 2] == 3:
        if new_evs[event + 1, 2] == 7:  # correct button left
            new_evs[event + 1, 2] = 11  # correct left congruent
        elif new_evs[event + 1, 2] == 10:  # incorrect left button
            new_evs[event + 1, 2] = 12  # incorrect left congruent
    # check if target is incongruent left
    elif new_evs[event, 2] == 5:        # incongruent left stimulus
        if new_evs[event + 1, 2] == 7:
            new_evs[event + 1, 2] = 13  # correct left incongruent
        elif new_evs[event + 1, 2] == 10:
            new_evs[event + 1, 2] = 14  # incorrect left incongruent
    # check if target is congruent right
    elif new_evs[event, 2] == 4:
        if new_evs[event + 1, 2] == 8:      # right button correct
            new_evs[event + 1, 2] = 15      # correct right congruent
        elif new_evs[event + 1, 2] == 9:    # left button incorrect
            new_evs[event + 1, 2] = 16      # incorrect right congruent
    # check if target is right incongruent
    elif new_evs[event, 2] == 6:
        if new_evs[event + 1, 2] == 8:      # correct right button
            new_evs[event + 1, 2] = 17      # correct right incongruent
        elif new_evs[event + 1, 2] == 9:    # left button incorrect
            new_evs[event + 1, 2] = 18      # incorrect right incongruent



target_events = {'correct_LC': 11,
                 'incorrect_LC': 12,

                 'correct_LI': 13,
                 'incorrect_LI': 14,

                 'correct_RC': 15,
                 'incorrect_RC': 16,

                 'correct_RI': 17,
                 'incorrect_RI': 18
                 }