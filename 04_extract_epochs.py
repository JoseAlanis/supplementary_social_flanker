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

sfreq = raw.info['sfreq']
block_end = new_evs[new_evs[:, 2] == 1, 0] / sfreq

flanker = []
target = []
block = []
rt = []
reaction = []
triallist = []
broken = []
trial = 0
rtarray = np.zeros((1248, 5))
# recode trigger events
for event in range(len(new_evs[:, 2])):
    # if event is a flanker
    if new_evs[event, 2] == 2:
        # save trial idx
        triallist.append(trial)
        if new_evs[event + 1, 2] == 3:
            flanker.append('left')
            target.append('congruent')
        elif new_evs[event + 1, 2] == 4:
            flanker.append('right')
            target.append('congruent')
        elif new_evs[event + 1, 2] == 5:
            flanker.append('left')
            target.append('incongruent')
        elif new_evs[event + 1, 2] == 6:
            flanker.append('right')
            target.append('incongruent')
        # first check a response followed the target
        if new_evs[event+2, 2] not in {7, 8, 9, 10} or new_evs[event+1, 2] in {7, 8, 9, 10}:
            print('trial %s is broken' % event)
            reaction.append(np.nan)
            broken.append('broken')
            rt.append(np.nan)
        else:
            if new_evs[event + 2, 2] in {7, 8}:
                reaction.append('correct')
                broken.append('valid')
            elif new_evs[event + 2, 2] in {9, 10}:
                reaction.append('incorrect')
                broken.append('valid')
            trial_rt = (new_evs[event+2, 0] - new_evs[event+1, 0]) / sfreq
            rt.append(trial_rt)
        if trial < 48:
            block.append(0)
        elif trial < 448:
            block.append(1)
        elif trial < 848:
            if int(subject) in {2, 4, 6, 8,
                             10, 11, 13, 15,
                             17, 19, 21, 23,
                             28}:
                block.append(3)
            else:
                block.append(2)
        elif trial < 1248:
            if int(subject) in {2, 4, 6, 8,
                             10, 11, 13, 15,
                             17, 19, 21, 23,
                             28}:
                block.append(2)
            else:
                block.append(3)
        # add 1 to trial counter
        trial += 1


metadata = {'trial': triallist,
            'condition': block,
            'reaction': reaction,
            'rt': rt,
            'target': target,
            'flanker': flanker,
            'broken': broken}

df = pd.DataFrame(metadata)
df['subject'] = subject
# save metadata to df
df.to_csv('/Users/philipplange/PycharmProjects/social_flanker/ernsoc_data_bids/metadata/subject' + str(subject) +  '.tsv',
          sep=' ')

epoch_events =  events[0].copy()

for event in range(len(epoch_events[:, 2])):
#  --- 1st check: if next event is a congruent left stimulus ---
if epoch_events[event, 2] == 2:
    if new_evs[event+2, 2] not in {7, 8, 9, 10} or new_evs[event+1, 2] in {7, 8, 9, 10}:
        epoch_events[event + 1, 2] = 11  # correct left congruent
    elif epoch_events[event + 1, 2] == 10:  # incorrect right button
        epoch_events[event + 1, 2] = 12  # incorrect left congruent
# check if target is incongruent left
elif epoch_events[event, 2] == 5:  # incongruent left stimulus
    if epoch_events[event + 1, 2] == 7:
        epoch_events[event + 1, 2] = 13  # correct left incongruent
    elif epoch_events[event + 1, 2] == 10:
        epoch_events[event + 1, 2] = 14  # incorrect left incongruent
# check if target is congruent right
elif epoch_events[event, 2] == 4:
    if epoch_events[event + 1, 2] == 8:  # right button correct
        epoch_events[event + 1, 2] = 15  # correct right congruent
    elif epoch_events[event + 1, 2] == 9:  # left button incorrect
        epoch_events[event + 1, 2] = 16  # incorrect right congruent
# check if target is right incongruent
elif epoch_events[event, 2] == 6:
    if epoch_events[event + 1, 2] == 8:  # correct right button
        epoch_events[event + 1, 2] = 17  # correct right incongruent
    elif epoch_events[event + 1, 2] == 9:  # left button incorrect
        epoch_events[event + 1, 2] = 18  # incorrect right incongruent





target_events = {'correct_LC': 11,
                 'incorrect_LC': 12,

                 'correct_LI': 13,
                 'incorrect_LI': 14,

                 'correct_RC': 15,
                 'incorrect_RC': 16,

                 'correct_RI': 17,
                 'incorrect_RI': 18
                 }

