"""
==================================
Extract epochs from continuous EEG
==================================

Extract epochs for each experimental condition

Authors:
Philipp Lange ≤philipp.lange0309@gmail.com>
José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import pandas as pd
import numpy as np

from mne import events_from_annotations, Epochs, open_report
from mne.io import read_raw_fif

# All parameters are defined in config.py
from config import fname, parser, LoggingFormat

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Extracting epochs for subject %s' % subject +
      LoggingFormat.END)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='repaired_with_ica',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

# only keep EEG channels
raw.pick_types(eeg=True)

###############################################################################
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

# recode trigger events
for event in range(len(new_evs[:, 2])):
    # if event is a flanker
    if new_evs[event, 2] == 2:
        # save trial idx
        triallist.append(trial)

        # first check a response followed the target
        if new_evs[event+2, 2] not in {7, 8, 9, 10} or new_evs[event+1, 2] in {7, 8, 9, 10}:
            print('trial %s is broken' % event)

            reaction.append(np.nan)
            broken.append(trial)
            rt.append(np.nan)
        # check if that answer was correct or incorrect
        else:
            if new_evs[event + 2, 2] in {7, 8}:
                reaction.append('correct')
                # if target congruent
                if new_evs[event + 1, 2] in {3, 4}:
                    # correct congruent
                    new_evs[event + 2, 2] = 11
                elif new_evs[event + 1, 2] in {5, 6}:
                    # correct incongruent
                    new_evs[event + 2, 2] = 12

            elif new_evs[event + 2, 2] in {9, 10}:
                reaction.append('incorrect')
                # if target congruent
                if new_evs[event + 1, 2] in {3, 4}:
                    # incorrect congruent
                    new_evs[event + 2, 2] = 13
                elif new_evs[event + 1, 2] in {5, 6}:
                    # incorrect incongruent
                    new_evs[event + 2, 2] = 14

            trial_rt = (new_evs[event+2, 0] - new_evs[event+1, 0]) / sfreq
            rt.append(trial_rt)

        if trial < 48:
            block.append(0)
        elif trial < 448:
            block.append(1)
        elif trial < 848:
            # subjects with cond 3 first
            if int(subject) in {2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 23, 28}:
                block.append(3)
            else:
                block.append(2)
        elif trial < 1248:
            # subjects with cond 2 first
            if int(subject) in {2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 23, 28}:
                block.append(2)
            else:
                block.append(3)

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

        # add 1 to trial counter
        trial += 1


metadata = {'trial': triallist,
            'condition': block,
            'reaction': reaction,
            'rt': rt,
            'target': target,
            'flanker': flanker}

df = pd.DataFrame(metadata)
df['subject'] = subject
# save metadata to df
df.to_csv('/Users/philipplange/PycharmProjects/social_flanker/ernsoc_data_bids/metadata/subject' + str(subject) +  '.tsv',
          sep=' ')


epoch_events = new_evs[np.where((new_evs[:, 2] == 11) | (new_evs[:, 2] == 12) | (new_evs[:, 2] == 13) | (new_evs[:, 2] == 14))]
# drop rows with at least 1 nan
metadata_epochs = df.dropna()

target_ids = {'congruent_correct': 11,
              'incongruent_correct': 12,

              'congruent_incorrect': 13,
              'incongruent_incorrect': 14}

###############################################################################
# 4) Extract the epochs

# rejection threshold
reject = dict(eeg=300e-6)

target_epochs = Epochs(raw,
                       epoch_events,
                       target_ids,
                       on_missing='ignore',
                       metadata=metadata_epochs,
                       tmin=-1,
                       tmax=1,
                       baseline=None,
                       preload=True,
                       reject_by_annotation=True,
                       reject=reject)

# 5) Save epochs

# output path for cues
reaction_output_path = fname.output(processing_step='reaction_epochs',
                               subject=subject,
                               file_type='epo.fif')
# resample and save to disk
target_epochs.resample(sfreq=100.)
target_epochs.save(reaction_output_path, overwrite=True)



