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

from mne import events_from_annotations, Epochs
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

# initialise place holders for metadata entries
flanker = []
target = []
block = []
rt = []
reaction = []
triallist = []
too_soon = []
missed = []
trial = 0

# recode trigger events
for event in range(len(new_evs[:, 2])):
    # if event is a flanker
    if new_evs[event, 2] == 2:
        # save trial idx
        triallist.append(trial)

        # first check if the subsequent target if followed by a response
        if new_evs[event+2, 2] \
                not in {7, 8, 9, 10}:
            # if no response followed, the trial is missed (i.e., there will be
            # no corresponding eeg segment for analysis)
            print('response missed in trial %s' % trial)
            # append nan for reaction dependent measures
            reaction.append(np.nan)
            missed.append(trial)
            rt.append(np.nan)
        elif new_evs[event+1, 2] in {7, 8, 9, 10}:
            # if a response followed the flankers (before target onset)
            # the trial is too_soon (i.e., there will be
            # no corresponding eeg segment for analysis)
            print('response to soon in trial %s' % trial)
            # append nan for reaction dependent measures
            reaction.append(np.nan)
            too_soon.append(trial)
            rt.append(np.nan)

        # if an answer followed, check if it was correct or incorrect
        else:
            # correct reactions
            if new_evs[event + 2, 2] in {7, 8}:
                reaction.append('correct')
                # if target congruent
                if new_evs[event + 1, 2] in {3, 4}:
                    # correct congruent
                    new_evs[event + 2, 2] = 11
                elif new_evs[event + 1, 2] in {5, 6}:
                    # correct incongruent
                    new_evs[event + 2, 2] = 12
            # incorrect reactions
            elif new_evs[event + 2, 2] in {9, 10}:
                reaction.append('incorrect')
                # if target congruent
                if new_evs[event + 1, 2] in {3, 4}:
                    # incorrect congruent
                    new_evs[event + 2, 2] = 13
                elif new_evs[event + 1, 2] in {5, 6}:
                    # incorrect incongruent
                    new_evs[event + 2, 2] = 14

            # save trial rt
            trial_rt = (new_evs[event+2, 0] - new_evs[event+1, 0]) / sfreq
            rt.append(trial_rt)
        # append block variable identifying the ongoing condition
        if trial < 48:
            # practice
            block.append(0)
        elif trial < 448:
            # individual condition
            block.append(1)
        elif trial < 848:
            # subjects with cond 2 first (i.e., positive interaction)
            if subject in {2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21,
                           23, 27, 29, 31, 33, 37, 38}:  # 39 too
                block.append(2)
            else:
                block.append(3)
        elif trial < 1248:
            # subjects with cond 3 first (i.e., negative interaction)
            if subject in {2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21,
                           23, 27, 29, 31, 33, 37, 38}:
                block.append(3)
            else:
                block.append(2)

        i = 1
        while new_evs[event + i, 2] not in {3, 4, 5, 6}:
            i += 1
        # add information about the flanker-target combination
        if new_evs[event + i, 2] == 3:
            flanker.append('left')
            target.append('congruent')
        elif new_evs[event + i, 2] == 4:
            flanker.append('right')
            target.append('congruent')
        elif new_evs[event + i, 2] == 5:
            flanker.append('left')
            target.append('incongruent')
        elif new_evs[event + i, 2] == 6:
            flanker.append('right')
            target.append('incongruent')

        # add 1 to trial counter
        trial += 1

###############################################################################
# check if subjects performed the positive condition first
if subject in {2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21,
               23, 27, 29, 31, 33, 37, 38}:
    neg = False
else:
    neg = True

# 4) Create data frame with epochs metadata
try:
    # try to build metadata-df
    metadata = {'trial': triallist,
                'condition': block,
                'reaction': reaction,
                'rt': rt,
                'target': target,
                'flanker': flanker,
                'block': block,
                'subject': np.repeat(subject, len(triallist)),
                'negative_first': np.repeat(neg, len(triallist))}
    metadata = pd.DataFrame(metadata)

    # save metadata structure for further analysis
    subj = str(subject).rjust(3, '0')
    metadata_export = fname.dataframes + '/rt_data_sub-%s.tsv' % subj
except ValueError:
      # print which variable in metadata is of unequal size to others and breaks df creation
        print("ValueError: unequal size of lists, can't create metadata dataframe")
        for k in metadata:
            print('length of list ' + k +  ': ' + str(len(metadata[k])))

# save metadata structure for further analysis
subj = str(subject).rjust(3, '0')
metadata_export = fname.dataframes + '/rt_data_sub-%s.tsv' % subj

# save metadata to df
metadata.to_csv(metadata_export,
                sep='\t')

###############################################################################
# 5) Set descriptive event names for extraction of epochs
reaction_ids = {'correct_congruent': 11,
                'correct_incongruent': 12,
                'incorrect_congruent': 13,
                'incorrect_incongruent': 14}

# only keep reaction events
react_events = new_evs[np.where((new_evs[:, 2] >= 11) & (new_evs[:, 2] <= 14))]

###############################################################################
# 6) Extract the epochs
# drop metadata rows that contain nas (e.g., missed reactions)
metadata = metadata.dropna()

# rejection threshold
reject = dict(eeg=250-6)

# set decimation rate to achieve desired sampling freq
decim = 1
if raw.info['sfreq'] == 256.0:
    decim = 2
elif raw.info['sfreq'] == 512.0:
    decim = 4
elif raw.info['sfreq'] == 1024.0:
    decim = 8

reaction_epochs = Epochs(raw,
                         react_events,
                         reaction_ids,
                         on_missing='ignore',
                         metadata=metadata,
                         tmin=-1.5,
                         tmax=1.5,
                         baseline=None,
                         preload=True,
                         reject_by_annotation=True,
                         reject=reject,
                         decim=decim)

###############################################################################
# 7) Save epochs
# output path for cues
reaction_output_path = fname.output(processing_step='reaction_epochs',
                                    subject=subject,
                                    file_type='epo.fif')
# save to disk
reaction_epochs.save(reaction_output_path, overwrite=True)
