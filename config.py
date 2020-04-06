"""
========================
Study configuration file
========================
Configuration parameters and global variable values for the study.
Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
import os
from os import path as op
import getpass
from socket import getfqdn

import argparse
import numpy as np

from utils import FileNames

from mne.channels import make_standard_montage


###############################################################################
class LoggingFormat:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


###############################################################################
# User parser to handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject',
                    metavar='sub###',
                    help='The subject to process',
                    type=int)

# Determine which user is running the scripts on which machine. Set the path to
# where the data is stored and determine how many CPUs to use for analysis.
user = getpass.getuser()  # Username
host = getfqdn()  # Hostname

# You want to add your machine to this list
if user == 'josealanis' and '.uni-marburg.de' in host:
    # iMac at work
    data_dir = '../data'
    n_jobs = 4  # iMac has 6 cores (we'll use 4).
# elif user == 'josealanis' and host == 'josealanis-desktop':
#     # pc at home
#     data_dir = '../data'
#     n_jobs = 8  # My workstation has 16 cores (we'll use 8).
else:
    # Defaults
    data_dir = '../data'
    n_jobs = 1

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)

###############################################################################
# Relevant parameters for the analysis.
sample_rate = 256.  # Hz
task_name = 'flasoc'
task_description = 'effects of social interaction on neural correlates of error processing with a flanker task'
# eeg channel names and locations
montage = make_standard_montage(kind='standard_1020')
# channels to be exclude from import
exclude = ['EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

# subjects to use for analysis
subjects = np.arange(1, 53)

# relevant events in the paradigm
event_ids = {'correct_target_button': 13,
             'correct_non_target_button': 12,
             'incorrect_target_button': 113,
             'incorrect_non_target_button': 112,
             'cue_0': 70,
             'cue_1': 71,
             'cue_2': 72,
             'cue_3': 73,
             'cue_4': 74,
             'cue_5': 75,
             'probe_0': 76,
             'probe_1': 77,
             'probe_2': 78,
             'probe_3': 79,
             'probe_4': 80,
             'probe_5': 81,
             'start_record': 127,
             'pause_record': 245}

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# directories to use for input and output
fname.add('data_dir', data_dir)
fname.add('bids_data', '{data_dir}/sub-{subject:03d}')
fname.add('subject_demographics', '{data_dir}/subject_data/subject_demographics.tsv')  # noqa: E501
fname.add('sourcedata_dir', '{data_dir}/sourcedata')
fname.add('derivatives_dir', '{data_dir}/derivatives')
fname.add('reports_dir', '{derivatives_dir}/reports')
fname.add('results', '{derivatives_dir}/results')
fname.add('figures', '{results}/figures')

# The paths for data file input
fname.add('source', '{sourcedata_dir}/sub-{subject:02d}/sub-{subject:02d}.bdf')


# The paths that are produced by the analysis steps
def output_path(path, processing_step, subject, file_type):
    path = op.join(path.derivatives_dir, processing_step, 'sub-%03d' % subject)
    os.makedirs(path, exist_ok=True)
    return op.join(path, 'sub-%03d-%s-%s' % (subject, processing_step, file_type))  # noqa: E501


# The full path for data file output
fname.add('output', output_path)


# The paths that are produced by the report step
def report_path(path, subject):
    h5_path = op.join(path.reports_dir, 'sub-%03d.h5' % subject)
    html_path = op.join(path.reports_dir, 'sub-%03d-report.html' % subject)
    return h5_path, html_path


# The full path for report file output
fname.add('report', report_path)

# File produced by check_system.py
fname.add('system_check', './system_check.txt')
