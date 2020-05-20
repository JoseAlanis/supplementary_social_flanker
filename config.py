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
    data_dir = '../ernsoc_data_bids'
    n_jobs = 4  # iMac has 6 cores (we'll use 4).
# elif user == 'josealanis' and host == 'josealanis-desktop':
#     # pc at home
#     data_dir = '../ernsoc_data_bids'
#     n_jobs = 8  # My workstation has 16 cores (we'll use 8).
elif user == 'philipplange' and '.uni-marburg.de' in host:
    data_dir = '../ernsoc_data_bids'
    n_jobs = 4   # philipp's office mac
else:
    # Defaults
    data_dir = '../data'
    n_jobs = 1

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)

###############################################################################
# Relevant parameters for the analysis.
sample_rate = 256.  # Hz
task_name = 'ernsoc'
task_description = 'effects of social interaction on neural correlates of ' \
                   'error processing in the flanker task'
# eeg channel names and locations
montage = make_standard_montage(kind='standard_1020')
# channels to be exclude from import
exclude = ['EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

# subjects to use for analysis
subjects = [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 14, 15, 16, 17, 18,
            19, 20, 23, 24, 27, 28, 29, 31, 32, 33, 34, 35,
            36, 37]

# relevant events in the paradigm
event_ids = {'flanker_onset': 71,
             # first digit in 11 tells whether target stimulus was
             # congruent (1) of incongruent (2) to the flanker stimuli.
             # the second digit tells whether target stimulus was a left (1) or
             # right (2) pointing arrow
             'target_congruent_left': 11,
             'target_congruent_right': 12,
             'target_incongruent_left': 21,
             'target_incongruent_right': 22,
             # button presses
             'correct_left': 101,
             'correct_right': 102,
             'incorrect_left': 201,
             'incorrect_right': 202,
             'end_of_block': 245}

ev_ids = {'245': 1,  # end of block
          '71': 2,   # onset of flanker stimuli
          '11': 3,   # target_C_L
          '12': 4,   # target_C_R
          '21': 5,   # target_I_L
          '22': 6,   # target_I_R
          '101': 7,  # left button pressed correctly
          '102': 8,  # right button pressed correctly
          '201': 9,  # left button pressed incorrectly
          '202': 10  # right button pressed incorrectly
          }

###############################################################################
# Templates for file names
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of file names.
# See fnames.py for details on how this class works.
fname = FileNames()

# Directories to use for input and output:

# bids directory
fname.add('data_dir', data_dir)
# path to sourcedata
fname.add('sourcedata_dir', '{data_dir}/sourcedata')
# target path for data in bids format
fname.add('bids_data', '{data_dir}/sub-{subject:03d}')
# path to derivatives
fname.add('derivatives_dir', '{data_dir}/derivatives')
# path for reports on processing steps
fname.add('reports_dir', '{derivatives_dir}/reports')
# path for results and figures
fname.add('results', '{derivatives_dir}/results')
fname.add('figures', '{results}/figures')
fname.add('dataframes', '{results}/dataframes')


def source_file(files, source_type, subject):
    if source_type == 'eeg':
        return files.sourcedata_dir + '/sub-%02d/%s/10%02d_ern_soc.bdf' % (subject, source_type, subject)  # noqa: E501
    elif source_type == 'demographics':
        return files.sourcedata_dir + '/sub-%02d/%s/10%02d_demographics.tsv' % (subject, source_type, subject)  # noqa: E501


# create full path for data file input
fname.add('source',
          source_file)  # noqa: E501


# create path for files that are produced in each analysis step
def output_path(path, processing_step, subject, file_type):
    path = op.join(path.derivatives_dir, processing_step, 'sub-%03d' % subject)
    os.makedirs(path, exist_ok=True)
    return op.join(path, 'sub-%03d-%s-%s' % (subject, processing_step, file_type))  # noqa: E501


# the full path for data file output
fname.add('output', output_path)


# create path for files that are produced by mne.report()
def report_path(path, subject):
    h5_path = op.join(path.reports_dir, 'sub-%03d.h5' % subject)
    html_path = op.join(path.reports_dir, 'sub-%03d-report.html' % subject)
    return h5_path, html_path


# the full path for the report file output
fname.add('report', report_path)

# path for file produced by check_system.py
fname.add('system_check', './system_check.txt')
