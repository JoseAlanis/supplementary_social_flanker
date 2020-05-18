"""
This script performs a series of checks on the system to see if everything is
ready to run the analysis pipeline.
"""

import os
import pkg_resources

import mne

from config import fname

# Check to see if the python dependencies are fulfilled.
dependencies = []
with open('./requirements.txt') as f:
    for line in f:
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        dependencies.append(line)

# This raises errors of dependencies are not met
pkg_resources.working_set.require(dependencies)

# Check that the data is present on the system
if not os.path.exists(fname.data_dir):
    raise ValueError('The `data_dir` points to non-existent directory: ' +
                     fname.data_dir)

# Make sure the output directories exist
os.makedirs(fname.derivatives_dir, exist_ok=True)

# directories for reports
os.makedirs(fname.reports_dir, exist_ok=True)

# directories for results
os.makedirs(fname.results, exist_ok=True)
os.makedirs(fname.figures, exist_ok=True)
os.makedirs(fname.dataframes, exist_ok=True)

# Prints some information about the system
mne.sys_info()

with open(fname.system_check, 'w') as f:
    f.write('System check OK.')

print("\nAll seems to be in order."
      "\nYou can now run the entire pipeline with: python -m doit")
