
"""
===============================================
Repair EEG artefacts caused by ocular movements
===============================================
Identify "bad" components in ICA solution (e.g., components which are highly
correlated the time course of the electrooculogram).
Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
import matplotlib.pyplot as plt

from mne import open_report
from mne.io import read_raw_fif
from mne.preprocessing import read_ica, create_eog_epochs, corrmap

# All parameters are defined in config.py
from config import fname, parser, LoggingFormat

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print(LoggingFormat.PURPLE +
      LoggingFormat.BOLD +
      'Finding and removing bad components for subject %s' % subject +
      LoggingFormat.END)

# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='repair_bads',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

# 2) Import ICA weights from precious processing step
input_file = fname.output(subject=subject,
                          processing_step='fit_ica',
                          file_type='ica.fif')
ica = read_ica(input_file)

# # 3) Find blink components via correlation with EOG-channels
# # get EOG-channel names
# eogs = raw.copy().pick_types(eog=True).ch_names
#
# for eog in eogs:
#     eog_epochs = create_eog_epochs(raw,
#                                    ch_name=eog,
#                                    reject_by_annotation=True)
#
#     # create average blink
#     eog_evoked = eog_epochs.average()
#     eog_evoked.apply_baseline(baseline=(None, -0.2))
#
#     # find components that correlate with activity recorded at eog
#     # channel in question
#     eog_indices, eog_scores = ica.find_bads_eog(raw,
#                                                 ch_name=eog,
#                                                 reject_by_annotation=True)
#
#     # if any "bad" components found:
#     if eog_indices and any(eog_indices) not in ica.exclude:
#
#         for eog_i in eog_indices:
#             # add component to list for exclusion
#             ica.exclude.append(eog_i)  # noqa
#
#             # create summary plots
#             fig = ica.plot_properties(eog_epochs,
#                                       picks=eog_i,
#                                       psd_args={'fmax': 35.},
#                                       image_args={'sigma': 1.},
#                                       show=False)[0]
#             plt.close(fig)
#             fig_evoked = ica.plot_sources(eog_evoked, show=False)
#             plt.close(fig_evoked)
#
#             # create HTML report
#             with open_report(fname.report(subject=subject)[0]) as report:
#                 report.add_figs_to_section(fig, 'Bad components identified '
#                                                 'by %s electrode' % eog,
#                                            section='ICA',
#                                            replace=True)
#                 report.add_figs_to_section(fig_evoked, 'Components sources as '
#                                                        'identified '
#                                                        'by %s electrode' % eog,
#                                            section='ICA',
#                                            replace=True)
#                 report.save(fname.report(subject=subject)[1], overwrite=True,
#                             open_browser=False)

# 4) Find any further components via correlation with template ICA
# (just in case previous step missed any bad components)

# load template file
template_raw_file = fname.output(subject=11,
                                 processing_step='repair_bads',
                                 file_type='raw.fif')
template_raw = read_raw_fif(template_raw_file)

# and template ICA
template_ica_file = fname.output(subject=11,
                                 processing_step='fit_ica',
                                 file_type='ica.fif')
template_ica_file = read_ica(template_ica_file)

# compute correlations with template
corrmap(icas=[template_ica_file, ica],
        template=(0, 1), threshold=0.9, label='blink_up', plot=False)
# placeholder for later, when a suitable component has been found
# corrmap(icas=[template_ica_file, ica],
#         template=(0, 4), threshold=0.9, label='blink_side', plot=False)

# if new components were found add them to exclusion list
if ica.labels_['blink_up'] and any(ica.labels_['blink_up']) not in ica.exclude:
    for component_up in ica.labels_['blink_up']:
        ica.exclude.append(component_up)  # noqa

if ica.labels_['blink_side'] and any(ica.labels_['blink_side']) not in \
        ica.exclude:
    for component_side in ica.labels_['blink_side']:
        ica.exclude.append(component_side)  # noqa

# 5) Remove bad components
# summary plot
sources_plot = ica.plot_sources(raw, show=False)

# apply ica weights to data
ica.apply(raw)

# output path
output_path = fname.output(processing_step='repaired_with_ica',
                           subject=subject,
                           file_type='raw.fif')

raw.save(output_path, overwrite=True)

with open_report(fname.report(subject=subject)[0]) as report:
    report.add_figs_to_section(sources_plot, 'Bad component sources',
                               section='ICA',
                               replace=True)
    report.save(fname.report(subject=subject)[1], overwrite=True,
                open_browser=False)