# -*- coding: utf-8 -*-
"""
.. _ex-: plot_csd

=================================================================
Analyze Data Using Current Source Density (CSD)
=================================================================

This script shows an example of how to use CSD. CSD
takes the spatial Laplacian of the sensor signal (derivative in both
x and y). It does what a planar gradiometer does in MEG. Spatial derivative
reduces point spread. CSD transformed data have a sharper or more distinct
topography, reducing the negative impact of volume conduction.
"""
# Authors: Alex Rockhill <aprockhill206@gmail.com>
#
# https://gist.github.com/sherdim/28b069c8ebca17073f24322e8e721e1f

# License: BSD (3-clause)
import matplotlib.pyplot as plt

import numpy as np
import mne
from mne.datasets import sample

from mne.preprocessing import compute_current_source_density

print(__doc__)

###############################################################################
# Load sample subject data
data_path = sample.data_path()

raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif',
                          preload=True)
raw = raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                     exclude=raw.info['bads'])
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=.5,
                    preload=True)
evo = epochs.average()

###############################################################################
# Let's look at the topography of CSD compared to average

fig, axes = plt.subplots(1, 6)
times = np.array([-0.1, 0., 0.05, 0.1, 0.15])
evo.plot_topomap(times=times, cmap='Spectral_r', axes=axes[:5],
                 outlines='skirt', contours=4, time_unit='s',
                 colorbar=True, show=False, title='Average Reference')
fig, axes = plt.subplots(1, 6)
csd_evo = compute_current_source_density(evo, copy=True)
csd_evo.plot_topomap(times=times, axes=axes[:5], cmap='Spectral_r',
                     outlines='skirt', contours=4, time_unit='s',
                     colorbar=True, title='Current Source Density')

###############################################################################
# Let's add the evoked plot

evo.plot_joint(title='Average Reference', show=False)
csd_evo.plot_joint(title='Current Source Density')

###############################################################################
# Let's look at the effect of smoothing and spline flexibility

fig, ax = plt.subplots(4, 4)
fig.set_size_inches(10, 10)
for i, lambda2 in enumerate([1e-7, 1e-5, 1e-3, 0]):
    for j, m in enumerate([5, 4, 3, 2]):
        this_csd_evo = compute_current_source_density(evo, stiffness=m,
                                                      lambda2=lambda2,
                                                      copy=True)
        this_csd_evo.plot_topomap(0.1, axes=ax[i, j],
                                  outlines='skirt', contours=4, time_unit='s',
                                  colorbar=False, show=False)
        ax[i, j].set_title('m=%i, lambda=%s' % (m, lambda2))

plt.show()

# References
# ----------
#
# Perrin et al. (1989, 1990)
#
# \MATLAB\eeglab14_1_1b\functions\popfunc\eeg_laplac.m
#    Juan Sebastian Gonzalez, DFI / Universidad de Chile
#
# MATLAB\eeglab14_1_1b\plugins\PrepPipeline0.55.3\utilities\private\spherical_interpolate.m
#    Copyright 2009-     by Jason D.R. Farquhar (jdrf@zepler.org)
#
# Wang K, Begleiter, 1999
#
# MATLAB\eeglab14_1_1b\plugins\erplab6.1.3\functions\csd\current_source_density.m
#    Copyright (C) 2003 by Jürgen Kayser (Email: kayserj@pi.cpmc.columbia.edu)
#    (published in appendix of Kayser J, Tenke CE,
#     Clin Neurophysiol 2006;117(2):348-368)
#
