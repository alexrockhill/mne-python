# -*- coding: utf-8 -*-
"""Test the compute_current_source_density function.

For each supported file format, implement a test.
"""
# Authors: Alex Rockhill <aprockhill206@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from scipy import io as sio

import pytest
from numpy.testing import assert_allclose
import mne
from mne.channels import make_standard_montage
from mne import create_info
from mne.io import RawArray
from mne.utils import run_tests_if_main
from mne.datasets import testing

from mne.preprocessing import compute_current_source_density

from mne.channels.interpolation import _calc_g, _calc_h
from mne.preprocessing._csd import _compute_csd
from mne.preprocessing._csd import _prepare_G

base_path = op.join(testing.data_path(download=False), 'preprocessing')


@testing.requires_testing_data
def test_csd():
    """Test replication of the CSD MATLAB toolbox."""
    mat_contents = sio.loadmat(op.join(base_path, 'test-eeg.mat'))
    data = mat_contents['data']
    n_channels, n_epochs = data.shape[0], data.shape[1] // 386
    sfreq = 250.
    radius = 1.0
    ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)] + ['STI 014']
    ch_types = ['eeg'] * n_channels + ['stim']
    data = np.r_[data, data[-1:]]
    data[-1].fill(0)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=data, info=info)
    montage = make_standard_montage('GSN-HydroCel-257')
    raw.set_montage(montage)

    triggers = np.arange(50, n_epochs * 386, 386)

    raw._data[-1].fill(0.0)
    raw._data[-1, triggers] = [10] * n_epochs

    events = mne.find_events(raw)
    event_id = {
        'foo': 10,
    }
    epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                        preload=True, reject=None, picks=None,
                        baseline=(None, 0), verbose=False)
    epochs.drop_channels(['STI 014'])
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')

    csd_data = sio.loadmat(op.join(base_path, 'test-eeg-csd.mat'))

    """Test G, H and CSD against matlab CSD Toolbox"""
    montage = make_standard_montage('EGI_256', head_size=0.100004)
    positions = np.array([montage.dig[pick]['r'] * 10 for pick in picks])
    cosang = np.dot(positions, positions.T)
    G = _calc_g(cosang)
    assert_allclose(G, csd_data['G'], atol=1e-3)
    H = _calc_h(cosang)
    assert_allclose(H, csd_data['H'], atol=1e-3)
    G_precomputed = _prepare_G(G.copy(), lambda2=1e-5)
    for i in range(n_epochs):
        csd_x = _compute_csd(
            epochs._data[i], G_precomputed=G_precomputed, H=H, radius=radius)
        assert_allclose(csd_x, csd_data['X'][i], atol=1e-3)

    """Test epochs_compute_csd function"""
    csd_raw = compute_current_source_density(raw)

    with pytest.raises(ValueError, match=('CSD already applied, '
                                          'should not be reapplied')):
        compute_current_source_density(csd_raw)

    csd_raw_test_array = np.array([[2.29938168e-07, 1.55737642e-07],
                                   [-9.63976630e-09, 8.31646698e-09],
                                   [-2.30898926e-07, -1.56250505e-07],
                                   [-1.81081104e-07, -5.46661150e-08],
                                   [-9.08835568e-08, 1.61788422e-07],
                                   [5.38295661e-09, 3.75240220e-07]])
    assert_allclose(csd_raw._data[:, 100:102], csd_raw_test_array, atol=1e-3)

    csd_epochs = compute_current_source_density(epochs)
    assert_allclose(csd_epochs._data, csd_data['X'], atol=1e-3)

    csd_epochs = compute_current_source_density(epochs)

    with pytest.raises(TypeError):
        csd_epochs = compute_current_source_density(None)

    fail_raw = raw.copy()
    with pytest.raises(ValueError, match='Zero position found'):
        fail_raw.info['chs'][0]['loc'][:3] = np.array([0., 0., 0.])
        compute_current_source_density(fail_raw)

    with pytest.raises(ValueError, match='Non-finite position found'):
        fail_raw.info['chs'][0]['loc'][:3] = np.array([np.inf, 0., 0.])
        compute_current_source_density(fail_raw)

    with pytest.raises(ValueError, match=('No EEG channels found.')):
        fail_raw = raw.copy().pick_types(eeg=False, stim=True)
        compute_current_source_density(fail_raw)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, lambda2='0')

    with pytest.raises(ValueError, match='lambda2 must be between 0 and 1'):
        compute_current_source_density(epochs, lambda2=2)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, stiffness='0')

    with pytest.raises(ValueError, match='stiffness must be non-negative'):
        compute_current_source_density(epochs, stiffness=-2)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, n_legendre_terms=0.1)

    with pytest.raises(ValueError, match=('n_legendre_terms must be '
                                          'greater than 0')):
        compute_current_source_density(epochs, n_legendre_terms=0)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, sphere=-0.1)

    with pytest.raises(ValueError, match=('sphere radius must be '
                                          'greater than 0')):
        compute_current_source_density(epochs, sphere=(-0.1, 0., 0., -1.))

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, copy=2)

    csd_evoked = compute_current_source_density(epochs.average())
    assert_allclose(csd_evoked.data, csd_data['X'].mean(0), atol=1e-3)
    assert_allclose(csd_evoked.data, csd_epochs._data.mean(0), atol=1e-3)


run_tests_if_main()
