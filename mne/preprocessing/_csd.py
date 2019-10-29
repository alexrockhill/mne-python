# Copyright 2003-2010 Jürgen Kayser <rjk23@columbia.edu>
# Copyright 2017 Federico Raimondo <federaimondo@gmail.com> and
#                Denis A. Engemann <dengemann@gmail.com>
#
#
# The original CSD Toolbox can be find at
# http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox/

# Authors: Denis A. Engeman <denis.engemann@gmail.com>
#          Alex Rockhill <aprockhill206@gmail.com>
#
# License: Relicensed under BSD (3-clause) and adapted with
#          permission from authors of original GPL code

import numpy as np

from scipy.linalg import inv

from mne import pick_types, pick_info
from mne.utils import _validate_type
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.channels.interpolation import _calc_g, _calc_h


def _prepare_G(G, lambda2):
    G.flat[::len(G) + 1] += lambda2
    # compute the CSD
    Gi = inv(G)

    TC = Gi.sum(0)
    sgi = np.sum(TC)  # compute sum total

    return Gi, TC, sgi


def _compute_csd(data, G_precomputed, H, radius):
    """Compute the CSD."""
    n_channels, n_times = data.shape
    mu = data.mean(0)[None]
    Z = data - mu
    X = np.zeros_like(data)
    radius **= 2

    Gi, TC, sgi = G_precomputed

    Cp2 = np.dot(Gi, Z)
    c02 = np.sum(Cp2, axis=0) / sgi
    C2 = Cp2 - np.dot(TC[:, None], c02[None, :])
    X = np.dot(C2.T, H).T / radius
    return X


def compute_current_source_density(inst, lambda2=1e-5, stiffness=4,
                                   n_legendre_terms=50,
                                   sphere=(0., 0., 0., 0.095),
                                   copy=True):
    """Get the current source density (CSD) transformation.

    Transformation based on spherical spline surface Laplacian.

    .. note:: This function applies an average reference to the data.
              Do not transform CSD data to source space.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Evoked
        The data to be transformed.
    lambda2 : float
        Regularization parameter, produces smoothnes. Defaults to 1e-5.
    stiffness : float
        Stiffness of the spline. Also referred to as `m`
        (if g_matrix or h_matrix is provided this will be ignored).
    n_legendre_terms : int
        Number of Legendre terms to evaluate (if g_matrix or h_matrix
        is provided this will be ignored).
    sphere : tuple, (float, float, float, float)
        The sphere, head-model of the form (x, y, z, r) where x, y, z
        is the center of the sphere and r is the radius.
    copy : bool
        Whether to overwrite instance data or create a copy.

    Returns
    -------
    inst_csd : instance of Epochs or Evoked
        The transformed data. Output type will match input type.
    """
    _validate_type(inst, (BaseEpochs, BaseRaw, Evoked), 'inst')

    if 'csd' in inst.info['comps']:
        raise ValueError('CSD already applied, should not be reapplied')

    inst = inst.copy() if copy else inst

    picks = pick_types(inst.info, meg=False, eeg=True, exclude='bads')

    if len(picks) == 0:
        raise ValueError('No EEG channels found.')

    if lambda2 is None:
        lambda2 = 1e-5

    _validate_type(lambda2, (float, int), 'lambda2')
    if 0 > lambda2 or lambda2 > 1:
        raise ValueError('lambda2 must be between 0 and 1, got %s' % lambda2)

    _validate_type(stiffness, (float, int), 'stiffness')
    if stiffness < 0:
        raise ValueError('stiffness must be non-negative got %s' % stiffness)

    _validate_type(n_legendre_terms, (int), 'n_legendre_terms')
    if n_legendre_terms < 1:
        raise ValueError('n_legendre_terms must be greater than 0, '
                         'got %s' % n_legendre_terms)

    _validate_type(sphere, tuple, 'sphere')
    x, y, z, radius = sphere
    _validate_type(x, float, 'x')
    _validate_type(y, float, 'y')
    _validate_type(z, float, 'z')
    _validate_type(radius, float, 'radius')
    if radius <= 0:
        raise ValueError('sphere radius must be greater than 0, '
                         'got %s' % radius)

    _validate_type(copy, (bool), 'copy')

    pos = np.array([inst.info['chs'][pick]['loc'][:3] for pick in picks])
    for this_pos, pick in zip(pos, picks):
        if this_pos.sum() == 0.:
            raise ValueError('Zero position found for '
                             '%s' % inst.ch_names[pick])
        if any([not np.isfinite(coord) for coord in this_pos]):
            raise ValueError('Non-finite position found for '
                             '%s' % inst.ch_names[pick])
    pos -= (x, y, z)

    G = _calc_g(np.dot(pos, pos.T), stiffness=stiffness,
                num_lterms=n_legendre_terms)
    H = _calc_h(np.dot(pos, pos.T), stiffness=stiffness,
                num_lterms=n_legendre_terms)

    G_precomputed = _prepare_G(G, lambda2)

    trans_csd = _compute_csd(np.eye(len(picks)),
                             G_precomputed=G_precomputed,
                             H=H, radius=radius)

    if isinstance(inst, BaseEpochs):
        for epo in inst._data[:, picks]:
            epo[picks] = np.dot(trans_csd, epo[picks])
    else:
        inst._data = np.dot(trans_csd, inst._data[picks])

    pick_info(inst.info, picks, copy=False)
    inst.info['comps'].append('csd')
    return inst

# References
# ----------
#
# [1] Perrin et al. (1989, 1990), published in appendix of Kayser J, Tenke CE,
#     Clin Neurophysiol 2006;117(2):348-368)
#
# [2] Perrin, Pernier, Bertrand, and Echallier. Electroenceph Clin Neurophysiol
#     1989;72(2):184-187, and Corrigenda EEG 02274 in Electroenceph Clin
#     Neurophysiol 1990;76:565.
