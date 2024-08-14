#
# Copyright (C) 2013-2024  Leo Singer, Rainer Corley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Cosmology-related utilities.

All functions in this module use the Planck15 cosmological parameters.
"""

import numpy as np
from astropy.cosmology import Planck15 as default_cosmo, z_at_value
import astropy.units as u


def set_cosmo(cosmo):
    """TC: Set the cosmology to be used by the functions in this module."""
    if cosmo is None:
        cosmo = default_cosmo
    return cosmo


def dDL_dz_for_z(z, cosmo=None):
    """Derivative of luminosity distance with respect to redshift."""
    cosmo = set_cosmo(cosmo)
    Ok0 = cosmo.Ok0
    DH = cosmo.hubble_distance
    DC_by_DH = (cosmo.comoving_distance(z) / DH).value
    if Ok0 == 0.0:
        ret = 1.0
    elif Ok0 > 0.0:
        ret = np.cosh(np.sqrt(Ok0) * DC_by_DH)
    else:  # Ok0 < 0.0 or Ok0 is nan
        ret = np.cos(np.sqrt(-Ok0) * DC_by_DH)
    ret *= (1 + z) * DH * cosmo.inv_efunc(z)
    ret += cosmo.comoving_transverse_distance(z)
    return ret


def dVC_dVL_for_z(z, cosmo=None):
    r"""Ratio, :math:`\mathrm{d}V_C / \mathrm{d}V_L`, between the comoving
    volume element and a naively Euclidean volume element in luminosity
    distance space; given as a function of redshift.

    Given the differential comoving volume per unit redshift,
    :math:`\mathrm{d}V_C / \mathrm{d}z`, and the derivative of luminosity
    distance in terms of redshift, :math:`\mathrm{d}D_L / \mathrm{d}z`, this is
    expressed as:

    .. math::

       \frac{\mathrm{d}V_C}{\mathrm{d}V_L} =
       \frac{\mathrm{d}V_C}{\mathrm{d}z}
       \left(
       {D_L}^2 \frac{\mathrm{d}D_L}{\mathrm{d}z}
       \right)^{-1}.
    """
    cosmo = set_cosmo(cosmo)
    Ok0 = cosmo.Ok0
    DH = cosmo.hubble_distance
    DM_by_DH = (cosmo.comoving_transverse_distance(z) / DH).value
    DC_by_DH = (cosmo.comoving_distance(z) / DH).value
    zplus1 = z + 1.0
    if Ok0 == 0.0:
        ret = 1.0
    elif Ok0 > 0.0:
        ret = np.cosh(np.sqrt(Ok0) * DC_by_DH)
    else:  # Ok0 < 0.0 or Ok0 is nan
        ret = np.cos(np.sqrt(-Ok0) * DC_by_DH)
    ret *= zplus1
    ret += DM_by_DH * cosmo.efunc(z)
    ret *= np.square(zplus1)
    return 1.0 / ret


@np.vectorize
def z_for_DL(DL, cosmo=None):
    """Redshift as a function of luminosity distance in Mpc."""
    cosmo = set_cosmo(cosmo)
    return z_at_value(cosmo.luminosity_distance, DL * u.Mpc).to_value(
        u.dimensionless_unscaled
    )


def dVC_dVL_for_DL(DL, cosmo=None):
    """Same as :meth:`dVC_dVL_for_z`, but as a function of luminosity
    distance.
    """
    return dVC_dVL_for_z(z_for_DL(DL, cosmo=cosmo), cosmo=cosmo)
