"""Routines for common plasma computations.

This module provides functions which perform common calculations for
plasmas.

The default mode of computation uses the commonly-accepted numerical
values for physical constants. If the "normalized" flag is set in any
function, all calculations are performed with physical constants set
to unity.

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


import numpy as np
import scipy.constants as spc


def electron_plasma_angular_frequency(n:float, normalize:bool=False):
    """Compute the electron plasma angular frequency.

    Compute the electron plasma angular frequency.

    Parameters
    ----------
    n : float
        Electron number density (units of m**-3).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    wp : float
        Electron plasma angular frequency (units of rad/s).
    """
    wp = np.sqrt(n*spc.e**2/(spc.epsilon_0*spc.me))
    return wp


def electron_thermal_speed(T:float, normalize:bool=False):
    """Compute the electron thermal speed.

    Compute the electron thermal speed. This is defined as the RMS
    speed along any single direection.

    Parameters
    ----------
    T : float
        Temperature (units of K).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    vth : float
        Electron thermal speed (units of m/s).

    """
    vth = np.sqrt(2*spc.k*T/spc.me)
    return vth


def electron_plasma_wave_angular_frequency(n:float, T:float, k:float, normalize:bool=False):
    """Compute the electron plasma wave angular frequency.

    Compute the electron plasma wave angular frequency.

    Parameters
    ----------
    n : float
        Electron number density (units of m**-3).
    T : float
        Temperature (units of K).
    k : float
        Wavenumber (units of rad/m).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    w : float
        Electron plasma wave angular frequency (units of rad/s).
    """
    wp = electron_plasma_angular_frequency(n)
    vth = electron_thermal_speed(T)
    w = np.sqrt(wp**2 + 1.5*vth**2*k**2)
    return w


def electron_plasma_wave_phase_speed(n:float, T:float, k:float, normalize:bool=False):
    """Compute the electron plasma wave phase speed.

    Compute the electron plasma wave phase speed.

    Parameters
    ----------
    n : float
        Electron number density (units of m**-3).
    T : float
        Temperature (units of K).
    k : float
        Wavenumber (units of rad/m).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    vphase : float
        Phase speed (units of m/s).
    """
    w = electron_plasma_wave_angular_frequency(n, T, k)
    vphase = w/k
    return vphase
