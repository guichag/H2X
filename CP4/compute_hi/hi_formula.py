"""HI formula"""
"""Source: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml"""

import sys
import os
import numpy as np


### CST ###


### FUNC ###

def Tcelsius_to_farhenheit(T):
    """Convert deg. C to Farhenheit"""
    out = (T * 9/5) + 32.

    return out


def Tfarhenheit_to_celsius(T):
    """Convert deg. C to Farhenheit"""
    out = (T - 32.) * (5/9)

    return out


def get_adjustment1(T, RH):
    """First adjustment"""
    out = ((13-RH)/4) * np.sqrt((17-np.abs(T-95.))/17)

    return out


def get_adjustment2(T, RH):
    """Second adjustment"""
    out = ((RH-85)/10) * ((87-T)/5)

    return out


def compute_hi(T, RH):
    """HI formula (Rothfusz regression): T in Farhenheit, RH in %"""
    out = - 42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH - 0.00683783*(T**2) \
          - 0.05481717*(RH**2) + 0.00122874*(T**2)*RH + 0.00085282*T*(RH**2) - .00000199*(T**2)*(RH**2)

    return out


def compute_hi_simple(T, RH):
    """HI simple formula"""
    out = 0.5 * (T + 61.0 + ((T-68.0) * 1.2) + (RH * 0.094))

    return out

