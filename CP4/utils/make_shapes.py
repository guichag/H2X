"""Make shapes"""

import sys
import os
import argparse
import numpy as np


### CST ###


### FUNC ###

def draw_circle(x, y, radius):
    """Draw circle"""
    xloc1 = np.arange(x - radius, x + radius + 1)
    yloc1 = np.arange(y - radius, y + radius + 1)
    xloc, yloc = np.meshgrid(xloc1, yloc1)
    distloc = ( ((xloc - x) * (xloc - x)) + ((yloc - y) * (yloc - y)) )**.5  # 2d-array of distance to center values
    indloc = (distloc <= radius).nonzero()  # -> 2d-tuple (y, x) with indices of values within distance
    ycirc = indloc[0] - radius + y
    xcirc = indloc[1] - radius + x

    return (ycirc, xcirc)


def draw_ellipse(x, y, xlength, ylength):
    """Draw ellipse"""
    xloc = np.arange(x-np.round(xlength), x+np.round(xlength)+1)
    yloc = np.arange(y-np.round(ylength), y+np.round(ylength)+1)[:,None]
    distloc = ((xloc - x)/xlength)**2 + ((yloc - y)/ylength)**2 <=1
    pos = np.where(distloc)
    ycirc = pos[0]+y
    xcirc = pos[1]+x

    return (ycirc, xcirc)
