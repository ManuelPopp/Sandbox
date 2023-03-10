#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:16:22 2023
"""
__author__ = "Manuel"
__date__ = "Fri Mar 10 19:16:22 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Production"

# Import modules
import os
import numpy as np
import spectral as spy
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

def cr_base(x, y):
    '''
    Apply continuum removal to an input spectrum.
    
    Parameters
    ----------
    x : numpy.array
        Array containing the wavelengths.
    y : numpy.array
        Array containing reflectance (or similar) values.

    Returns
    -------
    cr : numpy.array
        Continuum removed spectrum.
    '''
    pts = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)),
                         axis = 1)
    augmented = np.concatenate([pts, [(x[0], np.min(y) - 1),
                                      (x[-1], np.min(y) - 1)]],
                               axis = 0)
    conv_hull = ConvexHull(augmented)
    continuum_points = pts[np.sort([v for v in conv_hull.vertices \
        if v < len(pts)])]
    continuum_function = interp1d(*continuum_points.T)
    denominator = continuum_function(x)
    denominator[denominator == 0] = 1e-10
    cr = y / denominator
    return cr

def remove_continuum(input_raster, output_raster):
    '''
    Apply continuum removal to all pixels of a hyperspectral input raster and
    save the result as a new hyperspectral image.
    
    Parameters
    ----------
    input_raster : str
        Full path to the hyperspectral ENVI image.
    output_raster : str
        Full path to the output raster.

    Returns
    -------
    0
    '''
    # Set paths to input and output data and header files
    p_in_base, in_ext = os.path.splitext(input_raster)
    p_in_header = p_in_base + ".hdr"
    
    p_out_base, out_ext = os.path.splitext(output_raster)
    p_out_header = p_out_base + ".hdr"
    
    # Open input raster and extract wavelengths
    in_raster = spy.open_image(p_in_header)
    wls = np.array(in_raster.bands.centers)
    meta = in_raster.metadata
    
    r, c, b = in_raster.shape
    
    # Create output raster
    out_raster = spy.envi.create_image(p_out_header, metadata = meta)
    
    # Write to output raster using a memmap interface
    mm = out_raster.open_memmap(writable = True)
    
    for i in range(r):
        for j in range(c):
            pixel = in_raster.read_pixel(i, j)
            mm[i, j] = cr_base(wls, pixel)
    
    return 0