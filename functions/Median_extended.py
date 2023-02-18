#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 23:46:25 2023
"""
__author__ = "Manuel"
__date__ = "Fri Feb 17 23:46:25 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Production"

import numpy as np

def _midvals(a, n = 3, fix_dtype = True):
    '''
    Calculate the median (if n = 1) or the average of either the n central
    values (if both len(a) and n are of the same parity) or the n + 1 central
    values (if len(a) and n are of alternating parity).

    Parameters
    ----------
    a : numpy.array
        Array of input values.
    n : int, optional
        Number of central values to average. The default is 3.
    fix_dtype : bool, optional
        Keep input dtype. The default is True.
    
    Returns
    -------
    out : a.dtype
        Output value.
    
    Note
    ----
    This function does not account for array shape. Use the midvals function
    to summarise along axes of a numpy.array.
    '''
    l = len(a)
    fst = (l - n) // 2
    s = np.sort(a, kind = "quicksort")
    lst = fst + n if l & 0x1 == n & 0x1 else fst + n + 1
    c = s[fst : lst]
    out = np.mean(c).astype(a.dtype) if fix_dtype else np.mean(c)
    return out

def midvals(a, n = 3, axis = None, fix_dtype = True):
    '''
    Calculate the median (if n = 1) or the average of either the n central
    values (if both len(a) and n are of the same parity) or the n + 1 central
    values (if len(a) and n are of alternating parity).

    Parameters
    ----------
    a : numpy.array
        Array of input values.
    n : int, optional
        Number of central values to average. The default is 3.
    axis : bool, optional
        Select axis along which the _midvals function is to be applied. If
        no axis is selected, the function is applied to all values of the
        input array. In this case, the output is a single value. The default
        is None.
    fix_dtype : bool, optional
        Keep input dtype. The default is True.
    
    Returns
    -------
    out : ndarray or scalar
        Output array. Either a scalar (in case no axis is selected), or an
        array with a similar shape as the input array, except along the 'axis'
        dimension. In case 'out' is an array, it has one fewer dimension than
        the input array.
    '''
    out = _midvals(a, n, fix_dtype) if axis is None else np.apply_along_axis(
        _midvals, axis = axis, arr = a, n = n, fix_dtype = fix_dtype)
    return out
