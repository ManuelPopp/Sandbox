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

#------------------------------------------------------------------------------
# Module imports

import numpy as np

#------------------------------------------------------------------------------
# Base functions (direct application will result in scalar output)

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

def _drop_outer(a, n = 2, fix_dtype = True):
    '''
    Delete n outer values starting by symmetric dropping of minimum and
    maximum values. In case n is an uneven number, this initial removal of
    values is followed by removal of the value farest to the mean of the
    remaining elements.

    Parameters
    ----------
    a : numpy.array
        Array of input values.
    n : int, optional
        Number of values to drop. The default is 2.
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
    s = np.sort(a, kind = "quicksort")
    init_rem = n // 2
    idxp = np.array(range(init_rem))
    idxn = np.negative(idxp) - 1
    idx = list(idxp) + list(idxn)
    b = np.delete(s, idx)
    c = np.delete(b, np.argmax(b - np.mean(b))) if len(a) & 0x1 == 1 else b
    out = np.mean(c).astype(a.dtype) if fix_dtype else np.mean(c)
    return out

def _z_base(x, ax):
    '''
    Calculate z-scores for rows or columns of an array.
    
    Parameters
    ----------
    x : numpy.array
        Input values as numpy.array.
    ax : int {0, 1}
        Axis along which the z-scores are to be calculated.

    Returns
    -------
    numpy.array with one fewer dimension than the input array.
        Z-scores.
    '''
    m = np.apply_along_axis(np.mean, axis = ax, arr = x)
    s = np.apply_along_axis(np.std, axis = ax, arr = x)
    return (x - m) / s

def _compare(x, z, use_mean):
    '''
    Check whether the absolute of values of an array are below a threshold.
    
    Parameters
    ----------
    x : numpy.array
        Input array.
    z : float
        Threshold value.
    use_mean : bool
        If True, the function checks whether the average of the absolutes
        is below the threshold z, else it checks whether all absolutes are
        below the threshold.
    
    Returns
    -------
    out : bool
        Truth value for the selected condidion.
    '''
    out = np.mean(np.abs(x)) < z if use_mean else np.all(np.abs(x) < z)
    return out

#------------------------------------------------------------------------------
# Wrapper functions to make calculations applicable along certain axis of
# numpy.array

def midvals(a, n = 3, axis = None, fix_dtype = True):
    '''
    Calculate the median (if n = 1) or the average of either the n central
    values (if both len(a) and n are of the same parity) or the n + 1 central
    values (if len(a) and n are of alternating parity).

    Parameters
    ----------
    a : numpy.ndarray
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

def drop_outer(a, n = 2, axis = None, fix_dtype = True):
    '''
    Delete n outer values starting by symmetric dropping of minimum and
    maximum values. In case n is an uneven number, this initial removal of
    values is followed by removal of the value farest to the mean of the
    remaining elements. The mean is calculated on the elements remaining after
    these removal steps.

    Parameters
    ----------
    a : numpy.array
        Array of input values.
    n : int, optional
        Number of values to drop. The default is 2.
    axis : bool, optional
        Select axis along which the _midvals function is to be applied. If
        no axis is selected, the function is applied to all values of the
        input array. In this case, the output is a single value. The default
        is None.
    fix_dtype : bool, optional
        Keep input dtype. The default is True.

    Returns
    -------
    out : a.dtype
        Output value.
    
    Returns
    -------
    out : ndarray or scalar
        Output array. Either a scalar (in case no axis is selected), or an
        array with a similar shape as the input array, except along the 'axis'
        dimension. In case 'out' is an array, it has one fewer dimension than
        the input array.
    '''
    out = _drop_outer(a, n, fix_dtype) if axis is None else \
        np.apply_along_axis(_drop_outer, axis = axis, arr = a, n = n,
                            fix_dtype = fix_dtype)
    return out

def z_score(a, z = 3., axis = 0, fix_dtype = True, remove_by_mean = False):
    '''
    Delete outliers based on z-score.

    Parameters
    ----------
    a : numpy.array
        Array of input values.
    z : float, optional
        Z-score to set as boundary. The default is 3.
    axis : bool, optional
        Select axis along which the _midvals function is to be applied. If
        no axis is selected, the function is applied to all values of the
        input array. In this case, the output is a single value. The default
        is None.
    fix_dtype : bool, optional
        Keep input dtype. The default is True.

    Returns
    -------
    out : a.dtype
        Output value.
    
    Returns
    -------
    out : ndarray or scalar
        Output array. Either a scalar (in case no axis is selected), or an
        array with a similar shape as the input array, except along the 'axis'
        dimension. In case 'out' is an array, it has one fewer dimension than
        the input array.
    '''
    a = a.astype("float")
    b = a.copy()
    
    len_0 = a.shape[1] if axis == 0 else a.shape[0]
    for v in range(len_0):
        zs = _z_base(a[:,v], axis) if axis == 0 else _z_base(a[v,:], axis)
        b[:,v] = zs
    
    ax = 0 if axis == 1 else 1
    indices = np.apply_along_axis(_compare, axis = ax, arr = b, z = z,
                                  use_mean = remove_by_mean)
    c = a[:,indices] if ax == 0 else a[indices,:]
    mean_c = np.apply_along_axis(np.mean, axis = axis, arr = c)
    out = mean_c.astype(a.dtype) if fix_dtype else mean_c
    return out
