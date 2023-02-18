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

def midvals(a, n = 3, fix_dtype = True):
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
    '''
    l = len(a)
    fst = (l - n) // 2
    s = np.sort(a, kind = "quicksort")
    lst = fst + n if l & 0x1 == n & 0x1 else fst + n + 1
    c = s[fst : lst]
    return np.mean(c)