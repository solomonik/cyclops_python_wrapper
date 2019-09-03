from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import ctf
import numpy as np


def name():
    return 'ctf'

def cls():
    return ctf.tensor

def save_tensor_to_file(T, filename):
    np.save(filename, T.to_nparray())

def load_tensor_from_file(filename):
    try:
        T = np.load(filename)
        print('Loaded tensor from file ', filename)
    except FileNotFoundError:
        raise FileNotFoundError('No tensor exist on: ', filename)
    return ctf.from_nparray(T)

def from_nparray(arr):
    return ctf.from_nparray(arr)

def TTTP(T, A):
    return ctf.TTTP(T, A)

def is_master_proc():
    if ctf.comm().rank() == 0:
        return True
    else:
        return False

def printf(*string):
    if ctf.comm().rank() == 0:
        print(*string)

def tensor(shape, sp, *args):
    return ctf.tensor(shape, sp, *args)
        
def sparse_random(shape, begin, end, sp_frac):
    return ctf.tensor(shape, sp=True).fill_sp_random(begin, end, sp_frac)

def vecnorm(T):
    return ctf.vecnorm(T)

def list_vecnormsq(list_A):
    s = 0
    for t in [i**2 for i in list_A]:
        s += ctf.sum(t)
    return s
    
def list_vecnorm(list_A):
    return list_vecnormsq(list_A)**0.5
    
def mult_lists(list_A,list_B):
    s = 0
    for t in [A * B for (A, B) in zip(list_A, list_B)]:
        s += ctf.sum(t)
    return s
    
def norm(v):
    return v.norm2() 

def dot(A, B):
    return ctf.dot(A, B)

def svd(A, r=None):
    return ctf.svd(A, r)

def svd_rand(A, r):
    return ctf.svd_rand(A, r)

def cholesky(A):
    return ctf.cholesky(A)

def solve_tri(A, B, lower=True, from_left=False, transp_L=False):
    return ctf.solve_tri(A, B, lower, from_left, transp_L)

def einsum(string, *args):
    # doesn't check for invalid use of ellipses
    if '...' in string:
        left = string.split(',')
        left[-1], right = left[-1].split('->')
        symbols = ''.join(list(set(chr(i) for i in range(48, 127)) - set(string.replace('.', '').replace(',', '').replace('->', ''))))
        symbol_idx = 0
        for i, (s, tsr) in enumerate(zip(left, args)):
            num_missing = tsr.ndim - len(s.replace('...', ''))
            left[i] = s.replace("...", symbols[symbol_idx:symbol_idx+num_missing])
            symbol_idx += num_missing
        right = right.replace('...', symbols[:symbol_idx])
        string = ','.join(left) + '->' + right

    return ctf.einsum(string, *args)

def ones(shape):
    return ctf.ones(shape)

def zeros(shape):
    return ctf.zeros(shape)

def sum(A, axes=None):
    return ctf.sum(A, axes)

def random(shape):
    return ctf.random.random(shape)

def seed(seed):
    return ctf.random.seed(seed)

def list_add(list_A, list_B):
    return [A + B for (A, B) in zip(list_A, list_B)]
    
def scalar_mul(sclr, list_A):
    return [sclr * A for A in list_A]

def speye(*args):
    return ctf.speye(*args)

def eye(*args):
    return ctf.eye(*args)

def transpose(A):
    return ctf.transpose(A)

def argmax(A, axis=0):
    return abs(A).to_nparray().argmax(axis=axis)

def asarray(T):
    return ctf.astensor(T)
    
def reshape(A, shape, order='F'):
    return ctf.reshape(A, shape, order)

def tensordot(A, B, axes):
    return ctf.tensordot(A, B, axes)
    
def astensor(A):
    return ctf.astensor(A)

def einsvd(einstr, A, rank=None, threshold=None, size_limit=None, criterion=None, mult_s=True):
    """
    Perform Singular Value Decomposition according to the specified Einstein notation string. 
    Will always preserve at least one singular value during the truncation.

    Parameters
    ----------
    einstr: str
        A string of Einstein notations in the form of 'idxofA->idxofU,idxofV'. There must be one and only one contraction index.

    A: tensor_like
        The tensor to be decomposed. Should be of order 2 or more.

    rank: int or None, optional
        The minimum number of singular values/vectors to preserve. Will influence the actual truncation rank.

    threshold: float or None, optional
        The value used with criterion to decide the cutoff. Will influence the actual truncation rank.

    size_limit: int or tuple or None, optional
        The size limit(s) for both U and V (when specified as a int) or U and V respectively (when specified as a tuple).
        Will influence the actual truncation rank.

    criterion: int or None, optional
        The norm to be used together with threshold to decide the cutoff. Will influence the actual truncation rank.
        When being left as None, the threshold is treated as the plain cutoff value.
        Otherwise, cutoff rank is the largest int satisfies: threshold * norm(s) > norm(s[rank:]).

    mult_s: bool, optional
        Whether or not to multiply U and V by S**0.5 to decompose A into two tensors instead of three. True by default.
        
    Returns
    -------
    u: tensor_like
        A unitary tensor with indices ordered by the Einstein notation string.

    s: 1d tensor_like
        A 1d tensor containing singular values sorted in descending order.

    v: tensor_like
        A unitary tensor with indices ordered by the Einstein notation string.
    """
    str_a, str_uv = einstr.replace(' ', '').split('->')
    str_u, str_v = str_uv.split(',')
    char_i = list(set(str_v) - set(str_a))[0]
    shape_u = np.prod([A.shape[str_a.find(c)] for c in str_u if c != char_i])
    shape_v = np.prod([A.shape[str_a.find(c)] for c in str_v if c != char_i])

    rank = rank or min(shape_u, shape_v)

    if size_limit is not None:
        if np.isscalar(size_limit):
            size_limit = (size_limit, size_limit)
        if size_limit[0] is not None:
            rank = min(rank, int(size_limit[0] / shape_u) or 1)
        if size_limit[1] is not None:
            rank = min(rank, int(size_limit[1] / shape_v) or 1)

    if threshold is None or criterion is None:
        u, s, vh = A.i(str_a).svd(str_u, str_v, rank, threshold)
    else:
        u, s, vh = A.i(str_a).svd(str_u, str_v)
        threshold = threshold * ctf.norm(s, criterion)
        # will always preserve at least one singular value
        for i in range(rank, 0, -1):
            if ctf.norm(s[i-1:], criterion) >= threshold:
                rank = i
                break;
        if rank < s.size:
            u = u[tuple(slice(None) for i in range(str_u.find(char_i))) + (slice(0, rank),)]
            s = s[:rank]
            vh = vh[tuple(slice(None) for i in range(str_v.find(char_i))) + (slice(0, rank),)]

    if mult_s:
        char_s = list(set(string.ascii_letters) - set(str_v))[0]
        sqrtS = ctf.diag(s ** 0.5)
        vh = ctf.einsum(char_s + char_i + ',' + str_v + '->' + str_v.replace(char_i, char_s), sqrtS, vh)
        char_s = list(set(string.ascii_letters) - set(str_u))[0]
        u = ctf.einsum(str_u + ',' + char_s + char_i + '->' + str_u.replace(char_i, char_s), u, sqrtS)

    return u, s, vh

def squeeze(A):
    return A.reshape([s for s in A.shape if s != 1])
