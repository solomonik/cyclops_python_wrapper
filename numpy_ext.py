from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


def name():
    return 'numpy'

def cls():
    return np.ndarray

def save_tensor_to_file(T, filename):
    np.save(filename, T)

def load_tensor_from_file(filename):
    try:
        T = np.load(filename)
        print('Loaded tensor from file ', filename)
    except FileNotFoundError:
        raise FileNotFoundError('No tensor exist on: ', filename)
    return T

def TTTP(T, A):
    T_inds = "".join([chr(ord('a')+i) for i in range(T.ndim)])
    einstr = ""
    A2 = []
    for i in range(len(A)):
        if A[i] is not None:
            einstr += chr(ord('a')+i) + chr(ord('a')+T.ndim) + ','
            A2.append(A[i])
    einstr += T_inds + "->" + T_inds
    A2.append(T)
    return np.einsum(einstr, *A2)

def is_master_proc():
    return True

def printf(*string):
    print(*string)

def tensor(shape, sp, *args2):
    return np.ndarray(shape, *args2)

def list_add(list_A, list_B):
    return [A + B for (A, B) in zip(list_A, list_B)]

def scalar_mul(sclr,list_A):
    return [sclr * A for A in list_A]

def mult_lists(list_A,list_B):
    return np.sum(np.sum([A * B for (A, B) in zip(list_A, list_B)]))

def list_vecnormsq(list_A):
    return np.sum([i**2 for i in list_A])

def list_vecnorm(list_A):
    return np.sqrt(np.sum([i**2 for i in list_A]))

def sparse_random(shape, begin, end, sp_frac):
    tensor = np.random.random(shape) * (end - begin) + begin
    mask = np.random.random(shape) < sp_frac
    tensor = tensor * mask
    return tensor

def vecnorm(T):
    return la.norm(np.ravel(T))

def norm(v):
    return la.norm(v)

def dot(A, B):
    return np.dot(A, B)

def eigvalh(A):
    return la.eigvalh(A)

def eigvalsh(A):
    return la.eigvalsh(A)

def svd(A, r=None):
    U, s, VT = la.svd(A, full_matrices=False)
    if r is not None:
        U = U[:,:r]
        s = s[:r]
        VT = VT[:r,:]
    return U, s, VT

def svd_rand(A, r=None):
    return svd(A, r)

def cholesky(A):
    return la.cholesky(A)

def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
    if not from_left:
        return sla.solve_triangular(A.T, B.T, trans=transp_L, lower=not lower).T
    else:
        return sla.solve_triangular(A, B, trans=transp_L, lower=lower)

def einsum(string, *args):
    return np.einsum(string, *args)

#def einsum(string, *args, out=None):
#    if out is None:
#        return np.einsum(string, *args)
#    else:
#        out = np.einsum(string, *args)
#        return out

def ones(shape):
    return np.ones(shape)

def zeros(shape):
    return np.zeros(shape)

def sum(A, axes=None):
    return np.sum(A, axes)

def random(shape):
    return np.random.random(shape)

def seed(seed):
    return np.random.seed(seed)

def asarray(T):
    return np.array(T)

def speye(*args):
    return np.eye(*args)

def eye(*args):
    return np.eye(*args)

def transpose(A):
    return A.T

def argmax(A, axis=0):
    return abs(A).argmax(axis=axis)

def qr(A):
    return la.qr(A)

def reshape(A, shape, order='F'):
    return np.reshape(A, shape, order)

def tensordot(A, B, axes):
    return np.tensordot(A, B, axes)

def astensor(A):
    # if isinstance(A, np.ndarray):
    #     return A
    # if np.isscalar(A):
    #     return np.asarray(A)
    if hasattr(A, 'to_nparray'):
        return A.to_nparray()
    return np.asarray(A)

def einsvd(einstr, A, rank=None, threshold=None, size_limit=None, criterion=None, mult_s=True):
    """
    Perform Singular Value Decomposition according to the specified Einstein notation string. 
    Will always preserve at least one singular value during truncation.

    Parameters
    ----------
    einstr: str
        A string of Einstein notations in the form of 'idxofA->idxofU,idxofV'. There must be one and only one contraction index.

    A: tensor like
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
    """
    str_a, str_uv = einstr.replace(' ', '').split('->')
    str_u, str_v = str_uv.split(',')
    char_i = list(set(str_v) - set(str_a))[0]

    # transpose and reshape A into u.size * v.size and do svd
    u, s, vh = la.svd(np.einsum(str_a + '->' + (str_u + str_v).replace(char_i, ''), A).reshape((-1, np.prod([A.shape[str_a.find(c)] for c in str_v if c != char_i], dtype=int))), full_matrices=False)
    rank = rank or len(s)

    if size_limit is not None:
        if isinstance(size_limit, int):
            size_limit = (size_limit, size_limit)
        rank = min(rank, int(size_limit[0] / u.shape[0]))
        rank = min(rank, int(size_limit[1] / vh.shape[1]))

    if threshold is not None:
        # will always preserve at least one singular value
        if criterion is None:
            rank = 1 if threshold > s[0] else min((rank, len(s) - np.searchsorted(s[::-1], threshold)))
        else:
            threshold = threshold * la.norm(s, criterion)
            for i in range(rank, 0, -1):
                if la.norm(s[i:], criterion) >= threshold:
                    rank = i
                    break;

    if rank < len(s):
        u = u[:,:rank]
        s = s[:rank]
        vh = vh[:rank]

    if mult_s:
        sqrtS = np.diag(s ** 0.5)
        u = np.dot(u, sqrtS)
        vh = np.dot(sqrtS, vh)

    # reshape and transpose u and vh into tgta and tgtb
    u = np.einsum(str_u.replace(char_i, '') + char_i + '->' + str_u, u.reshape([A.shape[str_a.find(c)] for c in str_u if c != char_i] + [-1]))
    vh = np.einsum(char_i + str_v.replace(char_i, '') + '->' + str_v, vh.reshape([-1] + [A.shape[str_a.find(c)] for c in str_v if c != char_i]))
    return u, s, vh

def squeeze(A):
    return A.squeeze()
