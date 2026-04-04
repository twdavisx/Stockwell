import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import cupy as cp

'''''
All functions created for Nweke Research Lab.
'''

''''
fDOST for Real valued input only. If real valued, negative frequencies are conjugate symetric
Diagonal Ramp * IFFT matrix version
Runtime O(NlogN)
'''''
def fDOST(h: np.ndarray):
    n = len(h)
    H = np.fft.fft(h, norm='ortho')
    S = np.zeros(n, dtype=complex)

    S[n//2] = H[0]
    S[n//2 + 1] = -H[1] # simplifies to -1: np.exp(-2j*(n//2) * np.pi/n)

    for p in range(2, int(np.log2(n))):
        b = 2**(p-1)
        v = b + 2**(p-2)
        k = np.arange(v - b//2, v + b//2)
        tau = np.arange(b)

        R = np.exp(-2j*np.pi*tau/2)
        V = np.fft.ifft(H[k], norm='ortho')

        T = R*V
        S[n//2+b: n//2+b+b] = T


    S[1:n//2] = np.conj(S[n//2+1:][::-1])

    S[0] = H[n//2]

    return S

'''''
Previous versions to build NlogN version


Vectorized:
def fDOST(h: np.ndarray):
    n = len(h)
    H = np.fft.fft(h, norm='ortho')
    S = np.zeros(n, dtype=complex)

    S[n//2] = H[0]
    S[n//2 + 1] = -H[1] # simplifies to -1: np.exp(-2j*(n//2) * np.pi/n)

    for p in range(2, int(np.log2(n))):
        b = 2**(p-1)
        v = b + 2**(p-2)
        tau = np.arange(b)
        k = np.arange(v - b//2, v + b//2)
        exp_matrix = np.exp(2j*np.pi*np.outer(tau,k)/b)
        U = ((np.exp(-1j*np.pi*tau)*exp_matrix) @ H[k])* (1/np.sqrt(b))
        S[n//2+b: n//2+b+b] = U
    
    S[1:n//2] = np.conj(S[n//2+1:][::-1])

    S[0] = H[n//2]

    return S

  
Non Vectorized:
def fDOST(h):
    n = len(h)
    H = np.fft.fft(h, norm='ortho')
    S = np.zeros(n, dtype=complex)

    S[n//2] = H[0]
    S[n//2 + 1] = -H[1] # simplifies to -1: np.exp(-2j*(n//2) * np.pi/n)

    for p in range(2, int(np.log2(n))):
        b = 2**(p-1)
        v = b + 2**(p-2)
        for tau in range(b):
            s = 0j
            for k in range(v - b//2, v + b//2):
                s += np.exp(2j*np.pi*tau*k/b) * H[k] * np.exp(-1j*np.pi*tau)
            s /= np.sqrt(b)
            S[n//2 + b] = s
    
    S[1:n//2] = np.conj(S[n//2+1:][::-1])

    S[0] = H[n//2]

    return S
'''''


'''''
Professor given Stockwell Transform vectorized
'''

def stockwell(
    x: np.ndarray,
    dt: float,
    kmin: int = 0,
    kmax: Optional[int] = None,
    kstep: int = 1,
    k_bins: Optional[np.ndarray] = None,
    chunk_size: int = 256,
    mmap_path: Optional[str] = None,
    decimate: int = 1,
):
    real = np.float32
    out_dtype = np.complex64

    x = np.asarray(x, dtype=real)
    n = x.size

    if kmax is None:
        kmax = n // 2

    if k_bins is None:
        k_bins = np.arange(kmin, kmax + 1, kstep, dtype=int)

    kmin = int(k_bins[0])
    num_k  = k_bins.size

    n_out = len(range(0, n, decimate))

    if mmap_path is None:
        S = np.zeros((num_k, n_out), dtype=out_dtype)
    else:
        S = np.memmap(mmap_path, mode="w+", dtype=out_dtype, shape=(num_k, n_out))

    row_cursor = (kmin==0)
    if kmin == 0:
        S[0, :] = np.mean(x).astype(real)
        k_bins = k_bins[1:]
        num_k-=1

    X = np.fft.fft(x)
    X2 = np.concatenate([X, X])

    t_front = np.arange(n, dtype=np.float32)**2
    t_back  = np.arange(-n, 0, dtype=np.float32)**2

    cols = np.arange(n)

    for i in range(0, num_k, chunk_size):

        k_batch = k_bins[i : i + chunk_size]
        B = k_batch.size
        if B == 0:
            break

        scale = -2.0 * (np.pi ** 2) / (k_batch.astype(np.float32) ** 2)

        s = scale[:, None]

        gauss = np.exp(s * t_front[None, :], dtype=np.float32) + np.exp(s * t_back[None, :], dtype=np.float32)

        idx_start = k_batch[:, None] + cols[None, :]
        slices = X2[idx_start]

        rows = np.fft.ifft(slices * gauss, axis=1)
        S[row_cursor:row_cursor + B, :] = rows[:, ::decimate].astype(out_dtype, copy=False)
        row_cursor += B

    f_hz = kmax / (n * dt)

    return S, f_hz


#stockwell function optimized for GPU, version used for benchmarking and optimization
@cp.fuse(kernel_name='gauss_kernel')
def fused_gauss(k, tf, tb):
    s = -19.7392088 / (k * k)
    return cp.exp(s * tf) + cp.exp(s * tb)

def stockwell_GPU(
    x: cp.ndarray,
    dt: float,
    kmax: int,
    k_bins: cp.ndarray,
    chunk_size: int = 256,
    decimate: int = 1,
):
    
    real = cp.float32
    out_dtype = cp.complex64

    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()

    x = cp.asarray(x, dtype=real)
    n = x.size

    with stream1:
        if kmax is None:
            kmax = n // 2
            
        kmin = int(k_bins[0])
        num_k  = k_bins.size
        n_out = len(range(0, n, decimate))
        S = cp.zeros((num_k, n_out), dtype=out_dtype)
    
        row_cursor = (kmin==0)
        if kmin == 0:
            S[0, :] = cp.mean(x).astype(real)
            k_bins = k_bins[1:]
            num_k-=1
        t_front = cp.arange(n, dtype=cp.float32)**2
        t_back  = cp.arange(-n, 0, dtype=cp.float32)**2
        cols = cp.arange(n)

    with stream2:
        X = cp.fft.fft(x)
        X2 = cp.concatenate([X, X])

    stream1.synchronize()
    stream2.synchronize()

        
    for i in range(0, num_k, chunk_size):
        
        k_batch = k_bins[i : i + chunk_size]
        B = k_batch.size
        
        if B == 0:
            break
            
        gauss = fused_gauss(k_batch[:, None], t_front[None, :], t_back[None, :])
        
        idx_start = k_batch[:, None] + cols[None, :]
        slices = X2[idx_start]
        rows = cp.fft.ifft(slices * gauss, axis=1)
        
        S[row_cursor:row_cursor + B, :] = rows[:, ::decimate].astype(out_dtype, copy=False)
        row_cursor += B
        
    f_hz = kmax / (n * dt)
    
    return S, f_hz


'''''
Visualising FDOST functions
1. fdost2m
    turns the n coefficients into an n/2 x n matrix. 
    Only positive values. 
    Now takes in max f to speed up comuptation.
2. plot_st
    plotting matrix given linear bins
3. plot_2
    plotting matrix given log bins
'''

def fdost2m(arr: np.ndarray, f: int = None):
    #positive values only
    n = len(arr)
    m = np.zeros((n, n*2), dtype=complex)

    m[0] = arr[0]
    m[1] = arr[1]

    index = 2
    n*=2
    for i in range(1,int(np.log2(n//2))):
        k = 2**i
        j = n//k
        for p in range(k):
            m[index:index+k, j*p:j*(p+1)] = arr[index+p]
        index+=k
        if k != None:
            if index >= f:
                break
    
    if k != None:
        m = m[:f]

    return m

def plot_st(st_matrix, f, title="Stockwell Transform"):
    plt.figure(figsize=(12, 8))

    extent = [0, st_matrix.shape[1], 0, f]

    plt.imshow(
        np.abs(st_matrix),
        extent=extent,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )

    plt.colorbar(label="Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.yscale("log")
    plt.ylim(max(0.05, 0), f)

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_2(st_matrix, f_bins, title="Stockwell Transform", cap=1):
    st = np.abs(st_matrix)
    vmax = np.nanpercentile(st, 100 - cap)
    vmin = np.nanpercentile(st, cap)

    t = np.arange(st_matrix.shape[1])

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(t, f_bins, st, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time (samples)")
    plt.ylabel("Frequency (Hz)")
    plt.yscale("log")
    plt.ylim(f_bins[f_bins > 0].min(), f_bins.max())
    plt.title(title)
    plt.tight_layout()
    plt.show()

'''''
Test Functions:
    1. Parsevals
    2. orthogonality
'''

'''''
Parsavels Theorem function
'''
def Parsevals(h, func):
    S = func(h)
    energy_h = np.sum(np.abs(h)**2)
    energy_S = np.sum(np.abs(S)**2)
    print(f"Energy: ||h||² = {energy_h:.4f}, ||S||² = {energy_S:.4f}")
    print(f"Energy preserved: {np.isclose(energy_h, energy_S)}")
    return S


def test_basis_orthogonality(n, dost_func):
    """
    Check whether the FDOST operator is orthonormal by applying it
    to all basis vectors of length n.
    """
    I = np.eye(n)
    # Each row = DOST(e_i)
    basis = np.array([dost_func(I[:, i]) for i in range(n)])  # shape (n, n)

    # Gram matrix of rows
    G = basis @ basis.conj().T

    # Metrics
    err_mat = G - np.eye(n)
    offdiag_mask = ~np.eye(n, dtype=bool)
    total_offdiag = np.sum(np.abs(err_mat[offdiag_mask]))
    max_offdiag   = np.max(np.abs(err_mat[offdiag_mask])) if n > 1 else 0.0
    max_diag_dev  = np.max(np.abs(np.diag(G) - 1))

    print(f"Total off-diagonal energy: {total_offdiag:.2e}")
    print(f"Max off-diagonal entry:    {max_offdiag:.2e}")
    print(f"Max diag deviation:        {max_diag_dev:.2e}")

'''''
Other:
'''

def medianFilter(h, n):
    for _ in range(n):
        for i in range(1,h.size-1):
            h[i] = (h[i-1]+h[i+1])/2
    
    return h
