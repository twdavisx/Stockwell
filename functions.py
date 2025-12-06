import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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

def stockwell(timeseries: np.ndarray, minfreq=0, maxfreq=None, freqsamplingrate=1):

    n = len(timeseries)
    if maxfreq is None:
        maxfreq = n // 2

    vector_fft = np.fft.fft(timeseries)
    vector_fft = np.concatenate([vector_fft, vector_fft])

    step = int(freqsamplingrate)
    freq_counters = np.arange(step, (maxfreq - minfreq) + 1, step, dtype=int)
    current_freqs = (minfreq + freq_counters).astype(int)

    current_freqs = current_freqs[current_freqs > 0]

    spe_nelements = 1 + len(current_freqs)

    st_matrix = np.zeros((spe_nelements, n), dtype=complex)
    st_matrix[0, :] = np.mean(timeseries) * np.ones(n)

    if len(current_freqs) == 0:
        return st_matrix

    v_front = (np.arange(n, dtype=float) ** 2)
    v_back  = (np.arange(-n, 0, dtype=float) ** 2)
    vector = np.concatenate([v_front, v_back])

    scales = (-2.0 * np.pi**2) / (current_freqs[:, None].astype(float) ** 2)
    voice = vector[None, :] * scales

    gauss_windows = np.exp(voice[:, :n]) + np.exp(voice[:, n:])

    cols = np.arange(n)
    idx = current_freqs[:, None] + cols[None, :]
    fft_slices = vector_fft[idx]

    st_rows = np.fft.ifft(fft_slices * gauss_windows, axis=1)

    st_matrix[1:1 + len(current_freqs), :] = st_rows

    return st_matrix

def stockwell_chunked_vectorized(
    x: np.ndarray,
    dt: float,
    kmin: int = 0,
    kmax: Optional[int] = None,
    kstep: int = 1,
    chunk_size: int = 256,
    out_dtype=np.complex128,
    mmap_path: Optional[str] = None,
):

    x = np.asarray(x)
    n = x.size

    if kmax is None:
        kmax = n // 2

    # Build frequency bins
    k_bins = np.arange(kmin, kmax + 1, kstep, dtype=int)
    num_k  = len(k_bins)

    # Output matrix (optionally memmapped)
    if mmap_path is None:
        S = np.zeros((num_k, n), dtype=out_dtype)
    else:
        S = np.memmap(mmap_path, mode="w+", dtype=out_dtype, shape=(num_k, n))

    # Handle DC component
    row_cursor = 0
    if kmin == 0:
        S[0, :] = np.mean(x)
        k_bins = k_bins[1:]
        num_k  = len(k_bins)
        row_cursor = 1

    # Compute FFT once and duplicate to allow wrap slicing
    X = np.fft.fft(x)
    X2 = np.concatenate([X, X])     # length 2n (for no-wrap slicing)

    # Precompute t² vectors identical to original implementation
    t_front = np.arange(n, dtype=float)**2       # 0^2, 1^2, ..., (n-1)^2
    t_back  = np.arange(-n, 0, dtype=float)**2   # n^2, ..., 1^2
    t_vec   = np.concatenate([t_front, t_back])  # length 2n (MATCHES original)

    # Time index array for slicing FFT windows
    cols = np.arange(n)

    # Process frequency bins in chunks
    for i in range(0, num_k, chunk_size):

        k_batch = k_bins[i : i + chunk_size]
        B = len(k_batch)
        if B == 0:
            break

        # Gaussian scaling term: -2π² / k²
        scale = -2.0 * (np.pi ** 2) / (k_batch.astype(float) ** 2)

        # Build Gaussian window EXACTLY matching original:
        # gauss[b,:] = exp(scale[b] * t_front) + exp(scale[b] * t_back)
        voice = scale[:, None] * t_vec[None, :]    # shape (B, 2n)
        gauss = np.exp(voice[:, :n]) + np.exp(voice[:, n:])

        # Vectorized FFT slicing:
        # For each k0, take X2[k0 : k0+n]
        idx_start = k_batch[:, None] + cols[None, :]   # shape (B, n)
        slices = X2[idx_start]                         # <------ VECTORIZED

        # Apply Gaussian window in frequency domain, then inverse FFT
        rows = np.fft.ifft(slices * gauss, axis=1)

        # Store into output
        S[row_cursor:row_cursor + B, :] = rows.astype(out_dtype, copy=False)
        row_cursor += B

        f = kmax/(n*dt)
        
    return S, f

'''''
Function given by professor:

def st(timeseries, minfreq=0, maxfreq=len(timeseries)//2, freqsamplingrate=1):

    vector_fft = np.fft.fft(timeseries)
    vector_fft = np.concatenate([vector_fft, vector_fft])
    st_matrix = np.zeros((spe_nelements, n), dtype=complex)
    st_matrix[0, :] = np.mean(timeseries) * np.ones(n)

    freq_idx = 1
    freq_counter = int(freqsamplingrate)

    while freq_counter <= (maxfreq - minfreq) and freq_idx < spe_nelements:
        current_freq = minfreq + freq_counter
        
        if current_freq > 0:  # Skip zero frequency as it's handled above
            # Create frequency domain indices
            vector = np.zeros(2 * n)
            vector[:n] = np.arange(n) ** 2
            vector[n:] = np.arange(-n, 0) ** 2
            # Apply Gaussian scaling
            voice_vector = vector * (-1 * 2 * np.pi**2 / current_freq**2)
            # Compute Gaussian window
            gauss_window = np.exp(voice_vector[:n]) + np.exp(voice_vector[n:])
            
            fft_slice = vector_fft[int(current_freq):int(current_freq) + n]
            st_matrix[freq_idx, :] = np.fft.ifft(fft_slice * gauss_window)
            freq_idx += 1
        
        freq_counter += int(freqsamplingrate)
    return st_matrix

def stockwell_chunked(
    timeseries: np.ndarray,
    dt: float,
    kmin: int = 0,
    kmax: Optional[int] = None,
    kstep: int = 1,
    chunk_size: int = 256,
    out_dtype=np.complex128,
    mmap_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frequency-chunked Stockwell Transform (S-transform) closely following your logic.

    Parameters
    ----------
    timeseries : 1D np.ndarray (real or complex)
        Input signal of length n.
    dt : float
        Sampling interval (seconds per sample).
    kmin, kmax : int
        Frequency-bin index range to compute. k=0 is DC. Default kmax = n//2.
        NOTE: bin->Hz: f[k] = k / (n*dt)
    kstep : int
        Step in frequency-bin indices (e.g., 1 = every bin; 2 = every other bin).
    chunk_size : int
        Number of frequency bins to process at once (batch size).
    out_dtype : numpy dtype
        Output dtype for the S-transform matrix (complex).
    mmap_path : str or None
        If provided, results are written to a memory-mapped file on disk to limit RAM.
        The file will store a (num_k, n) complex array.

    Returns
    -------
    f_hz : 1D np.ndarray of shape (num_k,)
        Frequencies (Hz) corresponding to S-transform rows.
    S : 2D np.ndarray of shape (num_k, n), complex
        S-transform matrix. If mmap_path is provided, this is a memmap array.
        Row order matches f_hz.
    """
    x = np.asarray(timeseries)
    n = x.size
    if kmax is None:
        kmax = n // 2

    # Build the list of k bins to compute
    k_bins = np.arange(kmin, kmax + 1, kstep, dtype=int)
    num_k = k_bins.size

    # Output allocation (optionally memmap to avoid huge RAM usage)
    if mmap_path is None:
        S = np.zeros((num_k, n), dtype=out_dtype)
    else:
        S = np.memmap(mmap_path, mode='w+', dtype=out_dtype, shape=(num_k, n))

    # DC row handling: S[0,:] is the mean repeated if kmin == 0
    start_row = 0
    if kmin == 0:
        S[0, :] = np.mean(x)
        # Start computing from next frequency
        k_bins = k_bins[1:]
        num_k = k_bins.size
        start_row = 1

    # FFT once; duplicate to allow easy n-length slices from any start k
    X = np.fft.fft(x)
    X2 = np.concatenate([X, X])  # length 2n for wrap-free slicing of length n

    # Precompute index-squared arrays once (avoid rebuilding in the loop)
    idx = np.arange(n)
    idx2 = idx ** 2
    neg_idx2 = np.arange(-n, 0) ** 2

    # Process in frequency chunks
    row_cursor = start_row
    for i in range(0, num_k, chunk_size):
        k_batch = k_bins[i : i + chunk_size]           # shape (B,)
        B = k_batch.size
        if B == 0:
            break

        # For S-transform, Gaussian width ~ 1/k. Implement the classic factor:
        # exponent scale = - 2*pi^2 / k^2   (works in "bin index" domain)
        # Handle any accidental zeros (shouldn't happen since we skipped k=0)
        scale = -2.0 * (np.pi ** 2) / np.maximum(k_batch.astype(float), 1.0) ** 2  # shape (B,)

        # Build Gaussian windows for the batch with broadcasting:
        # gauss[b, :] = exp(scale[b]*idx^2) + exp(scale[b]*(neg_idx)^2)
        # Shape: (B, n)
        gauss_pos = np.exp(scale[:, None] * idx2[None, :])
        gauss_neg = np.exp(scale[:, None] * neg_idx2[None, :])
        gauss = gauss_pos + gauss_neg

        # Assemble spectral slices for the batch.
        # Each row b takes X2[k : k+n] for k=k_batch[b]
        # Using a Python list here is fine since B (chunk_size) is modest.
        slices = np.empty((B, n), dtype=X2.dtype)
        for b, k0 in enumerate(k_batch):
            slices[b, :] = X2[k0 : k0 + n]

        # Window in frequency domain, then inverse FFT along time axis (axis=1)
        # Result is the S-transform rows for this chunk
        rows = np.fft.ifft(slices * gauss, axis=1)

        # Store into output (casting to desired dtype)
        S[row_cursor : row_cursor + B, :] = rows.astype(out_dtype, copy=False)
        row_cursor += B

    # Frequencies (Hz) corresponding to each row (including k=0 if present)
    if kmin == 0:
        all_k = np.concatenate([[0], k_bins])
    else:
        all_k = k_bins
    f_hz = all_k / (n * dt)

    t_sec = np.arange(n) * dt

    return S, t_sec, f_hz
'''


'''''
Plotting function also given by professor
'''

def plot_st(st_matrix: np.ndarray, t, f, title="Stockwell Transform"):
    """
    Plot the Stockwell Transform as a time-frequency representation.
    
    Parameters:
    -----------
    st_matrix : ndarray
        Complex matrix from the Stockwell transform
    t : ndarray
        Time vector
    f : ndarray
        Frequency vector
    title : str
        Plot title
    """
    
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(t, f, np.abs(st_matrix), shading='auto')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.ylim(0.05, np.max(f))
    plt.title(title)
    plt.tight_layout()
    plt.show()

'''''
Visualising FDOST functions
1. fdost2m
    turns the n coefficients into an n/2 x n matrix. 
    Only positive values. 
    Now takes in max f to speed up comuptation.
2. plot_st
    plotting function for the matrix
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

'''''
CHATGPT orthogonality test
'''
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