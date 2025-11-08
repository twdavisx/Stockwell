import numpy as np
import matplotlib.pyplot as plt
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
GPU version of fDOST
'''''
def fDOSTgpu(h: cp.ndarray):
    n = len(h)
    H = cp.fft.fft(h, norm='ortho')
    S = cp.zeros(n, dtype=complex)

    S[n//2] = H[0]
    S[n//2 + 1] = -H[1] # simplifies to -1: np.exp(-2j*(n//2) * np.pi/n)

    for p in range(2, int(cp.log2(n))):
        b = 2**(p-1)
        v = b + 2**(p-2)
        k = cp.arange(v - b//2, v + b//2)
        tau = cp.arange(b)

        R = cp.exp(-2j*cp.pi*tau/2)
        V = cp.fft.ifft(H[k], norm='ortho')

        T = R*V
        S[n//2+b: n//2+b+b] = T


    S[1:n//2] = cp.conj(S[n//2+1:][::-1])

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

'''''
GPU Version of Stockwell
'''

def stockwellgpu(timeseries: cp.ndarray, minfreq=0, maxfreq=None, freqsamplingrate=1):

    n = len(timeseries)
    if maxfreq is None:
        maxfreq = n // 2

    vector_fft = cp.fft.fft(timeseries)
    vector_fft = cp.concatenate([vector_fft, vector_fft])

    step = int(freqsamplingrate)
    freq_counters = cp.arange(step, (maxfreq - minfreq) + 1, step, dtype=int)
    current_freqs = (minfreq + freq_counters).astype(int)

    current_freqs = current_freqs[current_freqs > 0]

    spe_nelements = 1 + len(current_freqs)

    st_matrix = cp.zeros((spe_nelements, n), dtype=complex)
    st_matrix[0, :] = cp.mean(timeseries) * cp.ones(n)

    if len(current_freqs) == 0:
        return st_matrix

    v_front = (cp.arange(n, dtype=float) ** 2)
    v_back  = (cp.arange(-n, 0, dtype=float) ** 2)
    vector = cp.concatenate([v_front, v_back])

    scales = (-2.0 * cp.pi**2) / (current_freqs[:, None].astype(float) ** 2)
    voice = vector[None, :] * scales

    gauss_windows = cp.exp(voice[:, :n]) + cp.exp(voice[:, n:])

    cols = cp.arange(n)
    idx = current_freqs[:, None] + cols[None, :]
    fft_slices = vector_fft[idx]

    st_rows = cp.fft.ifft(fft_slices * gauss_windows, axis=1)

    st_matrix[1:1 + len(current_freqs), :] = st_rows

    return st_matrix

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
2. plot_fdost
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


def plt_fdost(m, f = None):
    if f == None:
        f = m.shape[0]
    else:
        m = m[:f]
    plt.figure(figsize=(10,6))
    plt.imshow(np.abs(m), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.yscale('log')
    plt.ylim(0.05, f)
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