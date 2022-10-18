import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import timeit
#from timeit import timeit
plt.style.use('fivethirtyeight')

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N,1))
    M = np.exp(-2j * np.pi * k * n/ N)
    return np.dot(M,x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])

def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")
        
    N_min = min(N, 2)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    return X.ravel()

def rader(x, m=None, w=None, a=None):
    # Translated from GNU Octave's czt.m
    n = len(x)
    if m is None: m = n
    if w is None: w = np.exp(-2j * np.pi / m)
    if a is None: a = 1

    chirp = w ** (np.arange(1 - n, max(m, n)) ** 2 / 2.0)
    N2 = int(2 ** np.ceil(np.lib.scimath.log2(m + n - 1)))  # next power of 2
    xp = np.append(x * 1/a ** np.arange(n) * chirp[n - 1 : n + n - 1], np.zeros(N2 - n))
    ichirpp = np.append(1 / chirp[: m + n - 1], np.zeros(N2 - (m + n - 1)))
    r = sc.fft.ifft(sc.fft.fft(xp) * sc.fft.fft(ichirpp))
    return r[n - 1 : m + n - 1] * chirp[n - 1 : m + n - 1]

def signal(input_size):
    # Number of samplepoints
    N = input_size
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    return y

#fft_valori
xx = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152]
yy = [0.0000453, 0.0000749, 0.00012, 0.00021, 0.00038, 0.00072, 0.0031, 0.0036, 0.0067, 0.0119, 0.0223, 0.045, 0.09, 0.18, 0.40, 0.75, 1.48, 3.01, 6.08, 11.99, 24.87]

#dft
x_inputsize = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
y_sec = [0.0000374, 0.0000398, 0.0000442, 0.0000571, 0.00014, 0.015, 0.011, 0.0053, 0.020, 0.084, 0.33, 1.34, 6.05, 77.04]

#czt
xxx = [2,5,7,17,31,67,127,257,521,1031,2069, 4099, 8191, 16381, 32779, 65537, 131071, 262147, 524287, 1048613, 2097211]
yyy = [0.000161, 0.000209, 0.000156, 0.000201, 0.0037, 0.000189, 0.000218, 0.00041, 0.000729, 0.000938, 0.00156, 0.00272, 0.00580, 0.00960, 0.0432, 0.07129, 0.0926, 0.37, 0.4418, 1.4514, 3.14790]
#x = signal(2097211)
#print(timeit.timeit('czt(x)', 'from fft_implementation import czt, x', number=1))

plt.figure(figsize=(16,8))
plt.title('Grafico tempi di esecuzione')
plt.xlabel('size', fontsize=18)
plt.ylabel('time (seconds)', fontsize=18)
plt.plot(x_inputsize, y_sec)
plt.plot(xx,yy)
plt.plot(xxx,yyy)
plt.legend(['dft', 'fft', 'rader'])
plt.show()


"""
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train', 'Real price', 'Predictions'], loc='lower right')
#Comandi da shell
#%timeit dft(x)
#%timeit fft(x)
#%timeit fft_v(x)
#%timeit np.fft.fft(x)
"""

