import numpy as np
import time
import pandas as pd
from numpy import empty, arange, exp, real, pi
from numpy.fft import rfft
import matplotlib.pyplot as plt
from scipy import fftpack

def dct(x):
    # Calcola la trasformata discreta del coseno (DCT) di un vettore x
    N = len(x)
    x2 = empty(2 * N, float)
    x2[:N] = x[:]
    x2[N:] = x[::-1]

    X = rfft(x2)
    phi = exp(-1j * pi * arange(N) / (2 * N))
    return real(phi * X[:N])

def dct2(x):
    # Calcola la DCT bidimensionale di una matrice x
    M = x.shape[0]
    N = x.shape[1]
    a = empty([M, N], float)
    X = empty([M, N], float)

    for i in range(M):
        a[i, :] = dct(x[i, :])
    for j in range(N):
        X[:, j] = dct(a[:, j])
    return X

def main():
    np.random.seed(42)
    dim = [500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000]
    dct2_times = []
    scipy_times = []

    for i in dim:
        data = np.random.randint(low=0, high=255, size=(i, i))

        start1 = time.time()
        _ = dct2(data)
        end1 = time.time() - start1
        dct2_times.append(end1)

        start2 = time.time()
        _ = fftpack.dct(fftpack.dct(data, axis=0, norm='ortho'), axis=1, norm='ortho')
        end2 = time.time() - start2
        scipy_times.append(end2)

        print(i)

    # Grafico
    plt.plot(dim, dct2_times, label='dct2', marker = 'o')
    plt.plot(dim, scipy_times, label='scipy', marker = 'o')
    plt.xlabel('Dimensione')
    plt.ylabel('Tempo (s)')
    plt.legend()
    plt.show()

    #Dataframe
    df = pd.DataFrame({'Dimensione':dim, 'Dct2(s)':dct2_times, 'Scipy(s)':scipy_times})
    df.to_excel("times.xlsx")

if __name__ == "__main__":
    main()
