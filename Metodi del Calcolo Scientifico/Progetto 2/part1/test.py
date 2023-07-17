import numpy as np
from  scipy import fftpack
import pandas as pd
def scientific(x):
    return np.format_float_scientific(np.float32(x), precision=2)
def main():
    arr = np.array([ [ 231, 32, 233, 161, 24, 71, 140, 245],
                   [247, 40, 248, 245, 124, 204, 36, 107],
                   [234, 202, 245, 167, 9, 217, 239, 173],
                   [193, 190, 100, 167, 43, 180, 8, 70],
                   [11, 24, 210, 177, 81, 243, 8, 112],
                   [97, 195, 203, 47, 125, 114, 165, 181],
                   [193, 70, 174, 167, 41, 30, 127, 245],
                   [87, 149, 57, 192, 65, 129, 178, 228]])
    arr_dct2 = fftpack.dct(fftpack.dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')
    arr_dct = fftpack.dct(arr[0,:], norm='ortho')
    np.savetxt("dct2.csv", arr_dct2, delimiter=",", fmt="%.2e")
    np.savetxt("dct.csv", arr_dct, delimiter=",", fmt="%.2e")
    print(arr_dct.astype(np.float32))



if __name__ == '__main__':
    main()