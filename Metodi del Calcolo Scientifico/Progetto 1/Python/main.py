import numpy as np
import scipy.io as sio
import time
import os
import csv
from scipy.sparse.linalg import norm
from scipy import linalg
from memory_profiler import memory_usage
import pandas as pd
import matplotlib.pyplot as plt

def solveSystem(A, b):
    from scipy.sparse.linalg import spsolve
    return spsolve(A, b)

def main():
    fields = ["Nome", "Dimensione", "Tempo(s)", "Memoria(MiB)", "Err", "NNZ", "Cond"]
    data = []
    mem_usage_data = []
    mem_check_times = [0.1, 0.1, 1, 1]
    i = 0

    for file in os.listdir("./matrices/"):
        if file.endswith(('.mat')):
            #caricamento matrice
            _name = file
            file = sio.loadmat("./matrices/"+file)
            A = file['Problem']['A'][0, 0]

            _nnz = A.nnz
            _dim = A.shape[1]

            xe = np.ones(A.shape[1])
            b = A * xe

            _cond, _error, _time, _mem = "N/A", "N/A", "N/A", "N/A"
            try:
                #risolvo sistema
                start = time.time()
                x = solveSystem(A, b)
                # controllo tempo e memoria
                _time = time.time() - start
                mem_usage = memory_usage(proc=(solveSystem,(A,b)), interval=mem_check_times[i])
                i=i+1
                _mem = max(mem_usage) - min(mem_usage)
                _opName = _name + "_win"

                mem_usage_data.append({"Nome": _opName, "Memoria": mem_usage})
                #errore
                _error = linalg.norm(xe - x) / linalg.norm(xe)
                #aggiungo risultati
                data.append([_name, _dim, _time, _mem, _error, _nnz, _cond])
            except Exception as e:
                print("Error: {}".format(e))
                data.append([_name, _dim, "N/A", "N/A", "N/A", _nnz, "N/A"])

    with open("./report.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(data)

    # Creo un DataFrame per memory usage
    mem_usage_df = pd.DataFrame(mem_usage_data)
    mem_usage_df.to_csv("./memory_usage.csv", index=False)

if __name__ == '__main__':
    main()

