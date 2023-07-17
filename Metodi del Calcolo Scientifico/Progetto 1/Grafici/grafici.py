import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

def expand(arr):
    repeat = math.floor(800/len(arr))
    new_arr=[val for val in arr for _ in range(repeat)]  
    return new_arr

def shrink(arr):
    win = math.floor(len(arr)/800)
    b = False
    i = 0
    while b==False:
        print((len(arr)-i) % win)
        if (len(arr)-i) % win == 0:
            position_rest_equal_zero = len(arr)-i
            b=True
        else:
            i = i+1
    a = np.array(arr[0:position_rest_equal_zero])
    rest = np.array(arr[position_rest_equal_zero:len(arr)])
    shrinked_arr = np.average(a.reshape(-1, win), axis=1)
    if len(rest)!=0:
        new_arr = np.concatenate((shrinked_arr, rest), axis=0)
    else:
        new_arr = shrinked_arr
    return new_arr

def main():
    #Read csv
    mem_usage_win = pd.read_csv("./windows_memory_usage.csv")
    mem_usage_linux = pd.read_csv("./linux_memory_usage.csv")
    # Creare il grafico
    fig, ax = plt.subplots()
    for i, row in mem_usage_win.iterrows():
        mem_usage = [float(x) for x in row['Memoria'][1:-1].split(',')]
        if len(mem_usage)<400:
            mem_usage = expand(mem_usage)
        if len(mem_usage)>800:
            mem_usage = shrink(mem_usage)
        ax.plot(range(len(mem_usage)), mem_usage, label=row["Nome"], marker='o')
    for i, row in mem_usage_linux.iterrows():
        mem_usage = [float(x) for x in row['Memoria'][1:-1].split(',')]
        if len(mem_usage)<400:
            mem_usage = expand(mem_usage)
        if len(mem_usage)>800:
            mem_usage = shrink(mem_usage)
        ax.plot(range(len(mem_usage)), mem_usage, label=row["Nome"], marker='o')
    ax.set_xlabel("Campionamenti di memoria")
    ax.set_ylabel("Memoria (MiB)")
    ax.set_title("Utilizzo di memoria")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("./memory_usage.png")
    

if __name__ == '__main__':
    main()