import numpy as np
import matplotlib.pyplot as plt


def plotter():
    patterning = 2
    if patterning == 1:
        with open('../result/trace_patterning/rmse.npy', 'rb') as f:
            rmse = np.load(f)
    else:
        with open('../result/trace_conditioning/rmse.npy', 'rb') as f:
            rmse = np.load(f)

    num_plot = len(rmse)
    plt.plot(np.arange(num_plot), np.array(rmse).flatten())
    plt.show()

    # print(var_log[t3-1,17:19])
if __name__ == "__main__":
    plotter()
