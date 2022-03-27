import numpy as np
import matplotlib.pyplot as plt


def plotter():
    with open('../result/trace_conditioning/trial.npy', 'rb') as f:
    #with open('./result/trace_conditioning/trace_conditioning_trial_deep_traces.npy', 'rb') as f:
        pred = np.load(f)
        csus = np.load(f)
        tderror = np.load(f)

    num_plot = len(pred)

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    axs[0].step(np.arange(num_plot), np.array(pred).flatten(), where='post')
    axs[1].step(np.arange(num_plot), np.array(csus).flatten(), where='post')
    axs[2].step(np.arange(num_plot), np.array(tderror).flatten(), where='post')
    axs[0].legend(['prediction'])
    axs[1].legend(["CS US"])
    axs[2].legend(["td_error"])
    plt.show()

    # print(var_log[t3-1,17:19])
if __name__ == "__main__":
    plotter()
