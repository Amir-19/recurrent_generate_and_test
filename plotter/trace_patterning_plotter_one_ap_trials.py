import numpy as np
import matplotlib.pyplot as plt

def plotter():
    plot_w = 29
    with open('../result/trace_patterning/trial_pn.npy', 'rb') as f:
        pos_trial = np.load(f)
        neg_trial = np.load(f)
        ideal_pos = np.load(f)
        ideal_neg = np.load(f)

    num_plot = len(pos_trial)
    legend = []
    fig, axs = plt.subplots(4, sharex=True, sharey=True)

    axs[0].step(np.arange(num_plot), np.array(pos_trial).flatten(), where='post',color='purple')
    axs[1].step(np.arange(num_plot), np.array(ideal_pos).flatten(), where='post',color='green')
    axs[2].step(np.arange(num_plot), np.array(neg_trial).flatten(), where='post',color='blue')
    axs[3].step(np.arange(num_plot), np.array(ideal_neg).flatten(), where='post',color='red')
    axs[0].legend(['prediction'])
    axs[1].legend(["ideal prediction"])
    axs[2].legend(["prediction"])
    axs[3].legend(["ideal prediction"])
    plt.show()
    #print(var_log[t3-1,17:19])
if __name__ == "__main__":
    plotter()