import numpy as np
import matplotlib.pyplot as plt

def plotter():
    plot_w = 1
    with open('./result/trace_conditioning/trace_conditioning_deep_traces.npy', 'rb') as f:
        w_log = np.load(f)
        stepsize_log = np.load(f)
        wtrace_log = np.load(f)
        t3 = np.load(f)

    legend = []
    #interest_plot = [0]
    #interest_plot = [1]
    #interest_plot = [2,3,4,5,6,7,8,9,10,11]
    #interest_plot = [12,13,14,15,16,17,18,19,20]
    # interest_plot = []
    # for i in range(12,48):
    #     interest_plot.append(i)
    interest_plot = [0,1,2,3,4,5,6,7,8,9,10,11]
    #interest_plot = []
    # for k in range(num_features):
    for k in interest_plot:
        if plot_w == 1:
            plt.plot(w_log[:t3,k])
            legend.append("w" + str(k))
        elif plot_w == 2:
            plt.plot(stepsize_log[:t3,k])
            legend.append("step-size" + str(k))
        elif plot_w == 3:
            plt.plot(wtrace_log[:t3,k])
            legend.append("w_trace" + str(k))   
    plt.legend(legend)
    plt.show()
    #print(var_log[t3-1,17:19])
if __name__ == "__main__":
    plotter()