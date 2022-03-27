import numpy as np
import matplotlib.pyplot as plt

def plotter():
    plot_w = 1
    with open('../result/trace_patterning/weight_ss_.npy', 'rb') as f:
        w_log = np.load(f)
        # stepsize_log = np.load(f)
        # syn_log = np.load(f)
        # #w_trace_log = np.load(f)
        # t3 = np.load(f)

    #4396651
    legend = []
    #interest_plot = [0]
    interest_plot = [1,2,3,4,5,6,7,36]
    #interest_plot = [7,8,9,10,11,12,13,14,15,16]
    #interest_plot = [17,18]
    #interest_plot = [19]
    #interest_plot = [17,68,41,56,24,33]
    #interest_plot = [25,17,56,22,39,20,45,38,28,29,47,27]
    #interest_plot = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36]
    # for k in range(num_features):
    if plot_w ==3:
        plt.plot(syn_log[:t3])
        legend.append("number of synapses")
    else:
        for k in interest_plot:
            if plot_w == 1:
                plt.plot((w_log[:4396651,k]))
                #legend.append("w" + str(k))
            elif plot_w == 2:
                plt.plot(stepsize_log[:t3,k])
                legend.append("step-size" + str(k))
            # elif plot_w == 4:
            #     plt.plot((w_trace_log[:t3, k]))
            #     legend.append("w_trace" + str(k))
    #plt.legend(legend)
    plt.show()
    #print(var_log[t3-1,17:19])
if __name__ == "__main__":
    plotter()