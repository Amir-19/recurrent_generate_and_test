import numpy as np
from environment.classical_conditioning_suite import TraceConditioning, compute_return_error
from agent.trace_conditioning_generate_and_test_random import AgentStateConstruction
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from utils.experiment_settings import ExperimentSetting

def augment_observation(o, env, es):
    es = ExperimentSetting
    o = np.append(o, [float(env.US.onset == env.time_step)])
    o = np.append(o, [float(env.CS.onset == env.time_step)])
    for j in range(10):
        o = np.append(o, [float(env.distractors[j].onset == env.time_step)])
    return o

def activity_plotter():
    with open('../result/trace_conditioning/agent.pkl', 'rb') as f:
        ag = pickle.load(f)

    with open('../result/trace_conditioning/experiment_settings.pkl', 'rb') as f:
        es = pickle.load(f)


    np.random.seed(es.seed)
    env = TraceConditioning(es.seed, es.ISI, es.ITI, es.gamma, es.num_distractors,
                            {"CS": es.CS_duration,
                             "US": es.US_duration,
                             "distractor": es.distractor_duration})
    most = 15
    activity = np.zeros((most,100))
    cs = []
    us = []
    interest = 1
    num_plot = 17
    t = 0
    # observation
    input_dim = 2 + 2 * (es.num_cs + es.num_distractors)
    num_features = 1 + es.num_cs + es.num_distractors + es.num_features
    o = env.reset().observation
    # add the onset of the signals into the input vector
    # make o = [observation bits,us_onset,css_onset,distractors_onset]
    o = augment_observation(o, env, es)

    ts = env.trial_start
    sorted_features = ag.feature_start_index + np.argsort(np.abs(ag.w[ag.feature_start_index:]), axis=0)
    index_interest = np.transpose(sorted_features[-most:])
    print(index_interest)
    #index_interest = [[59,109,47,37,64,71,87,102,22,82]]
    while env.trial_start == ts:
        t += 1
        step = env.step(1)
        op = step.observation
        op = augment_observation(op, env, es)
        r = step.reward
        td_error, yest = ag.td_learn(o.reshape(input_dim, 1),
                                     op.reshape(input_dim, 1), r, es.gamma, es.gamma)

        if t<30:
            cs.append(o[1])
            us.append(o[0])
            for i in range(most):
                activity[i,t] = ag.xt_m1[index_interest[0][i]]
                # if i == 2:
                #     print(activity[i,t])
        o = op
    print(cs)


    fig, axs = plt.subplots(most+2, sharex=True, sharey=False)
    axs[0].step(np.arange(num_plot), np.array(cs[:num_plot]).flatten(), where='post',color="red")
    axs[1].step(np.arange(num_plot), np.array(us[:num_plot]).flatten(), where='post',color="green")
    for i in range(2,most+2):
        axs[i].step(np.arange(num_plot), np.array(activity[i-2,:num_plot]).flatten(), where='post')


    plt.show()
if __name__ == "__main__":
    activity_plotter()

