import numpy as np
from environment.classical_conditioning_suite import TracePatterning, compute_return_error
from agent.trace_conditioning_generate_and_test_random import AgentStateConstruction
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


class ExperimentSetting:

    # experiment settings
    # seed = np.random.randint(100)
    seed = 2
    num_trails = 40000
    trail_max_len = 140

    # environment settings
    ISI_len = 10
    gamma = 1 - (1 / ISI_len)
    ISI = [ISI_len, ISI_len]
    ITI = [80, 120]
    num_cs = 6
    num_distractors = 10
    num_activation_patterns = 1
    activation_pattern_prob = 0.5
    CS_duration = 4
    US_duration = 2
    distractor_duration = 4
    noise_prob = 0.0

    # agent settings
    step_size = 0.01  # step size for TD(lambda)
    bias_bit = 1
    lambda_td = 0.9
    theta = 0.02
    # step size init for meta gradient TD methods
    beta_init = np.log(step_size)
    learning_alg = 1  # 0: TD(lambda), 1: semi-gradient TIDBD(lambda)
    trace_w_decay = 0.99
    num_features = 260
    num_gen_pattern = 2
    num_remove_pattern = 2
    maturity_threshold = 5000
    active_maturity_threshold = 20
    short_memory_span = 50

    # deep trace settings
    num_gen = 2                     # maximum number of generated features per time step
    num_remove = 2                  # if at cap remove this many mature feature (established?)
    trace_decay_rate = 0.6          # trace decay rate for deep trace features


def augment_observation(o, env):
    es = ExperimentSetting
    o = np.append(o, [float(env.US.onset == env.time_step)])
    for j in range(es.num_cs):
        o = np.append(o, [float(env.CSs[j].onset == env.time_step)])
    for j in range(es.num_distractors):
        o = np.append(o, [float(env.distractors[j].onset == env.time_step)])
    return o

def activity_plotter():
    with open('../result/trace_patterning/agent.pkl', 'rb') as f:
        ag = pickle.load(f)
    es = ExperimentSetting
    np.random.seed(6)
    # 1 for off
    # 5 on
    env = TracePatterning(1, es.ISI, es.ITI, es.gamma, es.num_cs, es.num_activation_patterns,
                          es.activation_pattern_prob, es.num_distractors,
                          {"CS": es.CS_duration,
                           "US": es.US_duration,
                           "distractor": es.distractor_duration}, es.noise_prob)
    most = 10
    activity = np.zeros((most,100))
    cs = []
    us = []
    interest = 1
    num_plot = 17
    t = 0
    # observation

    input_dim = 2 + 2 * (es.num_cs + es.num_distractors)
    max_num_features = es.num_features
    num_features = 1 + es.num_cs + es.num_distractors + max_num_features
    o = env.reset().observation
    # add the onset of the signals into the input vector
    # make o = [observation bits,us_onset,css_onset,distractors_onset]
    o = augment_observation(o, env)



    ts = env.trial_start
    sorted_features = ag.feature_start_index + np.argsort(np.abs(ag.w[ag.feature_start_index:]), axis=0)
    index_interest = np.transpose(sorted_features[-most:])
    print(index_interest)
    # seed 1 vs 5 last 190
    index_interest = [[178,259,110,125,131,135,171,177,187,188]]
    while env.trial_start == ts:
        t += 1
        step = env.step(1)
        op = step.observation
        op = augment_observation(op, env)
        r = step.reward
        td_error, yest = ag.td_learn(o.reshape(input_dim, 1),
                                     op.reshape(input_dim, 1), r, es.gamma, es.gamma)

        if t<30:
            us.append(o[0])
            for i in range(most):
                activity[i,t] = np.round(ag.xt_m1[index_interest[0][i]],3)
                # if i == 2:
                #     print(activity[i,t])
        o = op
    print(cs)


    fig, axs = plt.subplots(most+1, sharex=True, sharey=False,figsize=(7, 9))
    axs[0].step(np.arange(num_plot), np.array(us[:num_plot]).flatten(), where='post',color="red")
    axs[0].axes.set_ylim(-0.05, 1.5)
    for i in range(1,most+1):
        axs[i].step(np.arange(num_plot), np.array(activity[i-1,:num_plot]).flatten(), where='post')
        if i != most:
            axs[i].axes.set_ylim(-0.05, 1)

    plt.show()
if __name__ == "__main__":
    activity_plotter()

