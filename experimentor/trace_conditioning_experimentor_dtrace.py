import numpy as np
from environment.classical_conditioning_suite import TraceConditioning, compute_return_error
from agent.trace_conditioning_generate_and_test_weight_magnitude import AgentStateConstruction
from tqdm import tqdm
import pickle
import utils.utils as utils
import os
from datetime import datetime
from pathlib import Path
from utils.experiment_settings import ExperimentSetting

def augment_observation(o, env, es):
    o = np.append(o, [float(env.US.onset == env.time_step)])
    for j in range(es.num_cs):
        o = np.append(o, [float(env.CS.onset == env.time_step)])
    for j in range(es.num_distractors):
        o = np.append(o, [float(env.distractors[j].onset == env.time_step)])
    return o


def trace_conditioning_exp_dtrace(exp_name, es):

    experiment_log_path = "../result/trace_conditioning/"+exp_name+"/dtrace_" + \
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    np.random.seed(es.seed)

    env = TraceConditioning(es.seed, es.ISI, es.ITI, es.gamma, es.num_distractors,
                            {"CS": es.CS_duration,
                             "US": es.US_duration,
                             "distractor": es.distractor_duration})

    # stats
    t = 0
    t2 = 0
    t3 = 0
    pred = []
    rs = []
    tderrors = []
    preds = []
    csus = []
    num_plot = es.num_saved_time_steps
    feature_ite = 0

    # observation and feature stats
    input_dim = 2 + 2 * (es.num_cs + es.num_distractors)
    num_features = 1 + es.num_cs + es.num_distractors + es.num_features

    # log for weights and step-sizes
    num_trials = es.num_trails
    # max_len_trial = es.trail_max_len
    # w_log = np.zeros((num_trials * max_len_trial, num_features + 1))
    # stepsize_log = np.zeros((num_trials * max_len_trial, num_features + 1))
    # wtrace_log = np.zeros((num_trials * max_len_trial, num_features + 1))

    # start of experiment code
    o = env.reset().observation
    # add the onset of the signals into the input vector
    # make o = [observation bits,us_onset,css_onset,distractors_onset]
    o = augment_observation(o, env, es)

    # agent
    ag = AgentStateConstruction(input_dim, num_features, es.step_size, es.bias_bit, es.lambda_td,
                                es.theta, es.beta_init, es.learning_alg, es.trace_w_decay,
                                es.maturity_threshold, es.active_maturity_threshold,
                                es.short_memory_span)

    # make the onset inputs backbone inputs to be used by the agent for feature generation
    ag.backbone_inputs = list(range(1 + es.num_cs + es.num_distractors + 1, input_dim))

    # make presence feature for the US, CSs, and distractors
    for i in range(1 + es.num_cs + es.num_distractors):
        ag.make_i_presence_of_j(i, num_features + i)
        feature_ite += 1

    # set feature capacity and indicies
    ag.set_feature_start_index(feature_ite)
    ag.set_feature_cap_types(feature_ite, feature_ite + es.reserved_num_imphe,
                             feature_ite + es.reserved_num_imphe)

    # experiment main loop
    for i in tqdm(range(num_trials), leave=False):
        t = 0
        ts = env.trial_start
        while env.trial_start == ts:

            # next step transition
            step = env.step(1)
            op = step.observation
            op = augment_observation(op, env, es)
            r = step.reward
            # forward-thinking generate-and-test
            for j in range(es.num_gen):
                feature_index = ag.find_empty_dtrace_feature()
                if feature_index is not None:
                    ag.make_deep_trace_feature(
                        feature_index)

            # TD update
            td_error, yest = ag.td_learn(o.reshape(input_dim, 1),
                                         op.reshape(input_dim, 1), r, es.gamma, es.gamma)

            # backward-thinking generate and also do the testing
            ag.test_features(es.num_remove)

            # gattering stats for results
            # for j in range(num_features + 1):
            #     w_log[t3, j] = ag.w[j]
            #     stepsize_log[t3, j] = ag.alpha[j]
            #     if j != num_features:
            #         wtrace_log[t3, j] = ag.trace_w_mag[j]
            # debug info
            if i == num_trials - 1:
                if t < num_plot:
                    pred.append(yest)
                    csus.append(o[0] + o[1])
                    tderrors.append(td_error)

            # next step transition and stats
            t += 1
            t3 += 1
            o = op
            rs.append(r)
            preds.append(yest)

    msre, _, _ = compute_return_error(rs, preds, es.gamma)
    rmse = utils.compute_return_error_over_bins(rs, preds, es.gamma, 1000)

    Path(experiment_log_path).mkdir(parents=True, exist_ok=True)
    # utils.save_w_stepsize_data(experiment_log_path, "w_stepsize.npy", w_log, stepsize_log, wtrace_log, t3)
    utils.save_trial_data(experiment_log_path, "trial.npy", pred, csus, tderrors, t3)
    utils.save_network(experiment_log_path, "agent.pkl", ag)
    utils.save_rmse_data(experiment_log_path, "rmse.npy", rmse)
    utils.save_experiment_settings(experiment_log_path, "experiment_settings.pkl", es)
    return msre
