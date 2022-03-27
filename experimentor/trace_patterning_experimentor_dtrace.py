import numpy as np
from environment.classical_conditioning_suite import TracePatterning, compute_return_error
from agent.trace_patterning_generate_and_test import AgentStateConstruction
from tqdm import tqdm
import utils.utils as utils
from datetime import datetime
from pathlib import Path


def augment_observation(o, env, es):
    o = np.append(o, [float(env.US.onset == env.time_step)])
    for j in range(es.num_cs):
        o = np.append(o, [float(env.CSs[j].onset == env.time_step)])
    for j in range(es.num_distractors):
        o = np.append(o, [float(env.distractors[j].onset == env.time_step)])
    return o


def trace_patterning_exp_dtrace_impc(exp_name, es, num_gen_p):

    experiment_log_path = "../result/trace_patterning/" + exp_name + "/dtrace_impc_" + \
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    np.random.seed(es.seed)
    # extras for trace patterning
    num_activation_pattern = 1
    noise_prob = 0
    activation_pattern_prob = 0.5
    num_gen_pattern = num_gen_p
    env = TracePatterning(es.seed, es.ISI, es.ITI, es.gamma, es.num_cs, num_activation_pattern,
                          activation_pattern_prob, es.num_distractors,
                          {"CS": es.CS_duration,
                           "US": es.US_duration,
                           "distractor": es.distractor_duration}, noise_prob)
    # stats
    t3 = 0
    rs = []
    preds = []
    trials_start_t = []
    num_plot = es.num_saved_time_steps

    # observation
    input_dim = 2+2*(es.num_cs+es.num_distractors)
    max_num_features = es.num_features
    num_features = 1+es.num_cs+es.num_distractors+max_num_features
    o = env.reset().observation
    # add the onset of the signals into the input vector
    # make o = [observation bits,us_onset,css_onset,distractors_onset]
    o = augment_observation(o, env, es)

    # agent
    ag = AgentStateConstruction(input_dim, num_features, es.step_size, es.bias_bit, es.lambda_td,
                                es.theta, es.beta_init, es.learning_alg, es.trace_w_decay,
                                es.maturity_threshold, es.active_maturity_threshold, es.short_memory_span)

    feature_ite = 0
    # make the onset inputs backbone inputs to be used by the agent for feature generation
    ag.backbone_inputs = list(range(1+es.num_cs+es.num_distractors+1, input_dim))

    # make presence features for the CSs and distractors
    for i in range(1+es.num_cs+es.num_distractors):
        ag.make_i_presence_of_j(i, num_features+i)
        feature_ite += 1
    ag.set_feature_start_index(feature_ite)
    ag.set_feature_cap_types(feature_ite, feature_ite+es.reserved_num_impc)

    # experiment main loop
    for _ in tqdm(range(es.num_trails), leave=False):
        trials_start_t.append(t3)
        t = 0
        ts = env.trial_start

        while env.trial_start == ts:
            # next step transition
            step = env.step(1)
            op = step.observation
            op = augment_observation(op, env, es)
            r = step.reward
            # forward-thinking generate-and-test
            # imprinting on configuration genrator
            # if ag.is_observation_important(o):
            #     for k in range(num_gen_pattern):
            #         candid_feature, lower_bound, upper_bound = ag.make_imprinting_feature_on_configuration_unknown(o)
            #         feature_id = ag.find_empty_impc_feature()
            #         is_candid_feature_new = ag.is_this_new(candid_feature)
            #         if feature_id is not None and is_candid_feature_new:
            #             ag.add_imprinting_feature_i(feature_id, candid_feature, lower_bound, upper_bound,
            #                                         feature_type=1.0)
            # deep trace generator
            for j in range(es.num_gen):
                feature_index = ag.find_empty_dtrace_feature()
                if feature_index is not None:
                    ag.make_deep_trace_feature(
                        feature_index, es.trace_decay_rate)

            # TD update
            td_error, yest = ag.td_learn(o.reshape(input_dim, 1), op.reshape(input_dim, 1), r, es.gamma,
                                         es.gamma)

            # backward-thinking generate and also do the testing
            ag.test_features(es.num_remove)

            # next step transitions
            t += 1
            t3 += 1
            o = op
            rs.append(r)
            preds.append(yest)

    msre, _, returns = compute_return_error(rs, preds, es.gamma)
    rmse = utils.compute_return_error_over_bins(rs, preds, es.gamma, 1000)

    Path(experiment_log_path).mkdir(parents=True, exist_ok=True)
    utils.save_rmse_data(experiment_log_path, "rmse.npy", rmse)
    utils.save_final_trials(experiment_log_path, "trial_pn.npy", returns, preds, trials_start_t, num_plot)
    # utils.save_w_stepsize_data(experiment_log_path, "w_stepsize.npy", w_log, stepsize_log, wtrace_log, t3)
    utils.save_network(experiment_log_path, "agent.pkl", ag)
    return msre
