import numpy as np
import pickle

def compute_return_error_over_bins(cumulants, predictions, gamma, bin_size):
    num_time_steps = len(cumulants)
    returns = np.zeros(num_time_steps)
    returns[-1] = cumulants[-1]
    for t in range(num_time_steps - 2, -1, -1):
        returns[t] = gamma * returns[t + 1] + cumulants[t]
    return_error = (predictions - returns) ** 2
    msre = []
    bin_sum_error = 0.0
    bin_i = 0

    for t in range(num_time_steps):
        bin_sum_error += return_error[t]
        bin_i += 1.0
        if bin_i >= bin_size or t == num_time_steps - 1:
            msre.append(bin_sum_error / bin_i)
            bin_i = 0.0
            bin_sum_error = 0.0
    return msre


def save_w_stepsize_data(path_to_save, file_name, w_log, stepsize_log, wtrace_log, t3):
    with open(path_to_save + "/" + file_name, 'wb') as f:
        np.save(f, w_log)
        np.save(f, stepsize_log)
        np.save(f, wtrace_log)
        np.save(f, t3)


def save_trial_data(path_to_save, file_name, pred, csus, tderrors, t3):
    with open(path_to_save + "/" + file_name, 'wb') as f:
        np.save(f, pred)
        np.save(f, csus)
        np.save(f, tderrors)
        np.save(f, t3)


def save_network(path_to_save, file_name, ag):
    with open(path_to_save + "/" + file_name, 'wb') as f:
        pickle.dump(ag, f, pickle.HIGHEST_PROTOCOL)


def save_rmse_data(path_to_save, file_name, rmse):
    with open(path_to_save + "/" + file_name, 'wb') as f:
        np.save(f, rmse)


def save_experiment_settings(path_to_save, file_name, es):
    with open(path_to_save + "/" + file_name, 'wb') as f:
        pickle.dump(es, f, pickle.HIGHEST_PROTOCOL)


def save_final_trials(path_to_save, file_name, rets, preds, ts, num_plot):
    ts_neg = 0
    ts_pos = 0
    for i in range(len(ts)-1, 0, -1):
        if np.round(rets[ts[i]], 2) == 0:
            ts_neg = i
            break
    for i in range(len(ts)-1, 0, -1):
        if np.round(rets[ts[i]], 2) > 0:
            ts_pos = i
            break

    pos_trial = []
    neg_trial = []
    ideal_pos = []
    ideal_neg = []
    for i in range(num_plot):
        pos_trial.append(preds[ts[ts_pos]+i])
        neg_trial.append(preds[ts[ts_neg]+i])
        ideal_pos.append(rets[ts[ts_pos]+i])
        ideal_neg.append(rets[ts[ts_neg]+i])
    with open(path_to_save + "/" + file_name, 'wb') as f:
        np.save(f, pos_trial)
        np.save(f, neg_trial)
        np.save(f, ideal_pos)
        np.save(f, ideal_neg)



# depricated functions
    # def find_nonzero_trace(self, x):
    #     traces = copy.deepcopy(x)
    #     traces[self.feature_type != 5] = 0
    #     trace_id = np.argwhere(np.invert(np.isclose(traces, 0)))
    #     if len(trace_id) == 0:
    #         return None
    #     else:
    #         return trace_id[0][0]

    # def add_imprinting_feature_i(self, i, candid_feature, lower, upper, feature_type):

    #     self.reinit_feature_i(i)
    #     self.v[:, i] = candid_feature
    #     self.feature_output_type[0, i] = 1.0
    #     self.lower_bound_thr[0, i] = lower
    #     self.upper_bound_thr[0, i] = upper
    #     self.feature_type[i] = feature_type
    #     self.feature_empty[i] = 0.0

    # def make_imprinting_feature_on_configuration(self, o, fan_in, based_on_dir_con=True):
    #     # backbone_indc_in_o = (np.array(self.backbone_inputs)).tolist()
    #     backbone_indc_in_x = (self.n + np.array(self.backbone_inputs)).tolist()
    #     # fan-in of the imprinting features
    #     fan_in = fan_in
    #     # select the inputs to connect to
    #     candid_feature = np.zeros(self.m + self.n)
    #     # calculate the probability of choosing based on the direct connections
    #     if based_on_dir_con:
    #         prob_sum = np.sum(
    #             np.abs(self.w[1:self.feature_start_index]), axis=0)
    #         if not np.isclose(prob_sum, 0):
    #             prob = (np.abs(self.w[1:self.feature_start_index])/prob_sum)
    #             prob = np.squeeze(prob, 1)
    #             imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=fan_in,
    #                                                       replace=False, p=prob)
    #         else:
    #             imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=fan_in,
    #                                                       replace=False, p=None)
    #     else:
    #         imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=fan_in,
    #                                                   replace=False, p=None)
    #     imprinting_inputs_in_o = imprinting_inputs_in_x - self.n
    #     candid_feature[imprinting_inputs_in_x] = o[imprinting_inputs_in_o] * 2 - 1
    #     return candid_feature, np.sum(o[imprinting_inputs_in_o]), np.inf

    # def is_this_time_step_represented(self, x_t, td_error):
    #     # if there is a feature activated and its not very old.
    #     if np.abs(td_error) < self.mean_td_error+10*self.sd_td_error:
    #         return True
    #     activity = self.active_age.copy()
    #     activity[x_t == 0] = np.Inf
    #     activity[self.feature_output_type.T == 0] = np.Inf
    #     if np.min(activity) < self.active_maturity_threshold:
    #         return True

    #     return False