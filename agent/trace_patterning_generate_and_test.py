import numpy as np
import copy


class AgentStateConstruction:

    def __init__(self, m, n, step_size=0.1, bias_bit=1, lambda_td=0.9, theta=0.01,
                 beta=np.log(0.005), learning_algorithm=1, trace_w_decay=0.9999,
                 maturity_threshold=1000, active_maturity_threshold=1000, short_memory_span=50):

        self.m = m
        self.n = n
        self.learning_algorithm = learning_algorithm  # 0 = TD, 1 = TIDBD
        self.maturity_threshold = maturity_threshold
        self.active_maturity_threshold = active_maturity_threshold
        self.short_memory_span = short_memory_span

        # network params
        self.v = np.zeros((m + n, n))
        self.w = np.zeros((n + 1, 1))
        self.f = np.zeros((n + 1, 1))
        self.lower_bound_thr = np.ones((1, n)) * np.NINF
        self.upper_bound_thr = np.ones((1, n)) * np.Inf
        # 0 does NOT participate in learning, 1 does
        self.feature_output_type = np.zeros((1, n))
        self.bias_bit = bias_bit

        # td learning params
        self.step_size = step_size
        self.z = np.zeros((n + 1, 1))
        self.lambda_td = lambda_td
        self.xt_m1 = np.zeros((n, 1))
        self.xt_m2 = np.zeros((n, 1))
        self.x_est = np.zeros((n, 1))

        # additional TIDBD params
        self.H = np.zeros((n + 1, 1))
        self.beta = np.ones((n + 1, 1)) * beta
        self.alpha = np.exp(self.beta)
        self.theta = theta
        self.beta_init = beta

        # feature stats
        # todo: check to get rid of extra stuff!
        self.age = np.zeros((n, 1))
        self.num_reference = np.zeros((n, 1))
        self.expected_activation = np.zeros((n, 1))
        self.active_age = np.zeros((n, 1))
        self.trace_w_mag = np.zeros((n, 1))
        self.trace_w_decay = trace_w_decay
        self.w_mean = np.zeros((n, 1))
        self.w_m2 = np.zeros((n, 1))
        self.w_variance = np.zeros((n, 1))
        # 0:empty 1:imp_c 2:deep trace
        self.feature_type = np.zeros((n, 1))
        self.feature_empty = np.ones((n, 1))  # 0: used, 1: empty
        # to get used by the direct presence connections
        self.feature_start_index = 0
        self.active_feature_error = np.zeros((n, 1))
        self.active_feature_sq_error = np.zeros((n, 1))
        self.s_score = np.zeros((n, 1))
        self.num_real_features = n
        self.is_traced = np.zeros((n, 1))  # 0: no, 1: yes
        self.trace_features_decay = 0.6
        self.correlation_matrix = np.zeros((n, n))
        self.covariance_matrix = np.zeros((n, n))
        self.preserved_age = np.zeros((n, 1))

        # input stats
        self.informative = np.zeros((m, 1))
        self.backbone_inputs = []

        # network stats
        self.mean_td_error = 0
        self.m2_td_error = 0
        self.variance_td_error = 0
        self.time_step = 0
        self.sd_td_error = 0

        # deep traces stats
        # for 0 elements -> safe to remove
        self.num_based_on = np.zeros((n, 1))
        self.source_trace = np.ones((n, 1)) * -1
        self.original_input = np.ones((n, 1)) * -1
        self.depth_input = np.zeros((n, 1))

        # capacity of different features
        self.feature_cap_type = np.zeros((n, 1))  # 0:dc 1:imp_c 2:deep trace
        self.start_impc_index = 0
        self.end_impc_index = 0
        self.start_dtrace_index = 0
        self.end_dtrace_index = 0
        self.num_dtrace = 0
        self.num_impc = 0

    def set_feature_start_index(self, start_index):

        self.feature_start_index = start_index
        self.num_real_features = self.n - start_index

    def set_feature_cap_types(self, imp_c_start_index, dtrace_start_index):

        # [direct-connections, imprinting_on_conf, d_trace_features]
        self.feature_cap_type[:imp_c_start_index] = 0
        self.feature_cap_type[imp_c_start_index:] = 1
        self.feature_cap_type[dtrace_start_index:] = 2
        self.start_impc_index = imp_c_start_index
        self.start_dtrace_index = dtrace_start_index
        self.end_impc_index = dtrace_start_index - 1
        self.end_dtrace_index = self.n - 1
        self.num_dtrace = self.n - dtrace_start_index
        self.num_impc = dtrace_start_index - imp_c_start_index

    def forward(self, o, x_tm1):

        assert o.shape == (self.m,) or o.shape == (self.m, 1)
        x_t = np.dot(np.concatenate((x_tm1, o), axis=0).T, self.v)
        # apply bounds
        maskl = np.greater_equal(x_t, self.lower_bound_thr).astype(float)
        maskh = np.less(x_t, self.upper_bound_thr).astype(float)
        mask = np.multiply(maskl, maskh)
        np.putmask(x_t, 1 - mask, [0])

        # LTU, LBU activation function
        mask = np.multiply(mask, self.feature_output_type == 1)
        np.putmask(x_t, mask, [1])

        # replacing trace
        mask = np.greater_equal(x_t, 1.0).astype(float)
        mask = np.multiply(mask, np.transpose(self.feature_type) == 5)
        np.putmask(x_t, mask, [1])

        used_in_f = x_t.copy()
        # put zero for the features that are not participating in learning
        # np.putmask(used_in_f, self.feature_output_type == 0, [0])
        return used_in_f, x_t

    def calculate_output(self, o, x_tm1):
        # this method calculate the output and the hidden state and also update the hidden state
        used_in_f, x_t = self.forward(o, x_tm1)
        self.f = np.append(used_in_f, [self.bias_bit])
        y = np.dot(self.f, self.w)[0]
        return y, x_t.transpose()

    def estimate_output(self, o, x_tm1):
        # this method calculate the output and the hidden state without updating the hidden state
        used_in_f, x_t = self.forward(o, x_tm1)
        f = np.append(used_in_f, [self.bias_bit])
        y = np.dot(f, self.w)[0]
        return y, x_t.transpose(), f

    def is_this_new(self, candid_feature):

        return not np.equal(candid_feature, self.v.transpose()).all(1).any()

    def is_this_new_imprinting_feature(self, j, lower, upper):
        arg = np.argwhere(self.v[j] == 1)
        if len(arg) == 0:
            return True
        condl = np.isclose(self.lower_bound_thr[0, arg], lower)
        condu = np.isclose(self.upper_bound_thr[0, arg], upper)
        return not np.bitwise_and(condl, condu).any(0)[0]

    def make_i_trace_of_j(self, i, j, lam, feature_type):

        self.reinit_feature_i(i)
        self.feature_output_type[0, i] = 0.0
        self.lower_bound_thr[0, i] = np.NINF
        self.upper_bound_thr[0, i] = np.Inf
        self.v[i, i] = lam
        self.v[j, i] = 1
        self.feature_empty[i] = 0.0
        self.feature_type[i] = feature_type

    def make_i_imprint_on_j(self, i, j, lower, upper, feature_type):

        self.reinit_feature_i(i)
        self.feature_output_type[0, i] = 1.0
        self.lower_bound_thr[0, i] = lower
        self.upper_bound_thr[0, i] = upper
        self.v[j, i] = 1.0
        self.feature_empty[i] = 0.0
        self.feature_type[i] = feature_type

    def make_i_deep_trace_of_j(self, i, j, lam, feature_type, output_type):

        self.reinit_feature_i(i)
        self.feature_output_type[0, i] = output_type
        self.lower_bound_thr[0, i] = np.NINF
        self.upper_bound_thr[0, i] = np.Inf
        self.v[i, i] = lam
        self.v[j, i] = 1 - lam
        self.feature_empty[i] = 0.0
        self.feature_type[i] = feature_type

    def make_i_presence_of_j(self, i, j, feature_type=4):

        self.reinit_feature_i(i)
        self.v[:, i] = np.zeros(self.m + self.n)
        self.feature_output_type[0, i] = 1.0
        self.lower_bound_thr[0, i] = 1.0
        self.upper_bound_thr[0, i] = 1.1
        self.v[j, i] = 1.0
        self.feature_empty[i] = 0.0
        self.feature_type[i] = feature_type

    def is_this_time_step_represented(self, x_t, td_error):
        # if there is a feature activated and its not very old.
        if np.abs(td_error) < self.mean_td_error + 0 * self.sd_td_error:
            return True
        activity = self.active_age.copy()
        activity[x_t == 0] = np.Inf
        activity[self.feature_output_type.T == 0] = np.Inf
        if np.min(activity) < self.active_maturity_threshold:
            return True

        return False

    def is_observation_configuration_represented(self, o):
        candid_feature = np.zeros(self.m + self.n)
        # todo: incorrect sum with n why?
        backbone_indc = (self.n + np.array(self.backbone_inputs)).tolist()
        # todo: wrong indices in o??
        candid_feature[backbone_indc] = o[self.backbone_inputs] * 2 - 1
        return self.is_this_new(candid_feature), candid_feature

    def is_observation_important(self, o):
        # todo: maybe even all zeros are important?? bias here?.
        if np.sum(o[self.backbone_inputs]) > 0:
            return True
        else:
            return False

    def make_imprinting_feature_on_configuration(self, o, fan_in, based_on_dir_con=True):
        # backbone_indc_in_o = (np.array(self.backbone_inputs)).tolist()
        backbone_indc_in_x = (self.n + np.array(self.backbone_inputs)).tolist()
        # fan-in of the imprinting features
        fan_in = fan_in
        # select the inputs to connect to
        candid_feature = np.zeros(self.m + self.n)
        # calculate the probability of choosing based on the direct connections
        if based_on_dir_con:
            prob_sum = np.sum(
                np.abs(self.w[1:self.feature_start_index]), axis=0)
            if not np.isclose(prob_sum, 0):
                prob = (np.abs(self.w[1:self.feature_start_index]) / prob_sum)
                prob = np.squeeze(prob, 1)
                imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=fan_in,
                                                          replace=False, p=prob)
            else:
                imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=fan_in,
                                                          replace=False, p=None)
        else:
            imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=fan_in,
                                                      replace=False, p=None)
        imprinting_inputs_in_o = imprinting_inputs_in_x - self.n
        candid_feature[imprinting_inputs_in_x] = o[imprinting_inputs_in_o] * 2 - 1
        return candid_feature, np.sum(o[imprinting_inputs_in_o]), np.inf

    def make_imprinting_feature_on_configuration_unknown(self, o):
        # backbone_indc_in_o = (np.array(self.backbone_inputs)).tolist()
        backbone_indc_in_x = (self.n + np.array(self.backbone_inputs)).tolist()
        # fan-in of the imprinting features
        # select the inputs to connect to
        candid_feature = np.zeros(self.m + self.n)
        # calculate the probability of choosing based on the direct connections
        prob_sum = np.sum(
            np.abs(self.trace_w_mag[1:self.feature_start_index]), axis=0)
        if not np.isclose(prob_sum, 0):
            prob = (
                    np.abs(self.trace_w_mag[1:self.feature_start_index]) / prob_sum)
            prob = np.squeeze(prob, 1)
            # prob_mask = np.random.uniform(low=0, high=1, size=prob.shape)
            baseline_prob = 1 / prob.shape[0]
            prob_mask = np.ones(prob.shape) * baseline_prob + \
                        np.random.normal(0, 1 * baseline_prob, prob.shape)

            selection = np.greater(prob, prob_mask).astype('int')
            selection_indc = np.array(backbone_indc_in_x)
            np.putmask(selection_indc, 1 - selection, [-1])
            selection_indc = selection_indc[selection_indc != -1]
            if len(selection_indc) == 0:
                # TODO: make a random feature
                imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=7,
                                                          replace=False, p=None)
            else:
                imprinting_inputs_in_x = selection_indc

        else:
            # TODO: make a random feature
            imprinting_inputs_in_x = np.random.choice(backbone_indc_in_x, size=7,
                                                      replace=False, p=None)
        imprinting_inputs_in_o = imprinting_inputs_in_x - self.n
        candid_feature[imprinting_inputs_in_x] = o[imprinting_inputs_in_o] * 2 - 1
        return candid_feature, np.sum(o[imprinting_inputs_in_o]), np.inf

    def find_empty_feature(self):
        nonzero = np.nonzero(self.feature_empty)

        if len(np.transpose(nonzero)) == 0:
            return None

        arg = np.transpose(nonzero)[0]

        if len(arg) > 0:
            return arg[0]
        else:
            return None

    def find_empty_impc_feature(self):
        empty_dtrace = self.feature_empty * (self.feature_cap_type == 1)
        nonzero = np.nonzero(empty_dtrace)

        if len(np.transpose(nonzero)) == 0:
            return None

        arg = np.transpose(nonzero)[0]

        if len(arg) > 0:
            return arg[0]
        else:
            return None

    def find_empty_dtrace_feature(self):
        empty_dtrace = self.feature_empty * (self.feature_cap_type == 2)
        nonzero = np.nonzero(empty_dtrace)

        if len(np.transpose(nonzero)) == 0:
            return None

        arg = np.transpose(nonzero)[0]

        if len(arg) > 0:
            return arg[0]
        else:
            return None

    def find_nonzero_trace(self, x):
        traces = copy.deepcopy(x)
        traces[self.feature_type != 5] = 0
        trace_id = np.argwhere(np.invert(np.isclose(traces, 0)))
        if len(trace_id) == 0:
            return None
        else:
            return trace_id[0][0]

    def add_imprinting_feature_i(self, i, candid_feature, lower, upper, feature_type):

        self.reinit_feature_i(i)
        self.v[:, i] = candid_feature
        self.feature_output_type[0, i] = 1.0
        self.lower_bound_thr[0, i] = lower
        self.upper_bound_thr[0, i] = upper
        self.feature_type[i] = feature_type
        self.feature_empty[i] = 0.0

    def reinit_feature_i(self, i):

        # feature related
        self.feature_empty[i] = 1.0
        self.v[:, i] = np.zeros(self.m + self.n)
        self.active_age[i] = 0
        self.age[i] = 0
        self.lower_bound_thr[0, i] = np.NINF
        self.upper_bound_thr[0, i] = np.Inf
        self.feature_type[i] = 0
        self.feature_output_type[0, i] = 0

        # stats related
        self.expected_activation[i] = 0.0
        self.trace_w_mag[i] = np.median(self.trace_w_mag)

        # learning related
        self.beta[i] = self.beta_init
        self.alpha[i] = np.exp(self.beta_init)
        self.H[i] = 0.0
        self.z[i] = 0.0
        self.w[i] = 0.0

        # deep trace related
        self.num_based_on[i] = 0
        self.source_trace[i] = -1
        self.original_input[i] = -1
        self.depth_input[i] = 0

    def make_deep_trace_feature(self, feature_index, trace_decay_rate):
        # makes deep trace feature based on the
        candid_feature = np.zeros(self.m + self.n)
        prob_sum = np.sum(np.abs(self.w[:-1]), axis=0)
        if not np.isclose(prob_sum, 0):
            prob = (np.abs(self.w[:-1]) / prob_sum)
            prob = np.squeeze(prob, 1)
            index_to_trace = np.random.choice(prob.shape[0], p=prob)
            original_index = index_to_trace
            original_input = -1
            depth_input = 0
            if index_to_trace < self.feature_start_index:
                index_to_trace = index_to_trace + self.n + (self.m // 2)
                original_input = original_index
                depth_input = 1
            else:
                original_input = self.original_input[index_to_trace]
                depth_input = self.depth_input[index_to_trace] + 1
            decay_rate = np.random.uniform()
            candid_feature[feature_index] = decay_rate
            candid_feature[index_to_trace] = 1 - decay_rate
            if self.is_this_new(candid_feature):
                self.make_i_deep_trace_of_j(feature_index, index_to_trace, decay_rate,
                                            feature_type=2, output_type=2)
                self.num_based_on[original_index] += 1
                self.source_trace[feature_index] = original_index
                self.original_input[feature_index] = original_input
                self.depth_input[feature_index] = depth_input
            return True
        return False

    def test_features(self, num_remove):

        sorted_impc = self.start_impc_index + \
            np.argpartition(
                np.abs(self.trace_w_mag[self.start_impc_index:self.start_dtrace_index]),
                self.trace_w_mag[self.start_impc_index:self.start_dtrace_index].shape[0]//2, axis=0)

        sorted_dtrace = self.start_dtrace_index + \
            np.argpartition(
                np.abs(self.trace_w_mag[self.start_dtrace_index:]),
                self.trace_w_mag[self.start_dtrace_index:].shape[0]//2, axis=0)

        todelete_impc = sorted_impc[:self.num_impc - self.num_impc//2]
        todelete_dtrace = sorted_dtrace[:self.num_dtrace - self.num_dtrace//2]

        num_deleted_impc = 0
        num_deleted_dtrace = 0

        # tester for imp_c
        if self.find_empty_impc_feature() is None:
            for i in todelete_impc:
                j = i[0]
                if self.age[j] > 10 * self.maturity_threshold and self.feature_type[j] == 1:
                    if self.num_based_on[j] == 0:
                        if self.source_trace[j] != -1:
                            self.num_based_on[self.source_trace[j].astype(
                                int)] -= 1
                        self.reinit_feature_i(j)
                        num_deleted_impc += 1
                if num_deleted_impc == num_remove:
                    break
        # tester for dtrace
        if self.find_empty_dtrace_feature() is None:
            for i in todelete_dtrace:
                j = i[0]
                if self.age[j] > self.maturity_threshold and self.feature_type[j] == 2:
                    if self.num_based_on[j] == 0:
                        if self.source_trace[j] != -1:
                            self.num_based_on[self.source_trace[j].astype(
                                int)] -= 1
                        self.reinit_feature_i(j)
                        num_deleted_dtrace += 1
                if num_deleted_dtrace == num_remove:
                    break


    def td_learn(self, o, op, r, gamma, gammap):
        # calculate the output
        y_est, x_t = self.calculate_output(o, self.xt_m1)
        yp_est, x_est, fp = self.estimate_output(op, x_t)

        # update stats
        self.age += 1
        self.expected_activation = (
                                           ((self.age - 1) * self.expected_activation) + x_t) / self.age
        self.active_age[x_t != 0] += 1
        self.trace_w_mag = self.trace_w_decay * self.trace_w_mag + \
                           (1 - self.trace_w_decay) * np.abs(self.w[:-1])

        # TD error
        delta = r + gammap * yp_est - y_est

        # TD lambda
        if self.learning_algorithm == 0:
            self.z = ((self.lambda_td * gamma *
                       self.z.transpose()) + self.f.T).transpose()
            update = self.step_size * delta * self.z
            self.w = np.add(self.w, update.reshape((self.n + 1, 1)))

        # semi-gradient TIDBD lambda
        elif self.learning_algorithm == 1:
            activation_t = np.expand_dims(self.f, 1)
            self.beta = self.beta + self.theta * delta * self.z * self.H
            self.alpha = np.exp(self.beta)
            self.z = ((self.lambda_td * gamma *
                       self.z.transpose()) + self.f.T).transpose()
            update = self.alpha * delta * self.z
            self.w = np.add(self.w, update.reshape((self.n + 1, 1)))
            postivizer = 1 - self.alpha * activation_t * self.z
            postivizer[postivizer <= 0] = 0
            self.H = self.H * postivizer + delta * self.alpha * self.z

        # transition to next step
        self.xt_m2 = self.xt_m1
        self.xt_m1 = x_t
        self.x_est = x_est

        # updating network stats
        self.time_step += 1
        past_mean_td_error = self.mean_td_error
        self.mean_td_error = (
                                     ((self.time_step - 1) * self.mean_td_error) + np.abs(delta)) / self.time_step
        self.m2_td_error += (np.abs(delta) - self.mean_td_error) * \
                            (np.abs(delta) - past_mean_td_error)
        self.variance_td_error = self.m2_td_error / self.time_step
        self.sd_td_error = np.sqrt(self.variance_td_error)

        # --correlation--

        return delta, y_est
