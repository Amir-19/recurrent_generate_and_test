import numpy as np

class ExperimentSetting:

    def __init__(self, seed, num_trials, trial_max_len, num_saved_time_steps, ISI_len, num_cs,
                 num_distractors, CS_duration, US_duration, distractor_duration, step_size, bias_bit,
                 lambda_td, theta, learning_alg, trace_w_decay, num_features, maturity_threshold,
                 active_maturity_threshold, short_memory_span, num_gen, num_remove, trace_decay_rate,
                 reserved_num_imphe, reserved_num_impc):

        # experiment settings
        # seed = np.random.randint(100)
        self.seed = seed
        self.num_trails = num_trials
        self.trail_max_len = trial_max_len
        self.num_saved_time_steps = num_saved_time_steps

        # environment settings
        self.ISI_len = ISI_len
        self.gamma = 1 - (1 / ISI_len)
        self.ISI = [ISI_len, ISI_len]
        self.ITI = [80, 120]
        self.num_cs = num_cs
        self.num_distractors = num_distractors
        self.CS_duration = CS_duration
        self.US_duration = US_duration
        self.distractor_duration = distractor_duration

        # agent settings
        self.step_size = step_size           # step size for TD(lambda) or initial step-size for TIDBD(lambda)
        self.bias_bit = bias_bit
        self.lambda_td = lambda_td
        self.theta = theta                   # meta paramter step-size for TIDBD(lambda
        self.beta_init = np.log(step_size)   # step-size init for meta gradient TD methods
        self.learning_alg = learning_alg     # 0: TD(lambda), 1: semi-gradient TIDBD(lambda)
        self.trace_w_decay = trace_w_decay   # decay parameter for trace of the weight magnitude
        self.num_features = num_features
        self.maturity_threshold = maturity_threshold  # the age that a feature considered mature
        self.active_maturity_threshold = active_maturity_threshold  # the active age for maturity
        self.short_memory_span = short_memory_span # number of time steps to keep a ramified configuration

        # deep trace settings
        self.num_gen = num_gen               # maximum number of generated features per time step
        self.num_remove = num_remove         # if at cap remove this many mature feature (established?)
        self.trace_decay_rate = trace_decay_rate  # trace decay rate for deep trace if not random

        # imprinting on times of high error settings
        self.reserved_num_imphe = reserved_num_imphe

        # imprinting on configuration
        self.reserved_num_impc = reserved_num_impc