from experimentor.trace_conditioning_experimentor_random_dtrace import trace_conditioning_exp_dtrace
from utils.experiment_settings import ExperimentSetting
import numpy as np
from tqdm import tqdm


def exp_runner_dtrace(name, ISI, num_features, num_saved):
    seed = 42
    num_runs = 30
    np.random.seed(seed)
    msre_dt = []
    num_tr = 20000
    for i in tqdm(range(num_runs),leave=False):
        in_sd = np.random.randint(100000)
        es = ExperimentSetting(seed=in_sd, num_trials=num_tr, trial_max_len=140,
                               num_saved_time_steps=num_saved,ISI_len=ISI, num_cs=1, num_distractors=10,
                               CS_duration=4, US_duration=2, distractor_duration=4, step_size=0.01,
                               bias_bit=1, lambda_td=0.9, theta=0.01, learning_alg=1, trace_w_decay=0.99,
                               num_features=num_features, maturity_threshold=0,
                               active_maturity_threshold=20, short_memory_span=50, num_gen=2,
                               num_remove=2, trace_decay_rate=0.6, reserved_num_imphe=0,
                               reserved_num_impc=0)
        msre = trace_conditioning_exp_dtrace(name, es)
        msre_dt.append(msre)
    print(msre)
    with open('../result/trace_conditioning/'+name+'/imphe_vs_dtrace_30_runs_ISI_10.npy', 'wb') as f:
        np.save(f, msre_dt)

def all_dtrace_runs():
    print("running for ISI 10")
    exp_runner_dtrace("random_dt_ISI_10_runs_30", 10, 100, 17)
    print("running for ISI 20")
    exp_runner_dtrace("random_dt_ISI_20_runs_30", 20, 200, 27)
    print("running for ISI 30")
    exp_runner_dtrace("random_dt_ISI_30_runs_30", 30, 300, 37)

if __name__ == "__main__":
    all_dtrace_runs()