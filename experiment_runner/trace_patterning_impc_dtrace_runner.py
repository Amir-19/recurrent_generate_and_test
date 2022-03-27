from experimentor.trace_patterning_experimentor_impc_dtrace import trace_patterning_exp_dtrace_impc
from utils.experiment_settings import ExperimentSetting
import numpy as np
from tqdm import tqdm


def exp_runner_dtrace_impc(name, ISI, num_features, num_saved,num_impc, num_gen_pattern):
    seed = 42
    num_runs = 1
    np.random.seed(seed)
    msre_dt = []
    num_tr = 20000
    for i in tqdm(range(num_runs), leave=False):
        in_sd = np.random.randint(100000)
        es = ExperimentSetting(seed=in_sd, num_trials=num_tr, trial_max_len=140,
                               num_saved_time_steps=num_saved,ISI_len=ISI, num_cs=6, num_distractors=10,
                               CS_duration=4, US_duration=2, distractor_duration=4, step_size=0.01,
                               bias_bit=1, lambda_td=0.9, theta=0.01, learning_alg=1, trace_w_decay=0.99,
                               num_features=num_features, maturity_threshold=0,
                               active_maturity_threshold=20, short_memory_span=50, num_gen=2,
                               num_remove=2, trace_decay_rate=0.6, reserved_num_imphe=0,
                               reserved_num_impc=num_impc)
        msre = trace_patterning_exp_dtrace_impc(name, es, num_gen_pattern)
        msre_dt.append(msre)
        print(msre)
    with open('../result/trace_patterning/'+name+'/msre_30_runs.npy', 'wb') as f:
        np.save(f, msre_dt)

def all_dtrace_runs():
    print("running for ISI 10")
    exp_runner_dtrace_impc("dt_impc_ISI_15_runs_30",10,260,17,60,2)

if __name__ == "__main__":
    all_dtrace_runs()