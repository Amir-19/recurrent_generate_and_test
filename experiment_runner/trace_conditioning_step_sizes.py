from experimentor.trace_conditioning_experimentor_random_dtrace import trace_conditioning_exp_dtrace
from utils.experiment_settings import ExperimentSetting
import numpy as np
from tqdm import tqdm


def exp_runner_dtrace(name, ISI, num_features, num_saved, learning_algorithm, step_size, meta_step_size):
    #seed = 42
    seed = 19
    num_runs = 2
    np.random.seed(seed)
    msre_dt = []
    num_tr = 20000
    sum_msre = 0
    for i in tqdm(range(num_runs),leave=False):
        in_sd = np.random.randint(100000)
        try:
            es = ExperimentSetting(seed=in_sd, num_trials=num_tr, trial_max_len=140,
                                   num_saved_time_steps=num_saved,ISI_len=ISI, num_cs=1,
                                   num_distractors=10,CS_duration=4, US_duration=2,
                                   distractor_duration=4, step_size=step_size,bias_bit=1,
                                   lambda_td=0.9, theta=meta_step_size,learning_alg=learning_algorithm,
                                   trace_w_decay=0.99,num_features=num_features, maturity_threshold=0,
                                   active_maturity_threshold=20, short_memory_span=50, num_gen=2,
                                   num_remove=2, trace_decay_rate=0.6, reserved_num_imphe=0,
                                   reserved_num_impc=0)
            msre = trace_conditioning_exp_dtrace(name, es)
            msre_dt.append(msre)
            sum_msre+=msre
        except BaseException as err:
            print("error occurred! skip to next run")

        print("run: ",i+1," msre: ", msre)
    print("average msre: ",sum_msre/num_runs)

    with open('../result/trace_conditioning/'+name+'/list_msre_runs.npy', 'wb') as f:
        np.save(f, msre_dt)

def all_dtrace_runs():

    # TD methods
    # print("running for TD 0.1")
    # exp_runner_dtrace("dt_ss_exp_TD_ss_1", 10, 100, 17, 0, 0.1, 0)
    #
    # print("running for TD 0.01")
    # exp_runner_dtrace("dt_ss_exp_TD_ss_01", 10, 100, 17, 0, 0.01, 0)
    #
    # print("running for TD 0.001")
    # exp_runner_dtrace("dt_ss_exp_TD_ss_001", 10, 100, 17, 0, 0.001, 0)
    #
    # print("running for TD 0.05")
    # exp_runner_dtrace("dt_ss_exp_TD_ss_05", 10, 100, 17, 0, 0.05, 0)
    #
    # print("running for TD 0.005")
    # exp_runner_dtrace("dt_ss_exp_TD_ss_005", 10, 100, 17, 0, 0.005, 0)

    # theta = 0.1
    # print("running for TIDBD alpha 0.1 theta 0.1")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_1_ms_1", 10, 100, 17, 1, 0.1, 0.1)
    #
    # print("running for TIDBD alpha 0.01 theta 0.1")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_01_ms_1", 10, 100, 17, 1, 0.01, 0.1)
    #
    # print("running for TIDBD alpha 0.001 theta 0.1")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_001_ms_1", 10, 100, 17, 1, 0.001, 0.1)
    #
    # print("running for TIDBD alpha 0.05 theta 0.1")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_05_ms_1", 10, 100, 17, 1, 0.05, 0.1)
    #
    # print("running for TIDBD alpha 0.005 theta 0.1")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_005_ms_1", 10, 100, 17, 1, 0.005, 0.1)

    # theta = 0.05
    # print("running for TIDBD alpha 0.1 theta 0.05")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_1_ms_05", 10, 100, 17, 1, 0.1, 0.05)
    #
    # print("running for TIDBD alpha 0.01 theta 0.05")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_01_ms_05", 10, 100, 17, 1, 0.01, 0.05)
    #
    # print("running for TIDBD alpha 0.001 theta 0.05")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_001_ms_05", 10, 100, 17, 1, 0.001, 0.05)
    #
    # print("running for TIDBD alpha 0.05 theta 0.05")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_05_ms_05", 10, 100, 17, 1, 0.05, 0.05)
    #
    # print("running for TIDBD alpha 0.005 theta 0.05")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_005_ms_05", 10, 100, 17, 1, 0.005, 0.05)

    # theta = 0.025
    # print("running for TIDBD alpha 0.1 theta 0.025")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_1_ms_025", 10, 100, 17, 1, 0.1, 0.025)
    #
    # print("running for TIDBD alpha 0.01 theta 0.025")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_01_ms_025", 10, 100, 17, 1, 0.01, 0.025)
    #
    # print("running for TIDBD alpha 0.001 theta 0.025")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_001_ms_025", 10, 100, 17, 1, 0.001, 0.025)
    #
    print("running for TIDBD alpha 0.05 theta 0.025")
    exp_runner_dtrace("dt_ss_exp_TIDBD_ss_05_ms_025", 10, 100, 17, 1, 0.05, 0.025)
    #
    # print("running for TIDBD alpha 0.005 theta 0.025")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_005_ms_025", 10, 100, 17, 1, 0.005, 0.025)

    # theta = 0.01
    # print("running for TIDBD alpha 0.1 theta 0.01")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_1_ms_01", 10, 100, 17, 1, 0.1, 0.01)
    #
    # print("running for TIDBD alpha 0.01 theta 0.01")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_01_ms_01", 10, 100, 17, 1, 0.01, 0.01)
    #
    # print("running for TIDBD alpha 0.001 theta 0.01")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_001_ms_01", 10, 100, 17, 1, 0.001, 0.01)
    #
    # print("running for TIDBD alpha 0.05 theta 0.01")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_05_ms_01", 10, 100, 17, 1, 0.05, 0.01)
    #
    # print("running for TIDBD alpha 0.005 theta 0.01")
    # exp_runner_dtrace("dt_ss_exp_TIDBD_ss_005_ms_01", 10, 100, 17, 1, 0.005, 0.01)

if __name__ == "__main__":
    all_dtrace_runs()