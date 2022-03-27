from experimentor.trace_conditioning_experimentor_protection import trace_conditioning_exp_dtrace
from utils.experiment_settings import ExperimentSetting
import numpy as np
from tqdm import tqdm


def exp_runner_dtrace(name, ISI, num_features, num_saved, protected):
    # seed = 42
    seed = 24
    num_runs = 20
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
                                   distractor_duration=4, step_size=0.01,bias_bit=1,
                                   lambda_td=0.9, theta=0.01,learning_alg=1,
                                   trace_w_decay=0.99,num_features=num_features, maturity_threshold=0,
                                   active_maturity_threshold=20, short_memory_span=50, num_gen=2,
                                   num_remove=2, trace_decay_rate=0.6, reserved_num_imphe=0,
                                   reserved_num_impc=0)
            msre = trace_conditioning_exp_dtrace(name, es, protected)
            msre_dt.append(msre)
            sum_msre+=msre
        except BaseException as err:
            msre= 0
            print("error occurred! skip to next run")

        print("run: ",i+1," msre: ", msre)
    print("average msre: ",sum_msre/num_runs)

    with open('../result/trace_conditioning/'+name+'/list_msre_runs.npy', 'wb') as f:
        np.save(f, msre_dt)

def all_dtrace_runs():

    # TD methods
    print("running for protection 0%")
    exp_runner_dtrace("protection_exp0", 10, 100, 17, 0)

    print("running for protection 25%")
    exp_runner_dtrace("protection_exp25", 10, 100, 17, 25)

    print("running for protection 50%")
    exp_runner_dtrace("protection_exp50", 10, 100, 17, 75)

    # print("running for protection 75%")
    # exp_runner_dtrace("protection_exp75", 10, 100, 17, 75)
    #
    # print("running for protection 90%")
    # exp_runner_dtrace("protection_exp90", 10, 100, 17, 90)
    #
    # print("running for protection 10%")
    # exp_runner_dtrace("protection_exp10", 10, 100, 17, 10)


if __name__ == "__main__":
    all_dtrace_runs()