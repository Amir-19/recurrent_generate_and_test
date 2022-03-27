import pickle
from pprint import pprint

def print_exp_setting():
    with open('./result/trace_conditioning/dtrace_imphe_2021-08-10_16-29-54/experiment_settings.pkl', 'rb') as f:
        es = pickle.load(f)
    print("---------------------------")
    print("seed: ", es.seed)
    print("number of trials: ", es.num_trails)
    print("trail max length: ",es.trail_max_len)
    #print("num_saved_time_steps: ",es.num_saved_time_steps)
    print("---------------------------")
    print("ISI: ",es.ISI_len)
    print("number of CSs: ",es.num_cs)
    print("number of distractors: ",es.num_distractors)
    print("CS duration: ",es.CS_duration)
    print("US duration: ",es.US_duration)
    print("distractor duration: ",es.distractor_duration)
    print("---------------------------")
    print("step size: ",es.step_size)
    print("bias bit: ",es.bias_bit)
    print("lambda TD: ",es.lambda_td)
    print("theta meta step size: ",es.theta)
    print("learning algorithm: ",es.learning_alg)
    print("trace_w_decay: ",es.trace_w_decay)
    print("number of features(max): ",es.num_features)
    print("maturity threshold: ",es.maturity_threshold)
    print("active maturity threshold: ",es.active_maturity_threshold)
    print("short-term memory span: ",es.short_memory_span)
    print("---------------------------")
    print("number of feature generation: ",es.num_gen)
    print("number of feature removal: ",es.num_remove)
    print("---------------------------")
    print("dtrace decay rate: ",es.trace_decay_rate)
    print("---------------------------")
    print("reserved features for imp high error: ",es.reserved_num_imphe)

if __name__ == "__main__":
    print_exp_setting()
