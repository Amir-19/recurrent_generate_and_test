import numpy as np
from environment.classical_conditioning_suite import TraceConditioning, compute_return_error
from agent.trace_conditioning_generate_and_test_random import AgentStateConstruction
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from utils.experiment_settings import ExperimentSetting


def dependency_plotter():
    with open('../result/trace_conditioning/agent.pkl', 'rb') as f:
        ag = pickle.load(f)

    for i in range(ag.source_trace[ag.feature_start_index:].shape[0]):
        print(i+ag.feature_start_index,"->",int(ag.source_trace[i+ag.feature_start_index][0]),";")
    #print(ag.num_based_on)

if __name__ == "__main__":
    dependency_plotter()


