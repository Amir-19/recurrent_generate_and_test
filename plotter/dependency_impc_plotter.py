import numpy as np
from environment.classical_conditioning_suite import TraceConditioning, compute_return_error
from agent.trace_conditioning_generate_and_test_random import AgentStateConstruction
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from utils.experiment_settings import ExperimentSetting
# dot -Tpdf -Grankdir=BT -Gdpi=400 tp.dot -o tp.pdf


def dependency_plotter():
    with open('../result/trace_patterning/agent.pkl', 'rb') as f:
        ag = pickle.load(f)

    for i in range(ag.source_trace[77:].shape[0]):
        print(i+77,"->",int(ag.source_trace[i+77][0]),";")
    for k in range(0, 77):
        print(k, np.round(ag.v[-16:-10, k], 2))
    # for i in range(ag.num_based_on.shape[0]):
    #     print(i,ag.source_trace[i])
    print(ag.num_based_on)

if __name__ == "__main__":
    dependency_plotter()