import numpy as np
import pickle
import time
import os

from utils.general_utils import parse_args
from experiment_main import experiment_main

"""
Potentially good hyperparameters: 
(0.05, 0.05, 0.95, 10, 0.01)
(0.05, 0.1, 0.8, 10, 0.01)
"""


def hyperparameter_sweep_srr_entropy(args):
    assert args.agent == "Sarsa_SR_Relative_Entropy"
    
    step_size_list = np.array([0.01, 0.05, 0.1, 0.25, 0.5])
    step_size_sr_list = np.array([0.01, 0.05, 0.1, 0.25, 0.5])
    gamma_sr_list = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
    beta_list = np.array([1, 10, 100, 1000, 10000])
    epsilon_list = np.array([0.01, 0.05, 0.1])

    returns_mean_table = np.zeros((len(step_size_list), len(step_size_sr_list), len(gamma_sr_list), len(beta_list), 
                                   len(epsilon_list)))
    returns_std_table = np.zeros((len(step_size_list), len(step_size_sr_list), len(gamma_sr_list), len(beta_list), 
                                  len(epsilon_list)))
    
    t0 = time.time()
    for i in range(len(step_size_list)):
        for j in range(len(step_size_sr_list)):
            for k in range(len(gamma_sr_list)):
                for l in range(len(beta_list)):
                    for m in range(len(epsilon_list)):
                        args.num_episodes = 100
                        args.step_size = step_size_list[i]
                        args.step_size_sr = step_size_sr_list[j]
                        args.gamma_sr = gamma_sr_list[k]
                        args.beta = beta_list[l]
                        args.epsilon = epsilon_list[m]
                        print(f'({args.step_size}, {args.step_size_sr}, \
                            {args.gamma_sr}, {args.beta}, {args.epsilon}):')
                        episode_return_list = experiment_main(args)
                        mean_return = np.mean(episode_return_list)
                        std_return = np.std(episode_return_list)
                        returns_mean_table[i, j, k, l, m] = mean_return
                        returns_std_table[i, j, k, l, m] = std_return
                        print(f'return mean: {mean_return:.2f} | return std: \
                            {std_return:.2f} | time: {(time.time() - t0):.2f}')
                        t0 = time.time()
    
    if not os.path.exists(f"logs/hyperparam_sweep/{args.agent}/"):
        os.makedirs(f"logs/hyperparam_sweep/{args.agent}/")
    with open(f"logs/hyperparam_sweep/{args.agent}/{args.env}.pkl", 'wb') as f:
        pickle.dump((returns_mean_table, returns_std_table), f)
    f.close()
    

if __name__=="__main__":
    args = parse_args()
    hyperparameter_sweep_srr_entropy(args)