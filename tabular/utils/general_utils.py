import numpy as np
import argparse


def argmax_random_break_tie(x):
    return np.random.choice(np.where(x == x.max())[0])


def discrete_entropy(p):
    if p.sum() != 1:
        p = p / (p.sum() + 1e-20)
    return -np.sum(p * np.log2(p + 1e-20))


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", default=1, type=int, 
                        help="random seed")
    parser.add_argument("--env", type=str, default='riverswim', 
                        help='environment')
    parser.add_argument("--num_episodes", default=1000, type=int, 
                        help="number of episodes")
    # agent params
    parser.add_argument('--agent', default="Sarsa", type=str, 
                        help="agent")
    parser.add_argument('--step_size', type=float, default=0.1, 
                        help="learning rate for q-learning")
    parser.add_argument('--step_size_sr', type=float, default=0.1, 
                        help="learning rate for TD-learning of SR")
    parser.add_argument('--step_size_fr', type=float, default=0.1, 
                        help="learning rate for TD-learning of FR")
    parser.add_argument('--step_size_pr', type=float, default=0.1, 
                        help="learning rate for TD-learning of PR")
    parser.add_argument("--beta", default=1.0, type=float, 
                        help="scalar multiplicative factor for intrinsic reward")
    parser.add_argument("--gamma", default=0.95, type=float, 
                        help="discounting factor for q-learning")
    parser.add_argument("--gamma_sr", default=0.95, type=float, 
                        help="discounting factor for TD-learning of SR")
    parser.add_argument("--gamma_fr", default=0.95, type=float, 
                        help="discounting factor for TD-learning of FR")
    parser.add_argument("--gamma_pr", default=0.95, type=float, 
                        help="discounting factor for TD-learning of PR")
    parser.add_argument("--epsilon", default=0.05, type=float, 
                        help="epsilon-greedy")
    parser.add_argument("--norm_ord", default=1, type=int, 
                        help="norm order")
    parser.add_argument("--full_sfpr", default=False, type=bool, 
                        help='using fixed sr/fr/pr')
    parser.add_argument("--intrinsic_type", default="spie", type=str, 
                        help="intrinsic reward type")
    
    parser.add_argument("--verbose", default=False, type=bool, 
                        help='verbose')
    
    args = parser.parse_args()
    
    return args