from tqdm import trange
import numpy as np

from utils.general_utils import parse_args
from env.base import MDP
from agents import *


def experiment_main(args):
    environment = MDP(f"tabular/env/{args.env}.mdp")
    
    episode_return_list = []
    
    with trange(args.num_episodes, dynamic_ncols=True) as pbar:
        for ep in pbar:
            if args.agent == "Sarsa":
                agent = Sarsa(
                    environment, 
                    step_size=args.step_size, 
                    gamma=args.gamma, 
                    epsilon=args.epsilon
                )
            elif args.agent == 'Sarsa_SR':
                agent = Sarsa_SR(
                    environment, 
                    step_size=args.step_size, 
                    step_size_sr=args.step_size_sr, 
                    gamma=args.gamma, 
                    gamma_sr=args.gamma_sr, 
                    epsilon=args.epsilon, 
                    beta=args.beta, 
                    norm_ord=args.norm_ord, 
                    full_sr=args.full_sfpr, 
                )
            elif args.agent == "Sarsa_FR":
                agent = Sarsa_FR(
                    environment, 
                    step_size=args.step_size, 
                    step_size_sr=args.step_size_fr, 
                    gamma=args.gamma, 
                    gamma_sr=args.gamma_fr, 
                    epsilon=args.epsilon, 
                    beta=args.beta, 
                    norm_ord=args.norm_ord, 
                    full_sr=args.full_sfpr, 
                )
            elif args.agent == "Sarsa_SR_Relative":
                agent = Sarsa_SR_Relative(
                    environment, 
                    step_size=args.step_size, 
                    step_size_sr=args.step_size_sr, 
                    gamma=args.gamma, 
                    gamma_sr=args.gamma_sr, 
                    epsilon=args.epsilon, 
                    beta=args.beta, 
                    norm_ord=args.norm_ord, 
                    full_sr=args.full_sfpr, 
                    intrinsic_type=args.intrinsic_type,
                )
            elif args.agent == "Sarsa_SR_PR":
                agent = Sarsa_SR_PR(
                    environment, 
                    step_size=args.step_size, 
                    step_size_sr=args.step_size_sr, 
                    step_size_pr=args.step_size_pr, 
                    gamma=args.gamma, 
                    gamma_sr=args.gamma_sr, 
                    gamma_pr=args.gamma_pr, 
                    epsilon=args.epsilon, 
                    beta=args.beta, 
                    norm_ord=args.norm_ord, 
                    full_sr=args.full_sfpr, 
                )
            elif args.agent == "Sarsa_SR_Relative_Entropy":
                agent = Sarsa_SR_Relative_Entropy(
                    environment, 
                    step_size=args.step_size, 
                    step_size_sr=args.step_size_sr, 
                    gamma=args.gamma, 
                    gamma_sr=args.gamma_sr, 
                    epsilon=args.epsilon, 
                    beta=args.beta, 
                    norm_ord=args.norm_ord, 
                    full_sr=args.full_sfpr, 
                )
            else:
                raise NotImplementedError(f"{args.agent} agent not implemented")

            time_step = 1
            while not environment.is_terminal():
                agent.step()
                time_step += 1
            environment.reset()
            
            episode_return = agent.get_avg_undisc_return()
            
            pbar.set_postfix({"ep": ep, "return": episode_return})
            
            episode_return_list.append(episode_return)
    
    episode_return_list = np.array(episode_return_list)
    
    if args.verbose:
        print(f"{agent} | return mean: {np.mean(episode_return_list):.3f} |\
            return std: {np.std(episode_return_list):.3f}")

    return episode_return_list


if __name__=="__main__":
    args = parse_args()
    episode_return_list = experiment_main(args)