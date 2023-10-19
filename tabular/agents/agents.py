# Author: Marlos C. Machado

from .base import Agent
import numpy as np
from env.base import MDP
from utils.general_utils import discrete_entropy


class Sarsa(Agent):

    def __init__(self, env: MDP, step_size: float, gamma: float, epsilon: float):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()

    def step(self):
        curr_a = self.epsilon_greedy(self.q[self.curr_s], epsilon=self.epsilon)
        r = self.env.act(curr_a)
        next_s = self.env.get_current_state()
        next_a = self.epsilon_greedy(self.q[next_s], epsilon=self.epsilon)

        self.update(self.curr_s, curr_a, r, next_s, next_a)

        self.curr_s = next_s

        self.current_undisc_return += r
        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0
    
    def intrinsic_reward(self, s: int, a: int, next_s: int):
        return 0

    def update(self, s: int, a: int, r: float, next_s: int, next_a: int):
        intrinsic_reward = self.intrinsic_reward(s, a, next_s)
        r_ = r + intrinsic_reward
        self.update_q_values(s, a, r_, next_s, next_a)

    def update_q_values(self, s: int, a: int, r: float, next_s: int, next_a: int):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * (1.0 - \
            self.env.is_terminal()) * self.q[next_s][next_a] - self.q[s][a])
        pass


class Sarsa_SR(Sarsa):

    def __init__(
        self, 
        env: MDP, 
        step_size: float, 
        step_size_sr: float,
        gamma: float, 
        gamma_sr: float, 
        epsilon: float, 
        beta: float, 
        norm_ord: int = 1, 
        full_sr: bool = False, 
    ):
        super().__init__(env, step_size, gamma, epsilon)

        self.beta = beta
        self.alpha_sr = step_size_sr
        self.gamma_sr = gamma_sr
        self.norm_ord = norm_ord
        self.full_sr = full_sr
        
        if full_sr:
            transmat = self.env.transition_matrix
            transmat = transmat.sum(axis=1)
            transmat = transmat / (transmat.sum(axis=-1, keepdims=True) + 1e-20)
            self.sr = np.linalg.inv(np.eye(transmat.shape[0]) - gamma_sr * transmat)
        else:
            self.num_states = self.env.get_num_states()
            self.sr = np.zeros((self.num_states, self.num_states))
            
    def intrinsic_reward(self, s: int, a: int, next_s: int):
        return self.beta * 1. / np.linalg.norm(self.sr[s], ord=self.norm_ord)
    
    def update(self, s: int, a: int, r: float, next_s: int, next_a: int):
        if not self.full_sr:
            self.update_sr(s, next_s)
            intrinsic_reward = self.intrinsic_reward(s, a, next_s)
            r_ = r + intrinsic_reward
            self.update_q_values(s, a, r_, next_s, next_a)

    def update_sr(self, s, next_s):
        one_hot = np.eye(self.num_states)
        self.sr[s, :] = self.sr[s, :] + self.alpha_sr * (one_hot[s] + self.gamma_sr * \
            (1.0 - self.env.is_terminal()) * self.sr[next_s, :] - self.sr[s, :])


class Sarsa_SR_Relative(Sarsa_SR):
    def __init__(
        self, 
        env: MDP, 
        step_size: float, 
        step_size_sr: float,
        gamma: float, 
        gamma_sr: float, 
        epsilon: float, 
        beta: float, 
        norm_ord: int = 1, 
        full_sr: bool = False, 
        intrinsic_type: str = "spie", 
    ):
        super().__init__(env, step_size, step_size_sr, gamma, gamma_sr, 
                         epsilon, beta, norm_ord, full_sr)

        self.intrinsic_type = intrinsic_type
    
    def intrinsic_reward(self, s: int, a: int, next_s: int):
        if self.intrinsic_type == "prospective":
            intrinsic_reward = self.sr[s, next_s]
        elif self.intrinsic_type == "retrospective":
            intrinsic_reward = -np.linalg.norm(self.sr[:, next_s], ord=self.norm_ord)
        elif self.intrinsic_type == "spie":
            intrinsic_reward = self.sr[s, next_s] - \
                np.linalg.norm(self.sr[:, next_s], ord=self.norm_ord)
        else:
            raise ValueError(f"Unsupported intrinsic reward type: \
                {self.intrinsic_type}")
        
        return self.beta * intrinsic_reward


class Sarsa_FR(Sarsa):
    def __init__(
        self, 
        env: MDP, 
        step_size: float, 
        step_size_fr: float,
        gamma: float, 
        gamma_fr: float, 
        epsilon: float, 
        beta: float, 
        norm_ord: int = 1, 
        full_fr: bool = False, 
    ):
        super().__init__(env, step_size, gamma, epsilon)

        self.gamma_fr = gamma_fr
        self.alpha_fr = step_size_fr
        self.beta = beta
        self.norm_ord = norm_ord
        self.full_fr = full_fr
        
        if self.full_fr:
            transmat = env.transition_matrix
            transmat = transmat.sum(axis=1)
            transmat = transmat / (transmat.sum(axis=-1, keepdims=True) + 1e-20)
            self.fr = np.linalg.inv(np.eye(transmat.shape[0]) - self.gamma_fr * \
                (np.ones((transmat.shape[0], transmat.shape[0])) - 
                 np.eye(transmat.shape[0])).dot(transmat))
        else:
            self.num_states = self.env.get_num_states()
            self.fr = np.eye(self.num_states)
    
    def intrinsic_reward(self, s: int, a: int, next_s: int):
        return self.beta * np.linalg.norm(self.fr[next_s, :], ord=self.norm_ord)
    
    def update(self, s: int, a: int, r: float, next_s: int, next_a: int):
        if not self.full_fr:
            self.update_fr(s, next_s)
            intrinsic_reward = self.intrinsic_reward(s, a, next_s)
            r_ = r + intrinsic_reward
            self.update_q_values(s, a, r_, next_s, next_a)
    
    def update_fr(self, s: int, next_s: int):
        delta = self.gamma_fr * self.fr[next_s, :] - self.fr[s, :]
        delta[s] = 0
        self.fr[s, :] = self.fr[s, :] + self.alpha_fr * delta
        

class Sarsa_SR_PR(Sarsa_SR):
    def __init__(
        self, 
        env: MDP, 
        step_size: float, 
        step_size_sr: float,
        step_size_pr: float, 
        gamma: float, 
        gamma_sr: float, 
        gamma_pr, 
        epsilon: float, 
        beta: float, 
        norm_ord: int = 1, 
        full_spr: bool = False, 
    ):
        super().__init__(env, step_size, step_size_sr, gamma, gamma_sr, 
                         epsilon, beta, norm_ord, full_spr)
        self.alpha_pr = step_size_pr
        self.gamma_pr = gamma_pr
        
        if self.full_spr:
            raise NotImplementedError("Coming soon")
        else:
            self.pr = np.zeros((self.num_states, self.num_states))
    
    def intrinsic_reward(self, s: int, a: int, next_s: int):
        return self.beta * (self.sr[s, next_s] - np.linalg.norm(self.pr[:, next_s], \
            ord=self.norm_ord))
        
    def update(self, s: int, a: int, r: float, next_s: int, next_a: int):
        if not self.spr:
            self.update_sr(s, next_s)
            self.update_pr(s, next_s)
            intrinsic_reward = self.intrinsic_reward(s, a, next_s)
            r_ = r + intrinsic_reward
            self.update_q_values(s, a, r_, next_s, next_a)
    
    def update_pr(self, s: int, next_s: int):
        one_hot = np.eye(self.num_states)
        self.pr[:, next_s] = self.pr[:, next_s] + self.alpha_pr * (one_hot[next_s] + \
            self.gamma_pr * (1.0 - self.env.is_terminal()) * self.pr[:, s] - \
                self.pr[:, next_s])
        

class Sarsa_SR_Relative_Entropy(Sarsa_SR_Relative):
    def __init__(
        self, 
        env: MDP, 
        step_size: float, 
        step_size_sr: float,
        gamma: float, 
        gamma_sr: float, 
        epsilon: float, 
        beta: float, 
        norm_ord: int = 1, 
        full_sr: bool = False, 
        intrinsic_type: str = "spie", 
    ):
        super().__init__(env, step_size, step_size_sr, gamma, gamma_sr, 
                         epsilon, beta, norm_ord, full_sr, intrinsic_type)
    
    def intrinsic_reward(self, s: int, a: int, next_s: int):
        if self.intrinsic_type == "spie":
            intrinsic_reward = discrete_entropy(self.sr[s, :]) - \
                np.linalg.norm(self.sr[:, next_s], ord=self.norm_ord)
        elif self.intrinsic_type == "prospective":
            intrinsic_reward = discrete_entropy(self.sr[s, :])
        elif self.intrinsic_type == "retrospective":
            intrinsic_reward = -np.linalg.norm(self.sr[:, next_s], 
                                               ord=self.norm_ord)
        else:
            raise ValueError(f"Unsupported intrinsic reward type: \
                {self.intrinsic_type}")
        
        return self.beta * intrinsic_reward