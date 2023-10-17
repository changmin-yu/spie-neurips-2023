# SPIE-neurips2023 

Python code for implementing our SPIE agent under various settings in our paper: [Successor-Predecessor Intrinsic Exploration](https://arxiv.org/abs/2305.15277) (to appear in NeurIPS 2023).

For any question regarding the paper or the code, please contact changmin.yu98[at]gmail.com.

Currently the codebase only contains implementation of SPIE agents (and associated baselines) for tabular MDP tasks (RiverSwim and SixArms). Further implementations on grid worlds and environments with continuous state space will be included soon.

For executing the codes, try running:
```
python tabular/experiment_main.py --agent Sarsa_SR_Relative --step_size 0.1 --step_size_sr 0.25 --gamma 0.95 --gamma_sr 0.95 --beta 10 --epsilon 0.01 --intrinsic_type spie
```

If you find the paper or the code helpful for your research, please consider citing us with the following format:
```
@article{yu2023successor,
  title={Successor-Predecessor Intrinsic Exploration},
  author={Yu, Changmin and Burgess, Neil and Sahani, Maneesh and Gershman, Sam},
  journal={arXiv preprint arXiv:2305.15277},
  year={2023}
}
```