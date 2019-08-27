# rlbase

Modular Deep RL infrastructure in PyTorch.

Currently implemented algorithms include PPO and PPOC (Option-Critic trained with Proximal Policy Optimization).

To train, first write a config file and save to `configs/ppo` or `configs/ppoc.` A config file specifies parameters in config objects (found in `core/config.py`) for each component of the model and experiment, such as optimization hyperparameters, network architectures, environments, logging behavior, etc. See example config at `configs/ppo/lightbot_cross.py`

The specified configuration and algorithm can then be run by calling:

`python train.py --config [config file name] --algo [ppo or ppoc]`
