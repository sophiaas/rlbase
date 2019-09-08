# rlbase

Modular Deep RL infrastructure in PyTorch.

Currently implemented algorithms include PPO and PPOC (Option-Critic trained with Proximal Policy Optimization).

To train, first write a config file and save to `configs/ppo` or `configs/ppoc.` A config file specifies parameters in config objects (found in `core/config.py`) for each component of the model and experiment, such as optimization hyperparameters, network architectures, environments, logging behavior, etc. See example config at `configs/ppo/lightbot.py`

The specified configuration and algorithm can then be run by calling:

`python train.py --config [config file name] --algo [ppo or ppoc]`

See `train.py` for other options.

To evaluate a pre-trained model, run:

`python evaluate.py --model_dir [logging directory] --episode [episode to load checkpoint from] --n_eval_steps [number of steps to evaluate]`

Data from the evaluated model is saved to `[logging directory]/evaluate/`
