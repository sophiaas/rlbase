## Debugging:
~~debug PPO~~ 
redo PPOC with lessons learned from PPO
running rewards
Is the option-state match fix working?

## Refactoring:
~~1. Configs and train files~~
~~2. Add ability to call train files with arguments~~
3. Consolidate PPO and PPOC
Make separate bodies nicer

## Performance boosting:
Optional
1. Anneal PPO epochs
2. Weight decay
3. Entropy penalty in loss function

## Novel
Change mujoco representation to 3d point cloud
Incorporate tensor field networks as body
PPOC block entropy penalty


## Completed:
Add to PPOC:
~~1. minibatching~~
~~2. lr sched~~

change algo to replicate old PPO on lightbot results:
~~1. minibatching~~
~~2. Add learning rate scheduler~~
~~3. Change activations ro relu (and fix deprecation warnings)~~
~~4. Make sure random sseed is working~~

~~Add continuous action space~~
~~Add coordinates to fourrooms environment for plotting~~
~~Run PPO and PPOC on fourrooms~~
~~Add sequence complexity penalty to PPOC loss~~
~~Make sure it's running on GPU!~~