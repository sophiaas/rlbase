## Debugging:

1. is PPOC really working?
2. Running rewards - what is happening?
3. Is the option-state match fix working?

## Refactoring:

1. Configs and train files
2. Add ability to call train files with arguments
3. Consolidate PPO and PPOC


## Completed:

~~add weight decay~~

Add to PPOC:
~~1. minibatching~~
~~2. lr sched~~

change algo to replicate old PPO on lightbot results:
~~1. minibatching~~
~~2. Add learning rate scheduler~~
~~3. Change activations ro relu (and fix deprecation warnings)~~
~~4. Make sure random seed is working~~

~~Add continuous action space~~
~~Add coordinates to fourrooms environment for plotting~~
~~Run PPO and PPOC on fourrooms~~
~~Add sequence complexity penalty to PPOC loss~~
~~Make sure it's running on GPU!~~