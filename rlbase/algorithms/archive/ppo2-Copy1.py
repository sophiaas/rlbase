import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
from envs import Lightbot
from core.config import *
from lightbot_config import config
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
#     def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
    def __init__(self, config):
#         self.lr = lr
#         self.betas = betas
#         self.gamma = gamma
#         self.eps_clip = eps_clip
#         self.K_epochs = K_epochs
        self.config = config
        self.env = self.config.env.init_env()
        print('env: {}'.format(self.env))
#         self.memory = ReplayBuffer(config)
#         self.logger = Logger(config)
        self.episode = 0
        self.running_rewards = None
        self.running_moves = None
        
        self.set_network_params()
        
        # initialize networks
        self.observation_net = config.network.init_body()
        self.model = config.network.init_heads(self.observation_net)
        self.policy = self.model['policy'].to(device)
        self.policy_old = copy.deepcopy(self.model['policy']).to(device)
        self.value_net = self.model['value'].to(device)
        
        # initialize optimizers
        self.policy_optimizer = config.training.optim(self.policy.parameters(), 
                                    lr=config.training.lr,
                                    weight_decay=config.training.weight_decay)
        self.value_optimizer = config.training.optim(self.value_net.parameters(), 
                                    lr=config.training.lr,
                                    weight_decay=config.training.weight_decay)
        self.optimizer = {'policy': self.policy_optimizer, 
                          'value': self.value_optimizer}
        
        # initialize learning rate schedulers
        self.policy_lr_scheduler = config.training.lr_scheduler( 
                                    self.policy_optimizer, step_size=1, 
                                    gamma=config.training.lr_gamma)
        self.value_lr_scheduler = config.training.lr_scheduler(
                                    self.value_optimizer, step_size=1, 
                                    gamma=config.training.lr_gamma)
        self.lr_scheduler = [self.policy_lr_scheduler, self.value_lr_scheduler]
        
        print('value_net \n {}'.format(self.value_net))
        print('policy \n {}'.format(self.policy))
        
    def policy_evalute(self, state, action):
        action_probs = self.policy(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_net(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
    def policy_old_act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()

                
    def set_network_params(self):
        # set network dimensions that are dependent on environment
        self.config.network.body.indim = self.env.observation_space.n
        self.config.network.heads['policy'].outdim = self.env.action_space.n
        self.config.network.heads['value'].outdim = 1

        
#         self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
#         self.optimizer = torch.optim.Adam(self.policy.parameters(),
#                                               lr=lr, betas=betas)
#         self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        
        self.MseLoss = nn.MSELoss()

    def policy_evaluate(self, state, action):
        action_probs = self.policy.forward(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_net(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.config.algorithm.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(4):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy_evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.config.algorithm.clip, 1+self.config.algorithm.clip) * advantages
            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
#             loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
#             for o in self.optimizer.values():
#                 o.zero_grad()
            self.optimizer['value'].zero_grad()
            value_loss.mean().backward(retain_graph=True)
            self.optimizer['value'].step()
            
            self.optimizer['policy'].zero_grad()
            policy_loss.mean().backward()
            self.optimizer['policy'].step()
#             self.optimizer.zero_grad()
#             policy_loss.mean().backward()
#             for o in self.optimizer.values():
#                 o.step()
#             loss.mean().backward()
#             self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
#     env_name = "LunarLander-v2"
    # creating environment
#     env = gym.make(env_name)
    env = Lightbot(LightbotConfig())
    state_dim = env.observation_space.n
    action_dim = 5
    render = False
#     solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
#     ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    ppo = PPO(config)

#     print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old_act(state, memory)
            state, reward, done, _ = env.step(action)
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
#         # stop training if avg_reward > solved_reward
#         if running_reward > (log_interval*solved_reward):
#             print("########## Solved! ##########")
#             torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
#             break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
