# AI 2018 ARS Papaer - following Udemy Course https://www.udemy.com/course/artificial-intelligence-ars/
#
# Paper: Simple random search provides a competitive approach to reinforcement learning
# March 20, 2018
#
# Description on page 6 / section 3
# Algorithm targeted at version 2, with normalized states and rewards

from enum import Enum
import gym
from gym import wrappers
import numpy as np
import os
import pybullet_envs


# To maintain the hyperparameter set
class HyperParameters:
    def __init__(self):
        # a step_size from paper not yet used?
        self.number_of_steps = 1024         # Number of time we're updating the model in one total run
        self.episode_length = 1200          # Maximum time AI is going to walk 
        self.learning_rate = 0.02           # how fast is AI to learn (adapt) alfa
        self.number_of_directions = 20      # N or pertubations
        self.number_of_best_directions = 16 # b The best, or the top
        self.noise = 0.03                   # v
        self.seed = 1                       # Basically seed for random numer to be used in environemnt
        self.environment_name = 'HalfCheetahBulletEnv-v0'          # Name of the environment
        
        # Some validation rules on hyperparameters configuration
        # Paper status b < N
        assert self.number_of_best_directions <= self.number_of_directions


# Normalizing the states, section 3.2 so all states are in same range, like 0..1
# State is vector containing all input, what's happening in our environment
# note we need to to this in real time, so for each step...
# Normalisation: 
#    x - mean / standard deviation
# alternative
#    x - min / (max - min) 
class Normalizer:
    # The number of inputs is what our perceptron need to process
    def __init__(self, number_of_inputs):
        self.n         = np.zeros(number_of_inputs) # So a vector of 0's
        self.mean      = np.zeros(number_of_inputs)
        self.mean_diff = np.zeros(number_of_inputs)
        self.var       = np.zeros(number_of_inputs)
    
    # observe a new state (x) - and update stats
    def observe(self, x):
        self.n += 1.0 # increment all elements of vector 
        # online computation of mean
        mean_last_cycle = self.mean.copy()
        self.mean += (x - mean_last_cycle) / self.n
        # online computation of variance
        self.mean_diff += (x - mean_last_cycle) * (x - self.mean)
        # clip variance to prevent div by 0 later
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    # So now normalize the input, what we 'see' of our environment
    def normalize(self, inputs):
        observed_mean = self.mean
        observed_std  = np.sqrt(self.var)
        return (inputs - observed_mean) / observed_std
        
    def observe_normalized(self, state):
        self.observe(state)
        return self.normalize(state)
    
# todo perhaps use enum
class Direction(Enum):
    negative = -1
    none = 0
    positive = 1


# Policy is function of our perceptron translating input into action(s)
# With ARS we explore multiple policies, or policy space
class Policy():
    def __init__(self, number_of_inputs, number_of_outputs, hyper_parameters):
        # matrix of weights theta row, columns (see paper), M
        self.hp = hyper_parameters
        self.theta = np.zeros((number_of_outputs, number_of_inputs))
        
    # Step 5 / V2
    # Create policy space - delta = pertubations pos/neg, noise is normal distribution
    def evaluate(self, input, delta = None, direction = None):
        # or split up in several methods ?
        if direction is None:
            # just matrix * vector
            return self.theta.dot(input)
        elif direction == 'positive':
            # should we use a global variable ?
            return (self.theta + self.hp.noise * delta).dot(input)
        else:
            return (self.theta - self.hp.noise * delta).dot(input)

    # get delta weight as matrix (like theta) distributed normaly (randn) for each direction 
    def sample_deltas(self):
        # * to get all dimensions of matrix
        # for all directions we need a matrix
        return [np.random.randn(*self.theta.shape) for i in range(self.hp.number_of_directions)]
        
    # 7 update step, kind of gradient descent (approximate by measuring difference between rewards)
    # Goal to increase the reward
    # see https://en.wikipedia.org/wiki/Finite_difference
    # episode or a rollout is list of (reward pos, reward neg, pertubation (d) of the direction)
    def update(self, rollouts, sigma_reward):
        # sum over best directions
        step = np.zeros(self.theta.shape) # unfortunetaly expects diff format
        for reward_pos, reward_neg, d in rollouts:
            step += (reward_pos - reward_neg) * d
        self.theta += self.hp.learning_rate / (self.hp.number_of_best_directions * sigma_reward) * step
            
# exploration / step 6 ?
# we need to explore the policy space 
# either time / length, goal or fail is achieved
# First a helper function for one exploration

# Explore one policy on one spcific direction and for one full episode
def explore(environment, normalizer, policy, hyper_parameters, direction = None, delta = None):
    # start fresh
    observation = environment.reset() # from pybullet https://pybullet.org/
    done = False
    number_actions_played = 0.0
    sum_rewards = 0.0 # or another relevant measure
    
    # Let explore the policy with given direction etc.
    while not done and number_actions_played < hyper_parameters.episode_length:
        # proces one step / feed input into perceptron
        observation = normalizer.observe_normalized(observation) # state aka inputs aka observation
        action = policy.evaluate(observation, delta, direction)
        observation, reward, done, _ = environment.step(action) # not using info
        # carefull with outliers of rewards limit range to -1..+1
        reward = max( min(reward, 1), -1)
        
        sum_rewards += reward
        number_actions_played += 1
    return sum_rewards


# Training of the AI, or explore the policy space
def train(environment, policy, normalizer, hyper_parameters):
    
    for step in range(0, hyper_parameters.number_of_steps):
        # Set up the space to explore, like deltas
        # so pertubation deltas and positive and negaive rewards
        deltas = policy.sample_deltas()
        positive_pertubation_rewards = [0] * hyper_parameters.number_of_directions
        negative_pertubation_rewards = [0] * hyper_parameters.number_of_directions
        
        # explore the space
        # getting the positive rewards in the positive direction
        for d in range(0, hyper_parameters.number_of_directions):
            positive_pertubation_rewards[d] = explore(environment, normalizer, policy, hyper_parameters, direction = 'positive', delta = deltas[d])

        # getting the positive rewards in the positive direction
        for d in range(0, hyper_parameters.number_of_directions):
            negative_pertubation_rewards[d] = explore(environment, normalizer, policy, hyper_parameters, direction = 'negative', delta = deltas[d])
        
        # scale / normalize
        # get all rewards in one list so we can get the statistics
        all_rewards = np.array(positive_pertubation_rewards + negative_pertubation_rewards)
        sigma_r = all_rewards.std()
        
        # sort (the rollouts) by maximum and select the best (directions), modding into dictionary
        max_rewards = { k:max(r_pos, r_neg)  for k,(r_pos, r_neg) in enumerate(zip(positive_pertubation_rewards, negative_pertubation_rewards)) }
        best_rewards_key_sorted = sorted(max_rewards.keys(), key = lambda x : max_rewards[x])[0:hyper_parameters.number_of_best_directions] # slice the top
       
        # get rewards using the keys
        rollouts = [(positive_pertubation_rewards[k], negative_pertubation_rewards[k], deltas[k]) for k in best_rewards_key_sorted]
        
        # time to do one step (phew) - update policy (weigths)
        policy.update(rollouts, sigma_r)
        
        # report best stuff
        reward_evaluation = explore(environment, normalizer, policy, hyper_parameters) # no direction or delta
        print('Step: {} Reward {} '.format(step, reward_evaluation ))
        

# helpers
# define path and create it if it doesn not yet exist
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    print('Lets get started - Set up')
    
    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')    
    
    hyper_parameters = HyperParameters()
    np.random.seed(hyper_parameters.seed) # (optional) seed so we can compare runs - testing
    
    # easiest way to use pybullet thorugh gym
    environment_gym = gym.make(hyper_parameters.environment_name)
    environment_gym = wrappers.Monitor(environment_gym, monitor_dir, force = True) # ignore errors
    
    number_of_inputs = environment_gym.observation_space.shape[0] # Todo check gym doc
    number_of_outputs = environment_gym.action_space.shape[0]
    
    policy = Policy(number_of_inputs, number_of_outputs, hyper_parameters)
    normalizer = Normalizer(number_of_inputs)
    
    print('Lets get started - Train')
    train(environment_gym, policy, normalizer ,hyper_parameters)