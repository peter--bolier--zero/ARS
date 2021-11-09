# AI 2018 ARS Papaer - following Udemy Course https://www.udemy.com/course/artificial-intelligence-ars/
#
# Paper: Simple random search provides a competitive approach to reinforcement learning
# March 20, 2018
#
# Description on page 6 / section 3
# Algorithm targeted at version 2, with normalized states and rewards

import numpy as np

# To maintain the hyperparameter set
class HyperParameters:
    def __init__(self):
        # a step_size from paper not yet used?
        self.number_of_steps = 1024         # Number of time we're updating the model in one total run
        self.episode_lenght = 1200          # Maximum time AI is going to wlak 
        self.learning_rate = 0.02           # how fast is AI to learn (adapt)
        self.number_of_directions = 20      # N or pertubations
        self.number_of_best_directions = 16 # b The best, or the top
        self.noise = 0.03                   # v
        self.seed = 1                       # Basically seed for random numer to be used in environemnt
        self.environment_name = ''          # Name of the environment
        
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
        self.var += (self.mean_diff / self.n).clip(min=1e-2)

    # So now normalize the input, what we 'see' of our environment
    def normalize(self, inputs):
        observed_mean = self.mean
        observed_std  = np.sqrt(self.var)
        return (inputs - observed_mean) / observed_std
        



# Policy is function of our perceptron translating input into action(s)

        