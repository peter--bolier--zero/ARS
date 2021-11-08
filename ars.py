# AI 2018 ARS Papaer - following Udemy Course https://www.udemy.com/course/artificial-intelligence-ars/
#
# Paper: Simple random search provides a competitive approach to reinforcement learning
# March 20, 2018
#
# Description on page 6 / section 3
# Algorithm targeted at version 2, with normalized states and rewards



# To maintain the hyperparameter set
class HyperParameters:
    def __init__(self):
        self.number_of_steps = 1024         # Number of time we're updating the model in one total run
        self.episode_lenght = 1200          # Maximum time AI is going to wlak 
        self.learning_rate = 0.02           # how fast is AI to learn (adapt)
        self.number_of_directions = 20      # or pertubations
        self.number_of_best_directions = 16 # The best, or the top
        self.noise = 0.03
        self.seed = 1                       # Basically seed for random numer to be used in environemnt
        self.environemnt_name = ''          # Name of the environment
        
        # Some validaito rules on hyperparameters configuration
        assert self.number_of_best_directions <= self.number_of_directions
        