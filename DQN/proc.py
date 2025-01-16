import numpy as np
from Environment import Game2048Env
from rl.core import Processor

class OneHotInputProcessor(Processor):
    def __init__(self, num_one_hot_matrices=16, window_length=1, model="dnn"):
        self.num_one_hot_matrices = num_one_hot_matrices
        self.window_length = window_length
        self.model = model
        
        self.game_env = Game2048Env() # we make an instance of the game environment so that we can use the functions implementing the game logic
        
        # Variables used by one_hot_encoding() function:
        self.table = {2**i:i for i in range(1,self.num_one_hot_matrices)} 
        self.table[0] = 0 # Add element {0: 0} to the dictionary
    
    def one_hot_encoding(self, grid):
        grid_onehot = np.zeros(shape=(self.num_one_hot_matrices, 4, 4))
        for i in range(4):
            for j in range(4):
                grid_element = grid[i, j]
                grid_onehot[self.table[grid_element],i, j]=1
        return grid_onehot

    def get_grids_next_step(self, grid):
        grids_list = [] # list storing the 4 possible grids at the next step, one for each possible movement
        for movement in range(4):
            grid_before = grid.copy()
            self.game_env.set_board(grid_before)
            try:
                # we retrieve all 4 matrices - even for illegal moves - in order to
                # keep the input to the network the same size
                _ = self.game_env.move(movement) # move() returns a score which is useless in this case
            except:
                pass
            grid_after = self.game_env.get_board()
            grids_list.append(grid_after)
        return grids_list

    def process_observation(self, observation):
        # We reshape the observation (i.e., the grid representing the board-matrix of the game 2048) to make sure we have a 4x4 numpy.array
        observation = np.reshape(observation, (4, 4))
        
        grids_list_step1 = self.get_grids_next_step(observation)
        grids_list_step2 =[]
        for grid in grids_list_step1:
            grids_list_step2.append(grid) # In the NN input I give both, the 1-step and 2-step grids
            grids_temp = self.get_grids_next_step(grid)
            for grid_temp in grids_temp:
                grids_list_step2.append(grid_temp)
        grids_list = np.array([self.one_hot_encoding(grid) for grid in grids_list_step2])
        
        return grids_list