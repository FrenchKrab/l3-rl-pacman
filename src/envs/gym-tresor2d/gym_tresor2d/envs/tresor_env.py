import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

#The different cell types
CELLTYPE_GROUND = 0
CELLTYPE_WALL = 1
CELLTYPE_TREASURE = 2


#-------maze generation functions------------

def generate_empty(size):
	maze = [[0 for x in range(size[1])] for y in range(size[0])]
	maze[size[0]-1][size[1]-1] = CELLTYPE_TREASURE
	return maze


def generate_random_obstacles(spawn_position, goal_position, size, density=0.2):
    maze = [[0 for x in range(size[1])] for y in range(size[0])]
    obstacle_count = int(size[0]*size[1]*density)
    random.seed(123409873)
    for _ in range(obstacle_count):
        x=random.randint(0, size[0]-1)
        y=random.randint(0, size[1]-1)
        maze[x][y]=CELLTYPE_WALL
    
    for x in range(size[0]):
        maze[x][0] = CELLTYPE_GROUND
    for y in range(size[1]):
        maze[size[0]-1][y] = CELLTYPE_GROUND

    maze[goal_position[0]][goal_position[1]] = CELLTYPE_TREASURE
    return maze


def generate_zigzag_maze(spawn_position, size):
    maze = [[0 for x in range(size[1])] for y in range(size[0])]
    for x in range(int(size[0]/2)):
        y_offset = 1 if x%2==0 else 0
        for y in range(size[1]-1):
            maze[x*2][y+y_offset] = CELLTYPE_WALL
    goal_position = [size[0]-1, 0 if size[0]/2%2==0 else size[1]-1]
    maze[goal_position[0]][goal_position[1]] = CELLTYPE_TREASURE
    return maze



class Tresor2dEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        #Retrieve kwargs
        self.maze_size = [5,5]
        self.generation_type = "zigzag"
        if "width" in kwargs:
            self.maze_size[0] = kwargs["width"]
        if "height" in kwargs:
            self.maze_size[1] = kwargs["height"]
        if "generation" in kwargs:
            self.generation_type = kwargs["generation"]

        #Terrain generation
        self.spawn_position = np.array([0,0])
        self.goal_position = np.array([max(0, self.maze_size[0]-1), max(0, self.maze_size[1]-1)])
        
        if self.generation_type == "empty":
        	self.maze = generate_empty(self.maze_size)
        elif self.generation_type == "random":
            self.maze = generate_random_obstacles(self.spawn_position, self.goal_position, self.maze_size, 0.5)
        else:
            self.maze = generate_zigzag_maze(self.spawn_position, self.maze_size)

        #Current state as an array [x,y]
        self.state_2d = np.copy(self.spawn_position)

        #Gym spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.maze_size[0]*self.maze_size[1])


    
    def step(self, action):
        """Step the environnment with an action"""
        reward = -1
        done = False
        
        #Find the offset linked to the action
        offset = np.array([0,0])
        if action==0:   #up
            offset = np.array([0,-1])
        elif action==1:     #right
            offset =  np.array([1,0])
        elif action==2:     #down
            offset =  np.array([0,1])
        elif action==3:     #left
            offset =  np.array([-1,0])
        
        #new position after applying the action
        new_2d_pos = np.add(offset, self.state_2d)

        #test if the state is valid (not outside the maze or in a wall)
        valid = self._is_state_valid(new_2d_pos)

        if not valid:
            reward = -10
        else:
            self.state_2d = np.copy(new_2d_pos)

            #test if the new position is a terminal state (reached the goal)
            done = self._is_state_terminal(self.state_2d)

            if done:
                reward = 100
            #prototype reward function that could help for DQN agents
            #else:
            #   reward = 50*(1-np.linalg.norm(self.goal_position-self.state_2d)/np.linalg.norm(self.goal_position-self.spawn_position))

        return [self._pos_2d_to_index(self.state_2d), reward, done, None]
        #return [self.state_2d, reward, done, None]

    
    def reset(self):
        """Reset the environment state"""
        self.state_2d = np.copy(self.spawn_position)
        return self._pos_2d_to_index(self.state_2d)
        #return self.state_2d

    
    def render(self, mode='human', close=False):
        """Display text in the console to render the game"""
        content = ""
        for y in range(self.maze_size[1]):
            line = ""
            for x in range(self.maze_size[0]):
                if self.state_2d[0]==x and self.state_2d[1]==y:
                    line += "o"
                else:
                    cell_type = self.maze[x][y]
                    if cell_type == CELLTYPE_GROUND:
                        line += "."
                    elif cell_type == CELLTYPE_WALL:
                        line += '#'
                    elif cell_type == CELLTYPE_TREASURE:
                        line += 'x'
                line += " "
            content += line + "\n"
        print(content + "\r")
        pass


    def _pos_2d_to_index(self, position):
        """Converts a 2d position to an index"""
        return position[0] + position[1] * self.maze_size[0]


    def _is_state_valid(self, state):
        """Returns True if the state is valid"""
        #If we go off the game space, or land on a wall
        if (state[0] < 0 or state[1]<0 or state[0] >= self.maze_size[0] or state[1] >=self.maze_size[1]
                or self.maze[state[0]][state[1]]==CELLTYPE_WALL):
            return False
        return True
    

    def _is_state_terminal(self, state):
        """Returns true if the state is terminal"""
        if (self.maze[state[0]][state[1]]==CELLTYPE_TREASURE):
            return True
        return False
