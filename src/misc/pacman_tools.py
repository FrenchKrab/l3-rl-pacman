import numpy as np


#---------post process functions------------

def mspacman_init_data():
    data = {}
    data["lives"] = 3
    return data

def ms_pacman_step_post_process(new_state, reward, done, infos, data):
    dead = infos['ale.lives']<data["lives"]     #Test if pacman is dead
    data["lives"] = infos['ale.lives']          #Update lives counter

    if dead:
        reward = -100   #give negative reward when pacman dies
    
    return new_state, reward, done, infos

def ms_pacman_reset_post_process(data):
    data["lives"] = 3


#--------pre process------------


def normalize_ram(x):
    """Normalizes the ram of an ATARI game"""
    return x / 255