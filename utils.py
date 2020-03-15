import numpy as np

def roll_dice(sides):
    return(np.random.random_integers(1, sides, 1)[0])


def calculate_distance(coord1, coord2):
    return(np.sum(np.sqrt((coord1 - coord2)**2)))