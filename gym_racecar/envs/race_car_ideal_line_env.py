import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import linalg as LA
import math

from gym_racecar.envs.race_car_env import RaceCarEnv


class RaceCarIdealLineEnv(RaceCarEnv):
    def __init__(self,episode_len = 400,action_obs = True):
        super(RaceCarIdealLineEnv, self).__init__(track_name="TrackIdeal",episode_len = episode_len,model_rand = False,action_obs = action_obs)

