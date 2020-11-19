import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import linalg as LA
import math

from gym_racecar.envs.race_car_env import RaceCarEnv


class RaceCarCenterLineRealisticEnv(RaceCarEnv):
    def __init__(self,episode_len = 400,action_obs = True):
        super(RaceCarCenterLineRealisticEnv, self).__init__(track_name="TrackCenter",episode_len = episode_len,model_rand = True,action_obs = action_obs)

