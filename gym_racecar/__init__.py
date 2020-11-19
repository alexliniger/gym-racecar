
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='RaceCar-v0',
    entry_point='gym_racecar.envs:RaceCarIdealLineEnv'
)

register(
    id='RaceCarRealistic-v0',
    entry_point='gym_racecar.envs:RaceCarIdealLineRealisticEnv'
)


register(
    id='RaceCarHard-v0',
    entry_point='gym_racecar.envs:RaceCarCenterLineEnv',
)

register(
    id='RaceCarRealisticHard-v0',
    entry_point='gym_racecar.envs:RaceCarCenterLineRealisticEnv',
)