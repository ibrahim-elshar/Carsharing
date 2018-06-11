import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Carsharing-v0',
    entry_point='gym_carsharing.envs:CarsharingEnv',
)
