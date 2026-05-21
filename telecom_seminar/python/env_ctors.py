from rlberry.envs import Acrobot
from rlberry.wrappers import RescaleRewardWrapper


def acrobot_ctor():
    env = Acrobot()
    env = RescaleRewardWrapper(env, (0.0, 1.0))
    return env
