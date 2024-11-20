import numpy as np
from rlberry.envs import MountainCar
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rs_kernel_ucbvi import RSKernelUCBVIAgent
from rlberry.manager import AgentManager, MultipleManagers
from env_ctors import acrobot_ctor

#
# Params
#
N_EPISODES = 5_000


ENVS = { 
    # "Ball2D": (PBall2D, dict(reward_centers=[np.array([0.6, 0.6])])),
    # "MountainCar": (MountainCar, {}),
    "Acrobot": (acrobot_ctor, {}),
}

INIT_KWARGS = {
    "Ball2D": dict(   
        gamma=1.0,
        bonus_scale_factor=0.01,
        horizon=25,
        real_time_dp=False,
        min_dist=0.05,
        bandwidth=0.05,
        store_data_for_vis=True),
    "MountainCar":  dict(   
        gamma=1.0,
        bonus_scale_factor=0.01,
        horizon=160,
        real_time_dp=False,
        min_dist=0.05,
        bandwidth=0.01,
        store_data_for_vis=True),
    "Acrobot": dict(
        gamma=0.99,
        horizon=300,
        bonus_scale_factor=0.01,
        min_dist=0.2,
        bandwidth=0.05,
        beta=1.0,
        store_data_for_vis=True),
}


#
# Build managers
#
managers = MultipleManagers()

for env_name in ENVS:
    env = ENVS[env_name]
    managers.append(
        AgentManager(
            RSKernelUCBVIAgent,
            env,
            fit_budget=N_EPISODES,
            init_kwargs=INIT_KWARGS[env_name],
            n_fit=1,
            output_dir=f'media/managers/{env_name}',
            outdir_id_style=None,
        )
    )

managers.run(save=True)


