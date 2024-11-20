from rlberry.manager import AgentManager


MANAGER_FILES = [
    "media/managers/Ball2D/manager_data/manager_obj.pickle",
    "media/managers/MountainCar/manager_data/manager_obj.pickle",
    "media/managers/Acrobot/manager_data/manager_obj.pickle",
]


#
# Video parameters
#
EVAL_EPISODES = 2
VIDEO_LENGTH = 10 # seconds



#
# Load agents
#

managers = []

for fname in MANAGER_FILES:
    managers.append(AgentManager.load(fname))


agents = [manager.get_agent_instances()[0] for manager in managers]


#
# Run policies
#

for agent in agents:
    horizon = agent.horizon
    eval_env = agent.eval_env
    eval_env.enable_rendering()
    n_frames = 0
    for _ in range(EVAL_EPISODES):
        state = eval_env.reset()
        for hh in range(horizon):
            action = agent.policy(state)
            state, _, done, _ = eval_env.step(action)
            n_frames +=1
            if done:
                break
    framerate = max(10, n_frames // VIDEO_LENGTH)
    eval_env.save_video(f"media/{eval_env.unwrapped.name}.mp4", framerate=framerate)
