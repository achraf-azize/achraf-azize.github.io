import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rlberry.manager import AgentManager

import matplotlib
from pylab import rcParams
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 42
matplotlib.rcParams.update({'errorbar.capsize': 0})
matplotlib.rcParams['text.usetex'] = True


MANAGER_FILE = "media/managers/Ball2D/manager_data/manager_obj.pickle"



#
# Video parameters
#
EVAL_EPISODES = 2
VIDEO_LENGTH = 10 # seconds



#
# Load agents
#

manager = AgentManager.load(MANAGER_FILE)

agent = manager.get_agent_instances()[0]

#
# Run policies
#

# horizon = agent.horizon
# eval_env = agent.eval_env
# eval_env.enable_rendering()
# n_frames = 0
# for _ in range(EVAL_EPISODES):
#     state = eval_env.reset()
#     for hh in range(horizon):
#         action = agent.policy(state)
#         state, _, done, _ = eval_env.step(action)
#         n_frames +=1
#         if done:
#             break

# eval_env.render()


#
# Plot Backward Induction idea
#

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))


horizon = agent.horizon
repr_states = agent.representative_states[:agent.M]
v_func = agent.Q[horizon // 2, :agent.M].max(axis=-1)


axes[0].scatter(repr_states[:, 0], repr_states[:, 1], s=256)
axes[0].axis("off")
axes[0].set_title("observed states $s_h^i$")

axes[1].scatter(repr_states[:, 0], repr_states[:, 1], s=256, c=v_func, cmap=cm.cividis)
axes[1].axis("off")
axes[1].set_title("$\\widetilde{Q}_h^t(s_h^i, a)$")

#
# Interpolation
#

N = 100
M = agent.M
theta_values = np.linspace(0, 2.0*np.pi, N)
radius_values = np.linspace(0.0, 1.0, N)
LipQ    = 10.0  # only for visualization

THETA, RADIUS     = np.meshgrid(theta_values, radius_values)
X = RADIUS * np.cos(THETA)
Y = RADIUS * np.sin(THETA)
coords   = np.vstack(  (X.flatten(), Y.flatten()) ).T
mask = (coords**2 + coords**2).sum(axis=-1) <= 1
interp_q = np.zeros( N*N )


for ii in range(N*N):
    x = coords[ii, 0]
    y = coords[ii, 1]
    minval = np.inf
    for jj in range(M):
        dist = (x- repr_states[jj, 0])**2 + (y- repr_states[jj, 1])**2
        dist = np.sqrt(dist)
        aux = v_func[jj] + LipQ*dist  
        if aux < minval:
            minval = aux 
    interp_q[ii] = minval

axes[2].scatter(
    coords[mask, 0],
    coords[mask, 1],
    c = interp_q[mask],
    s=256, cmap=cm.cividis)
axes[2].axis("off")
axes[2].set_title("$Q_h^t(s, a)$")


fig.savefig("images/kernel_vi.png", transparent=True, bbox_inches='tight')
plt.show()