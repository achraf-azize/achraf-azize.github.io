import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rlberry.manager import AgentManager
from rlberry.rendering.utils import video_write
from sklearn.manifold import TSNE

import matplotlib
from pylab import rcParams
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 42
matplotlib.rcParams.update({'errorbar.capsize': 0})
matplotlib.rcParams['text.usetex'] = True


MANAGER_FILES = [
    "media/managers/Ball2D/manager_data/manager_obj.pickle",
    "media/managers/MountainCar/manager_data/manager_obj.pickle",
    "media/managers/Acrobot/manager_data/manager_obj.pickle",
]


#
# Video parameters
#
VIDEO_LENGTH = 20 # seconds
N_FRAMES = 200
FRAMERATE = max(10, N_FRAMES // VIDEO_LENGTH)


#
# Load agents
#

managers = []

for fname in MANAGER_FILES:
    managers.append(AgentManager.load(fname))


agents = [manager.get_agent_instances()[0] for manager in managers]


#
# Check if PCA needs to be applied (obs_dim > 2)
#
agents_transformed_representatives = []

for agent in agents:
    if agent.vis_all_representatives.shape[-1] <= 2:
        agents_transformed_representatives.append(None)
    else:
        print(f"Fitting t-SNE ({agent.env})")
        frame_idx = -1
        n_repr = agent.vis_n_representatives[frame_idx]
        X = agent.vis_all_representatives[frame_idx, :n_repr, :]
        X_embedded =TSNE(
            n_components=2, perplexity=100.0,
            learning_rate='auto', init='random').fit_transform(X)
        agents_transformed_representatives.append(X_embedded)



#
# Write video
#

all_images = []
max_weights = [agent.vis_weights.max() for agent in agents]
min_weights = [agent.vis_weights.min() for agent in agents]


frames_to_save = []
for agent in agents:
    frames_to_save.append(
        np.linspace(
            0, agent.vis_n_representatives.shape[0],
            num=N_FRAMES, endpoint=False, dtype=np.int32)
    )

for counter in range(N_FRAMES):
    print(f"getting frame {counter} / {N_FRAMES}")
    fig, axes = plt.subplots(nrows=1, ncols=len(MANAGER_FILES), figsize=(8 * len(agents), 8))
    for ii, agent in enumerate(agents): 
        frame_idx = frames_to_save[ii][counter]
        ax = axes[ii]
        n_repr = agent.vis_n_representatives[frame_idx]

        if agents_transformed_representatives[ii] is not None:
            representatives = agents_transformed_representatives[ii][:n_repr]
        else:
            representatives = agent.vis_all_representatives[frame_idx, :n_repr]

        repr_x = representatives[:, 0]
        repr_y = representatives[:, 1]
        vals = agent.vis_weights[frame_idx, :n_repr]
        vals = np.log(1 + vals)
        min_val = np.log(1 + min_weights[ii])
        max_val = np.log(1 + max_weights[ii])

        eval_env = agent.eval_env
        scale_factor = 1.35
        if agents_transformed_representatives[ii] is None:
            ax.set_xlim([scale_factor*eval_env.observation_space.low[0], scale_factor*eval_env.observation_space.high[0]])
            ax.set_ylim([scale_factor*eval_env.observation_space.low[1], scale_factor*eval_env.observation_space.high[1]])
        else:
            xmin = scale_factor*agents_transformed_representatives[ii][:, 0].min()
            xmax = scale_factor*agents_transformed_representatives[ii][:, 0].max()

            ymin = scale_factor*agents_transformed_representatives[ii][:, 1].min()
            ymax = scale_factor*agents_transformed_representatives[ii][:, 1].max()
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])


        ax.scatter(repr_x, repr_y, c=vals, cmap=cm.cividis, s=512, vmin=min_val, vmax=max_val)

        # ax.text(-0.9, 0.9, s=f"t={frame_idx+1}", fontsize=32, alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis("off")

        if "Ball2D" in MANAGER_FILES[ii]:
            ax.text(
                0.01, 0.5, 
                "y-position",
                ha="left",
                va="top",
                transform=ax.transAxes,
                rotation=90,
                color="red")
            ax.text(
                0.05, 0.1, 
                "x-position",
                ha="left",
                va="top",
                transform=ax.transAxes,
                color="red")

        elif "MountainCar" in MANAGER_FILES[ii]:
            ax.text(
                0.01, 0.5, 
                "car velocity",
                ha="left",
                va="top",
                transform=ax.transAxes,
                rotation=90,
                color="red")
            ax.text(
                0.05, 0.1, 
                "car position",
                ha="left",
                va="top",
                transform=ax.transAxes,
                color="red")
        else:
            ax.text(
                0.01, 0.5, 
                "y-projection",
                ha="left",
                va="top",
                transform=ax.transAxes,
                rotation=90,
                color="red")
            ax.text(
                0.05, 0.1, 
                "x-projection",
                ha="left",
                va="top",
                transform=ax.transAxes,
                color="red")

    fig.tight_layout(pad=0.5)

    # if counter == 2:
    #     plt.show()
    #     break

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    all_images.append(frame)

    plt.close('all')


# save video
video_write("media/kernel_vid.mp4", all_images, framerate=FRAMERATE)