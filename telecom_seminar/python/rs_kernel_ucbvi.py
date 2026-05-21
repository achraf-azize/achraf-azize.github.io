import logging

import numpy as np
from rlberry.utils.jit_setup import numba_jit
from rlberry.utils.factory import load

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.dynprog.utils import backward_induction
from rlberry.agents.dynprog.utils import backward_induction_in_place
from rs_kernel_ucbvi_utils import map_to_representative
from rs_kernel_ucbvi_utils import kernel_func
from rs_kernel_ucbvi_utils import metric_lp
from rs_kernel_ucbvi_utils import update_value_and_get_action


logger = logging.getLogger(__name__)


@numba_jit
def update_model(repr_state, action, repr_next_state, reward,
    n_representatives, repr_states, lp_metric, scaling, bandwidth,
    bonus_scale_factor, beta, v_max, bonus_type, kernel_type,
    N_sa, B_sa, P_hat, R_hat, ns_discount):
    """
    Model update function, lots of arguments so we can use JIT.
    """
    # aux var for transition update
    dirac_next_s = np.zeros(n_representatives)
    dirac_next_s[repr_next_state] = 1.0

    for u_repr_state in range(n_representatives):
        # compute weight
        dist = metric_lp(
            repr_states[repr_state, :],
            repr_states[u_repr_state, :],
            lp_metric,
            scaling)
        weight = kernel_func(dist / bandwidth, kernel_type=kernel_type)

        # aux variables
        prev_N_sa = beta + N_sa[u_repr_state, action]  # regularization beta
        current_N_sa = beta + weight + ns_discount * N_sa[u_repr_state, action] 

        # update weights
        N_sa[u_repr_state, action] = weight + ns_discount * N_sa[u_repr_state, action]

        # update transitions
        P_hat[u_repr_state, action, :n_representatives] = (
            dirac_next_s * weight / current_N_sa +
            ns_discount * (prev_N_sa / current_N_sa) *
            P_hat[u_repr_state, action, :n_representatives]
        )

        # update rewards
        R_hat[u_repr_state, action] = weight * reward / current_N_sa + \
            ns_discount * (prev_N_sa / current_N_sa) * R_hat[u_repr_state, action]

        # update bonus
        B_sa[u_repr_state, action] = compute_bonus(N_sa[u_repr_state, action],
                                                   beta, bonus_scale_factor,
                                                   v_max, bonus_type)


@numba_jit
def compute_bonus(sum_weights, beta, bonus_scale_factor, v_max, bonus_type):
    n = beta + sum_weights
    if bonus_type == "simplified_bernstein":
        return bonus_scale_factor * np.sqrt(1.0 / n) + (1 + beta) * (v_max) / n
    else:
        raise NotImplementedError("Error: unknown bonus type.")


class RSKernelUCBVIAgent(AgentWithSimplePolicy):
    """
    Implements KernelUCBVI [1] with representative states [2, 3].

    Value iteration with exploration bonuses for continuous-state environments,
    using a online discretization strategy + kernel smoothing:
    - Build (online) a set of representative states
    - Using smoothing kernels, estimate transtions an rewards on the
    finite set of representative states and actions.

    Criterion: finite-horizon with discount factor gamma.
    If the discount is not 1, only the Q function at h=0 is used.

    The recommended policy after all the episodes is computed without
    exploration bonuses.


    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    gamma : double
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
    lp_metric: int
        The metric on the state space is the one induced by the p-norm,
        where p = lp_metric. Default = 2, for the Euclidean metric.
    kernel_type : string
        See rlberry.agents.kernel_based.kernels.kernel_func for
        possible kernel types.
    scaling: numpy.ndarray
        Must have the same size as state array, used to scale the states
        before computing the metric.
        If None, set to:
        - (env.observation_space.high - env.observation_space.low) if high
            and low are bounded
        - np.ones(env.observation_space.shape[0]) if high or low
        are unbounded
    bandwidth : double
        Kernel bandwidth.
    min_dist : double
        Minimum distance between two representative states
    max_repr : int
        Maximum number of representative states.
        If None, it is set to  (sqrt(d)/min_dist)**d, where d
        is the dimension of the state space
    bonus_scale_factor : double
        Constant by which to multiply the exploration bonus,
        controls the level of exploration.
    beta : double
        Regularization constant.
    bonus_type : string
            Type of exploration bonus. Currently, only "simplified_bernstein"
            is implemented.
    real_time_dp: bool, default: False
        If True, use real-time dynamic programming
    embedding_fn : Callable
        Function to preprocess states.
    non_stationary_discount : float, default = 1.0
        Discount to handle non-stationary environments (RS-KeRNS).
        Set to 1.0 for stationary environments.
    restart_reward_period: int, default = None
        If given, reward estimator is set to zero every `restart_reward_period`
        episodes. Used as baseline for non-stationary environments.
    store_data_for_vis: bool, default = False
        If True, store data for visualization
    break_when_done: bool, default = False
        If True, break interaction loop when done=True
    References
    ----------
    [1] Domingues et al., 2020
        Regret Bounds for Kernel-Based Reinforcement Learning
        https://arxiv.org/abs/2004.05599
    [2] Domingues et al., 2020
        A Kernel-Based Approach to Non-Stationary Reinforcement Learning
        in Metric Spaces
        https://arxiv.org/abs/2007.05078
    [3] Kveton & Theocharous, 2012
        Kernel-Based Reinforcement Learning on Representative States
        https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewFile/4967/5509
    """

    name = "RS-KernelUCBVI"

    def __init__(self, env,
                 gamma=0.99,
                 horizon=None,
                 lp_metric=2,
                 kernel_type="epanechnikov",
                 scaling=None,
                 bandwidth=0.05,
                 min_dist=0.1,
                 max_repr=1000,
                 bonus_scale_factor=1.0,
                 beta=0.01,
                 bonus_type="simplified_bernstein",
                 real_time_dp=False,
                 embedding_fn=None,
                 non_stationary_discount=1.0,
                 restart_reward_period=None,
                 store_data_for_vis=False,
                 break_when_done=False,
                 **kwargs):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.horizon = horizon
        self.lp_metric = lp_metric
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.min_dist = min_dist
        self.bonus_scale_factor = bonus_scale_factor
        self.beta = beta
        self.bonus_type = bonus_type
        self.real_time_dp = real_time_dp
        self.non_stationary_discount = non_stationary_discount
        self.restart_reward_period = restart_reward_period
        self.store_data_for_vis = store_data_for_vis
        self.break_when_done =break_when_done

        # handle possible embedding function
        if isinstance(embedding_fn, str):
            embedding_fn = load(embedding_fn)
        self.embedding_fn = embedding_fn

        if non_stationary_discount < 1.0:
            self.name = "RS-KeRNS"
        
        if restart_reward_period is not None:
            self.name = "Restart-Baseline"
    
        if real_time_dp:
            self.name = self.name + "+RTDP"

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, \
                "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))

        # state dimension
        self.state_dim = self.env.observation_space.shape[0]

        # compute scaling, if it is None
        if scaling is None:
            # if high and low are bounded
            if (self.env.observation_space.high == np.inf).sum() == 0 \
                    and (self.env.observation_space.low == -np.inf).sum() == 0:
                scaling = self.env.observation_space.high \
                    - self.env.observation_space.low
                # if high or low are unbounded
            else:
                scaling = np.ones(self.state_dim)
        else:
            assert scaling.ndim == 1
            assert scaling.shape[0] == self.state_dim
        self.scaling = scaling

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon-1)):
            self.v_max[hh] = r_range + self.gamma*self.v_max[hh+1]


        # number of representative states and number of actions
        if max_repr is None:
            max_repr = int(np.ceil((1.0 * np.sqrt(self.state_dim)
                                    / self.min_dist) ** self.state_dim))
        self.max_repr = max_repr

        # current number of representative states
        self.M = None
        self.A = self.env.action_space.n

        # declaring variables
        self.episode = None  # current episode
        self.representative_states = None  # coordinates of all repr states
        self.N_sa = None  # sum of weights at (s, a)
        self.B_sa = None  # bonus at (s, a)
        self.R_hat = None  # reward  estimate
        self.P_hat = None  # transitions estimate
        self.Q = None  # Q function
        self.V = None  # V function

        self.Q_policy = None  # Q function for recommended policy

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.M = 0
        self.representative_states = np.zeros((self.max_repr, self.state_dim))
        self.N_sa = np.zeros((self.max_repr, self.A))
        self.B_sa = self.v_max[0] * np.ones((self.max_repr, self.A))

        self.R_hat = np.zeros((self.max_repr, self.A))
        self.P_hat = np.zeros((self.max_repr, self.A, self.max_repr))

        self.V = self.v_max[0] * np.ones((self.horizon, self.max_repr))
        self.Q = np.zeros((self.horizon, self.max_repr, self.A))
        self.Q_policy = None

        self.episode = 0

    def policy(self, observation):
        state = observation
        if self.embedding_fn is not None:
            state = self.embedding_fn(state)

        assert self.Q_policy is not None
        repr_state = self._map_to_repr(state, False)
        return self.Q_policy[0, repr_state, :].argmax()

    def fit(self, budget: int, **kwargs):
        del kwargs

        if self.store_data_for_vis:
            self.vis_all_representatives = np.zeros((budget, self.max_repr, self.state_dim))
            self.vis_n_representatives = np.zeros(budget, dtype=np.uint32)
            self.vis_weights = np.zeros((budget, self.max_repr))

        for ep in range(budget):
            self._run_episode()
            if self.store_data_for_vis:
                self.vis_all_representatives[ep] = self.representative_states
                self.vis_n_representatives[ep] = self.M
                self.vis_weights[ep] = self.N_sa.sum(axis=-1)

        # compute Q function for the recommended policy
        self.Q_policy, _ = backward_induction(
            self.R_hat[:self.M, :],
            self.P_hat[:self.M, :, :self.M],
            self.horizon, self.gamma)

    def _map_to_repr(self, state, accept_new_repr=True):
        repr_state = map_to_representative(
            state,
            self.lp_metric,
            self.representative_states,
            self.M,
            self.min_dist,
            self.scaling,
            accept_new_repr)
        # check if new representative state
        if repr_state == self.M:
            self.M += 1
        return repr_state

    def _update(self, state, action, next_state, reward, hh):
        repr_state = self._map_to_repr(state)
        repr_next_state = self._map_to_repr(next_state)

        # Only discount previous count at the beginning of an episode.
        # Assumes that the MDP can only change from one episode to another.
        if hh == 0:
            ns_discount = self.non_stationary_discount
        else:
            ns_discount = 1.0

        update_model(
            repr_state, action, repr_next_state, reward,
            self.M,
            self.representative_states,
            self.lp_metric,
            self.scaling,
            self.bandwidth,
            self.bonus_scale_factor,
            self.beta,
            self.v_max[0],
            self.bonus_type,
            self.kernel_type,
            self.N_sa,
            self.B_sa,
            self.P_hat,
            self.R_hat,
            ns_discount)

    def _get_action(self, state, hh=0):
        repr_state = self._map_to_repr(state, accept_new_repr=False)
        if not self.real_time_dp:
            assert self.Q is not None
            # add noise to break ties
            qvals = self.Q[hh, repr_state, :] + 1e-10 * self.rng.uniform(size=self.env.action_space.n)
            return qvals.argmax()
        else:
            if self.M > 0:
                update_fn = update_value_and_get_action
                return update_fn(
                    repr_state,
                    hh,
                    self.V[:, :self.M],
                    self.R_hat[:self.M, :],
                    self.P_hat[:self.M, :, :self.M],
                    self.B_sa[:self.M, :],
                    self.gamma,
                    self.v_max[hh],
                    )
            else:
                return self.env.action_space.sample()

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        if self.embedding_fn is not None:
            state = self.embedding_fn(state)
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)

            if self.embedding_fn is not None:
                next_state = self.embedding_fn(next_state)

            self._update(state, action, next_state, reward, hh)
            state = next_state
            episode_rewards += reward

            if done and self.break_when_done:
                break

        # run backward induction
        if not self.real_time_dp:
            backward_induction_in_place(
                self.Q[:, :self.M, :], self.V[:, :self.M],
                self.R_hat[:self.M, :] + self.B_sa[:self.M, :],
                self.P_hat[:self.M, :, :self.M],
                self.horizon, self.gamma, self.v_max[0])

        self.episode += 1
        #
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)
            self.writer.add_scalar("representative_states", self.M, self.episode)

        if self.restart_reward_period is not None:
            if self.episode % self.restart_reward_period == 0:
                self.R_hat = np.zeros((self.max_repr, self.A))
                self.B_sa = self.v_max[0] * np.ones((self.max_repr, self.A))

        # return sum of rewards collected in the episode
        return episode_rewards
