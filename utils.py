import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import product
import pandas as pd


# create a function that runs multiple value_iterations runs with different seeds
def run_value_iterations(env, n_iters=10, n_seeds=5, gamma=1, theta=1e-10):
    planner = Planner(env.P)
    V_list = []
    V_track_list = []
    pi_list = []

    for seed in tqdm(range(n_seeds)):
        np.random.seed(seed)
        random.seed(seed)
        V, V_track, pi = planner.value_iteration(
            gamma=gamma, theta=theta, n_iters=n_iters
        )
        V_list.append(V)
        V_track_list.append(V_track)
        pi_list.append(pi)

    return V_list, V_track_list, pi_list


def value_iter_grid_search(
    env, n_iters=1000, gammas=[0.9, 0.95, 0.99, 0.999], thetas=[1e-3, 1e-6, 1e-9, 1e-12], convert_state_obs=lambda state : state
):
    results = {}
    planner = Planner(env.P)

    # create a list of all possible combinations of gamma and theta using the itertools product function
    for gamma, theta in product(gammas, thetas):
        print(f"Running value iteration for gamma={gamma} and theta={theta}")
        V, V_track, pi, pi_track, timings = planner.value_iteration(
            gamma=gamma, theta=theta, n_iters=n_iters
        )
        episodes_rewards, run_length,_ = TestEnv.test_env(env=env, n_iters=100, pi=pi, convert_state_obs=convert_state_obs)
        mean_reward = np.mean(episodes_rewards)
        print(f"Mean reward: {mean_reward}\n")
        results[(gamma, theta)] = (V, V_track, pi, pi_track, mean_reward, timings)

    return results


def policy_iter_grid_search(
    env, n_iters=1000, gammas=[0.9, 0.95, 0.99, 0.999], thetas=[1e-3, 1e-6, 1e-9, 1e-12], convert_state_obs=lambda state : state
):
    results = {}
    planner = Planner(env.P)

    # create a list of all possible combinations of gamma and theta using the itertools product function
    for gamma, theta in tqdm(product(gammas, thetas)):
        print(f"Running value iteration for gamma={gamma} and theta={theta}")
        V, V_track, pi, pi_track, timings = planner.policy_iteration(
            gamma=gamma, theta=theta, n_iters=n_iters
        )
        episodes_rewards, run_length,_ = TestEnv.test_env(env=env, n_iters=100, pi=pi, convert_state_obs=convert_state_obs)
        mean_reward = np.mean(episodes_rewards)
        print(f"Mean reward: {mean_reward}")
        results[(gamma, theta)] = (V, V_track, pi, pi_track, mean_reward, timings)

    return results


def q_learning_grid_search(
    env,
    gammas=[0.99],
    init_alphas=[0.5],
    min_alphas=[0.01],
    alpha_decay_ratios=[0.5],
    init_epsilons=[1.0],
    min_epsilons=[0.1],
    epsilon_decay_ratios=[0.9],
    seeds = [0,1,2,3,4],
    n_episodes=10000,
):
    results = {}
    combinations = product(gammas, init_alphas, min_alphas, alpha_decay_ratios, init_epsilons, min_epsilons, epsilon_decay_ratios, seeds)
    print(f"Total number of combinations: {len(list(combinations))}")
    for (
        gamma,
        init_alpha,
        min_alpha,
        alpha_decay_ratio,
        init_epsilon,
        min_epsilon,
        epsilon_decay_ratio,
        seed,
    ) in product(
            gammas,
            init_alphas,
            min_alphas,
            alpha_decay_ratios,
            init_epsilons,
            min_epsilons,
            epsilon_decay_ratios,
            seeds,
    ):
        print(
            f"Running Q-learning for gamma={gamma}, alpha={init_alpha}, epsilon={init_epsilon}, n_episodes={n_episodes}, seed={seed}\n \
            with alpha decay ratio={alpha_decay_ratio} and epsilon decay ratio={epsilon_decay_ratio}"
        )
        Q, V, pi, Q_track, pi_track, rewards, endings = RL(env).q_learning(
            gamma=gamma,
            init_alpha=init_alpha,
            min_alpha=min_alpha,
            alpha_decay_ratio=alpha_decay_ratio,
            init_epsilon=init_epsilon,
            min_epsilon=min_epsilon,
            epsilon_decay_ratio=epsilon_decay_ratio,
            n_episodes=n_episodes,
            seed=seed,
        )
        episodes_rewards, run_length, _ = TestEnv.test_env(env=env, n_iters=100, pi=pi)
        mean_reward = np.mean(episodes_rewards)
        mean_run_length = np.mean(run_length)
        print(f"Mean reward: {mean_reward}, Mean run length: {mean_run_length}\n")
        results[
            (
                gamma,
                init_alpha,
                min_alpha,
                alpha_decay_ratio,
                init_epsilon,
                min_epsilon,
                epsilon_decay_ratio,
                seed,
            )
        ] = {
            "Q": Q,
            "V": V,
            "pi": pi,
            "Q_track": Q_track,
            "pi_track": pi_track,
            "mean_reward": mean_reward,
            "rewards": rewards,
            "endings": endings,
            "mean_run_length": mean_run_length,
        }

    return results


def rewards_per_gamma_bar_plot(results, title):
    gammas = []
    mean_rewards = []
    # std_rewards = []

    for key in results:
        # insert new row
        gammas.append(key[0])
        mean_rewards.append(results[key][4])

    sns.barplot(
        x=gammas, y=mean_rewards, hue=gammas, legend=False, palette="viridis"
    ).set_title(title)
    plt.xlabel("Gamma")
    plt.ylabel("Success Rate")
    plt.show()


def v_iters_plot_gammas(results, mode="mean"):
    plt.figure(figsize=(8, 5))
    # find the length of the longest V_track with trim  zeros
    max_len = max(
        [
            len(np.trim_zeros(np.mean(V_track, axis=1), "b"))
            for (_, V_track, _, _, _, _) in results.values()
        ]
    )
    for (gamma, theta), (_, V_track, _, _, _, _) in results.items():
        if mode == "mean":
            value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), "b")
            std_value_per_iter = np.trim_zeros(np.std(V_track, axis=1), "b")
        elif mode == "max":
            value_per_iter = np.trim_zeros(np.max(V_track, axis=1), "b")
            std_value_per_iter = np.trim_zeros(np.std(V_track, axis=1), "b")
        else:
            raise ValueError("Mode must be either 'mean' or 'max'")
        std_value_per_iter = std_value_per_iter ** 3

        # pad the shorter V_track with last value
        value_per_iter = np.pad(
            value_per_iter, (0, max_len - len(value_per_iter)), mode="edge"
        )
        std_value_per_iter = np.pad(
            std_value_per_iter, (0, max_len - len(std_value_per_iter)), mode="edge"
        )

        plt.plot(value_per_iter, label=f"γ: {gamma}")
        if mode == "mean":
            plt.fill_between(
                range(len(value_per_iter)),
                value_per_iter - std_value_per_iter,
                value_per_iter + std_value_per_iter,
                alpha=0.3,
            )
    plt.xlabel("Iterations", fontsize=25)
    if mode == "mean":
        plt.ylabel("Mean V", fontsize=25)
    else:
        plt.ylabel("Max V", fontsize=25)

    # increase font of xticks and yticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # add legend above the plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        shadow=False,
        ncol=5,
        fontsize=16,
    )

    # plt.legend(loc='upper left', fontsize=14)

    plt.savefig(
        f"figures/frozen_lake/{mode}_V_iters.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()


def plot_state_values_evolution(V_track):
    for i in range(V_track.shape[1]):
        plt.plot(
            np.trim_zeros(V_track[:, i], "b"),
            label=f"State {i}",
            color=plt.cm.jet(i / V_track.shape[1]),
        )

    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Frozen Lake Value Iteration Convergence per State")
    plt.show()


def plot_multiple_policies(
    V_track, pi_track, indices=[5, 10, 25, 50], map_size=(8, 8), filename=None
):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fl_actions = {0: -1, 1: 0, 2: 1}

    


    for i, idx in enumerate(indices):
        
        val_max, policy_map = Plots.get_policy_map(
            {i: np.argmax(pi_track[idx][i]) for i in range(256)}, V_track[idx].reshape(map_size), fl_actions, map_size
        )
        p = pi_track[idx]
        sns.heatmap(
            val_max,
            # annot=policy_map,
            cmap=sns.color_palette("viridis", as_cmap=True),
            fmt="",
            # cmap=sns.color_palette("", as_cmap=True),
            # linewidths=0.1,
            # linecolor="black",
            # annot_kws={"fontsize": "xx-large", "fontweight": "bold", "style": "italic"},
            annot=False,
            ax=axs[i],
            cbar=True,
        )
        axs[i].set_title(f"Policy/Value @ Iteration {idx}", fontsize=20)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    if filename:
        plt.tight_layout()
        plt.savefig(
            f"figures/mountain_cart/{filename}.pdf", format="pdf", bbox_inches="tight"
        )
    plt.show()


def policy_changes_plot(pi_track, title, upper_bound=100):
    changes = []
    for i in range(0, upper_bound):
        # count the number of different values between the two policies betweeen i and i-1 (they are dictionaries)
        n_changes = sum(
            [1 for key in pi_track[i] if pi_track[i][key] != pi_track[i - 1][key]]
        )

        changes.append(n_changes)
    plt.plot(changes)
    if upper_bound < 10:
        plt.xticks(range(upper_bound))
    plt.xlabel("Iterations")
    plt.ylabel("Policy Changes")
    plt.title(title)
    plt.show()




# new success rate plot that takes in multiple pi_tracks and plots as many plot as there are pi_tracks (on a single plot)
def success_rate_plot_multiple(env, pi_tracks, upperbound, step=1, title=""):
    success_rate_seeds = []
    for pi_track in pi_tracks:
        success_rate = []
        for i in tqdm(range(0, upperbound, step)):
            episode_rewards, run_length = TestEnv.test_env(
                env=env, n_iters=100, pi=pi_track[i]
            )
            episode_rewards = np.array(episode_rewards)
            # count number of 1s in episode_rewards
            success_count = np.count_nonzero(
                episode_rewards == env.custom_rewards[b"G"]
            )
            success_rate.append(success_count)
        success_rate_seeds.append(success_rate)
    success_rate_seeds = np.array(success_rate_seeds)
    mean_success_rate = np.mean(success_rate_seeds, axis=0)
    std_success_rate = np.std(success_rate_seeds, axis=0)
    max = np.max(success_rate_seeds, axis=0)
    min = np.min(success_rate_seeds, axis=0)
    plt.plot(mean_success_rate)
    plt.fill_between(
        range(len(mean_success_rate)),
        mean_success_rate - std_success_rate,
        mean_success_rate + std_success_rate,
        alpha=0.2,
    )
    plt.plot(max, linestyle="dashed")
    plt.plot(min, linestyle="dashed")
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate")
    plt.title(title)
    plt.show()


def success_rate_plots(setups, upperbound=100, step=1, title=""):
    """
    Computes the success rate of the policy over time for each setup in setups
    and plots the results

    setups: list of tuples (env, pi_track, label)
    upperbound: int, upperbound of the plot
    title: str, title of the plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # find the longest pi_track and pad the others with the last policy
    max_len = max([len(pi_track) for env, pi_track, label in setups])
    for i, (env, pi_track, label) in enumerate(setups):
        if len(pi_track) < max_len:
            last_policy = pi_track[-1]
            for _ in range(max_len - len(pi_track)):
                pi_track.append(last_policy)
        setups[i] = (env, pi_track, label)

    for env, pi_track, label in setups:

        success_rate = []
        mean_run_length = []
        std_run_length = []

        bound = min(max_len, upperbound)

        for i in tqdm(range(1, bound, step)):
            if len(pi_track) == i:
                break
            episode_rewards, run_length,_ = TestEnv.test_env(
                env=env, n_iters=100, pi=pi_track[i]
            )
            episode_rewards = np.array(episode_rewards)
            # count number of 1s in episode_rewards
            if env.custom_rewards is not None:
                success_count = np.count_nonzero(episode_rewards == (env.custom_rewards[b'G'] - env.step_penalty))
            else:
                success_count = np.count_nonzero(episode_rewards == 1)
            success_rate.append(success_count)
            mean_run_length.append(np.mean(run_length))
            std_run_length.append(np.std(run_length))
        ax[0].plot(range(1, bound, step), success_rate, label=label)
        ax[1].plot(range(1, bound, step), mean_run_length, label=label)
        ax[1].fill_between(
            range(1, bound, step),
            np.array(mean_run_length) - np.array(std_run_length),
            np.array(mean_run_length) + np.array(std_run_length),
            alpha=0.2,
        )
    # ax[0].set_title(title)
    ax[0].set_xlabel("Iterations", fontsize=14)
    ax[0].set_ylabel("Success Rate (%)", fontsize=14)
    ax[0].legend(fontsize=14)
    ax[1].set_xlabel("Iterations", fontsize=14)
    ax[1].set_ylabel("Mean Steps per Episode", fontsize=14)
    ax[1].legend(fontsize=14)
    plt.savefig(
        f"figures/frozen_lake/{title}_success_rate_mean_run_duration.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def precompute_mutli_seed_success_rates(env, pi_tracks, upperbound=100, step=1):
    """
    Computes the success rate of the policy over time for each run in pi_tracks
    and plots the results

    env: gym environment
    pi_tracks: list of pi_tracks
    upperbound: int, upperbound of the plot
    title: str, title of the plot
    """
    # find the longest pi_track and pad the others with the last policy
    max_len = max([len(pi_track) for pi_track in pi_tracks])
    for i, pi_track in enumerate(pi_tracks):
        if len(pi_track) < max_len:
            last_policy = pi_track[-1]
            for _ in range(max_len - len(pi_track)):
                pi_track.append(last_policy)
        pi_tracks[i] = pi_track

    success_rate_mean = []
    success_rate_std = []
    mean_run_length = []
    std_run_length = []

    bound = min(max_len, upperbound)

    for i in tqdm(range(1, bound, step)):
        current_success_rates = []
        current_mean_run_length = []
        for pi_track in pi_tracks:
            if len(pi_track) == i:
                break
            episode_rewards, run_length = TestEnv.test_env(
                env=env, n_iters=100, pi=pi_track[i]
            )
            episode_rewards = np.array(episode_rewards)
            # count number of 1s in episode_rewards
            if env.custom_rewards is not None:
                success_count = np.count_nonzero(
                    episode_rewards == env.custom_rewards[b"G"]
                )
            else:
                success_count = np.count_nonzero(episode_rewards == 1)
            current_success_rates.append(success_count / 100)
            current_mean_run_length.append(np.mean(run_length))

        success_rate_mean.append(np.mean(current_success_rates))
        success_rate_std.append(np.std(current_success_rates) / 3)
        mean_run_length.append(np.mean(current_mean_run_length))
        std_run_length.append(np.std(current_mean_run_length) / 3)

    return (success_rate_mean, success_rate_std, mean_run_length, std_run_length)


def one_shot_eval(env, pi,n_iters=100):
    episode_rewards, run_length, endings = TestEnv.test_env(env=env, n_iters=n_iters, pi=pi)
    endings = np.array(endings)
    # num_ones = np.count_nonzero(episode_rewards == env.custom_rewards[b"G"])
    # num_zeros = np.count_nonzero(episode_rewards == env.custom_rewards[b"F"])
    # num_neg_ones = np.count_nonzero(episode_rewards == env.custom_rewards[b"H"])
    num_ones = np.count_nonzero(endings == "G")
    num_zeros = np.count_nonzero(endings == "T")
    num_neg_ones = np.count_nonzero(endings == "H")
    print(f"Success Rate: {(num_ones / n_iters * 100):1.1f} %")
    print(f"Out of moves: {(num_zeros / n_iters * 100):1.1f} %")
    print(f"Fell in hole: {(num_neg_ones / n_iters * 100):1.1f} %")
    print("Mean run length: ", np.mean(run_length))
    print(f"Mean reward: {np.mean(episode_rewards)}")



def grid_results_to_df(g):
    df = pd.DataFrame(columns=['gamma', 'init_alpha', 'min_alpha', 'alpha_decay_ratio', 'init_epsilon', 'min_epsilon', 'epsilon_decay_ratio', 'reward', 'seed', 'endings', 'pi_track', 'Q_track'])
    for key in g:
        new_row = {}
        new_row['gamma'] = key[0]
        new_row['init_alpha'] = key[1]
        new_row['min_alpha'] = key[2]
        new_row['alpha_decay_ratio'] = key[3]
        new_row['init_epsilon'] = key[4]
        new_row['min_epsilon'] = key[5]
        new_row['epsilon_decay_ratio'] = key[6]
        new_row['reward'] = g[key]['mean_reward']
        new_row['seed'] = key[7]
        new_row['endings'] = [g[key]['endings']]
        new_row['pi_track'] = [g[key]['pi_track']]
        new_row['Q_track'] = [g[key]['Q_track']]
        df2 = pd.DataFrame(new_row, index=[0])
        df = pd.concat([df, df2], ignore_index=True)
    return df






###################### Q-LEARNING ######################

from pathos.multiprocessing import ProcessingPool
from functools import partial

def worker(seed, env, kwargs):
    print(f"Running Q-learning for seed {seed}")
    kwargs["seed"] = seed
    Q, V, pi, Q_track, pi_track, steps_per_episode = RL(env).q_learning(**kwargs)
    return {
        "Q": Q,
        "V": V,
        "pi": pi,
        "Q_track": Q_track,
        "pi_track": pi_track,
        "steps_per_episode": steps_per_episode,
    }

def multi_seed_q_learning(env, n_seeds, **kwargs):
    results = {}
    with ProcessingPool() as pool:
        worker_partial = partial(worker, env=env, kwargs=kwargs)
        results_list = pool.map(worker_partial, range(n_seeds))
    
    for seed, result in enumerate(results_list):
        results[seed] = result
    
    return results





def get_steps_per_episode_from_multi_seed(results):
    steps_per_episode = []
    for seed in results:
        steps_per_episode.append(results[seed]['steps_per_episode'])
    return steps_per_episode


def get_mean_std_steps(steps_per_episode):
    mean_steps = np.mean(steps_per_episode, axis=0)
    std_steps = np.std(steps_per_episode, axis=0)
    return mean_steps, std_steps

def eval_multi_seed(env, results, n_iters=100, convert_state_obs=lambda state : state):
    success_rate = []
    for seed in results:
        _, steps, _ = TestEnv.test_env(env=env, n_iters=n_iters, pi=results[seed]['pi'], convert_state_obs=convert_state_obs)
        mean_steps_reward = np.mean(steps)
        success_rate.append(mean_steps_reward)
    return np.mean(success_rate), np.std(success_rate)