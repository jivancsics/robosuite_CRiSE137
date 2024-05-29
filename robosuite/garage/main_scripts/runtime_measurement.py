"""File for plotting measuring and plotting runtime per-environment-step differences between the Sawyer, default IIWA
and costum, Robolab-imitating IIWA 14 robot with fixed linear axis"""

import robosuite as suite
from time import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib

# Configure .pgf LaTex export
matplotlib.use("pgf")
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    # 'font.size': 8.0,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'figure.figsize': (5.8476, 5.5),
    'figure.titlesize': 'medium',
    'axes.titlesize': 'medium',
})


robots = ["Sawyer", "IIWA", "IIWA14_extended_nolinear"]
envs = ["Lift", "Stack", "NutAssembly", "NutAssemblySquare", "NutAssemblyRound", "PickPlaceMilk",
        "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan", "Door"]
name_of_envs = ["LiftBlock", "StackBlocks", "NutAssemblyMixed", "NutAssemblySquare", "NutAssemblyRound",
                "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan", "OpenDoor"]
name_of_robots = ["Sawyer", "IIWA", "IIWA14Ext"]


class RobosuiteEnv:
    def __init__(self, env_name, robot):
        self.env_name = env_name
        self.robot = robot
        self.horizon = 500

        self.env = suite.make(
            env_name=self.env_name,
            robots=self.robot,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            hard_reset=False,
        )

    def __call__(self):
        per_step_env_runtime = []

        obs = self.env.reset()

        for steps in tqdm(range(self.horizon)):
            action = np.random.randn(self.env.robots[0].dof)
            start = time()
            _ = self.env.step(action)
            end = time()
            per_step_env_runtime.append(end - start)

        self.env.close()
        return per_step_env_runtime


def runtime_measurement_random_policy():

    print("Simple per-step environment runtime measurement with the Rethink Robotics Sawyer, the default Robosuite "
          "Kuka IIWA and the TU Vienna Robolab Kuka IIWA14 with fixed position on the linear axis")
    print("Based on a random-sample policy")
    print("\033[92m {}\033[00m".format("Starting to measure environment step times in all Meta 7 test and train tasks"))
    print("\033[92m {}\033[00m".format("Iterating over all robots"))

    mean_runtimes_per_env = np.zeros((len(robots), len(envs)))  # three robots and ten environments in total

    for i, robot in enumerate(robots):
        for j, env in enumerate(envs):
            current_env = RobosuiteEnv(env, robot)
            mean_runtimes_per_env[i, j] = np.mean(current_env())

    return mean_runtimes_per_env


if __name__ == "__main__":
    all_runtimes_per_env = runtime_measurement_random_policy()
    all_runtimes_per_env *= 1000

    figure, axis = plt.subplots()
    figure.set_figheight(3.5)
    figure.set_figwidth(5.8476)
    bar_y_pos = np.arange(len(robots))
    plt.barh(bar_y_pos, np.mean(all_runtimes_per_env, axis=1), align='center', alpha=0.7)
    plt.yticks(bar_y_pos, name_of_robots)
    plt.title("Mean per-environment-step runtime over all Meta 7 train/test tasks", fontsize=13)
    plt.xlabel("Mean runtime (ms)", fontsize=8)
    plt.setp(axis.get_yticklabels(), fontsize=8)
    plt.savefig('runtime_measurements_average.pgf')

    figure, axis = plt.subplots()
    figure.set_figheight(5.8)
    figure.set_figwidth(5.8476)
    index = np.arange(len(envs))
    width_single_bar = 0.15
    bars_sawyer = plt.bar(index, all_runtimes_per_env[0, :], width_single_bar, color="r", label=name_of_robots[0])
    bars_iiwa = plt.bar(index + width_single_bar, all_runtimes_per_env[1, :], width_single_bar,
                        color="g", label=name_of_robots[1])
    bars_iiwa14 = plt.bar(index + 2 * width_single_bar, all_runtimes_per_env[2, :], width_single_bar,
                          color="y", label=name_of_robots[2])
    # plt.xlabel("Meta 7 task environments", fontsize=8)
    plt.ylabel("Mean runtime (ms)", fontsize=8)
    plt.title("Mean per-environment-step runtime of all Meta 7 train/test tasks", fontsize=13)
    plt.xticks(index + width_single_bar, envs)
    plt.setp(axis.get_xticklabels(), rotation='vertical', fontsize=6)
    plt.legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_measurements_AllMeta7Tasks.pgf')
    # plt.show()
