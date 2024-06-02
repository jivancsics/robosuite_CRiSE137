"""
Script for measuring and plotting mean runtime per-environment-step when utilising the Sawyer, the default IIWA7
and the TU Vienna ACIN Robo Lab-imitating IIWA14 robot on fixed linear axes"""

import robosuite as suite
from time import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Configure .pgf LaTex export
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


robots = ["Sawyer", "IIWA7", "IIWA14"]
robots_envs = ["Sawyer", "IIWA", "IIWA14_extended_nolinear"]
envs = ["Lift", "Stack", "NutAssembly", "NutAssemblySquare", "NutAssemblyRound", "PickPlaceMilk",
        "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan", "Door"]
name_of_envs = ["LiftBlock", "StackBlocks", "NutAssemblyMixed", "NutAssemblySquare", "NutAssemblyRound",
                "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan", "OpenDoor"]


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

        for steps in range(self.horizon):
            action = np.random.randn(self.env.robots[0].dof)
            start = time()
            _ = self.env.step(action)
            end = time()
            per_step_env_runtime.append(end - start)

        self.env.close()
        return per_step_env_runtime


def runtime_measurement_random_policy():

    print("Simple per-step environment runtime measurement with the Rethink Robotics Sawyer, the default Robosuite "
          "Kuka IIWA and the TU Vienna Robolab Kuka IIWA14 with fixed position on the linear axes")
    print("Based on a random-sample policy")
    print("\033[92m {}\033[00m".format("Starting to measure environment step times in all CRiSE 7 test and train tasks"))
    print("\033[92m {}\033[00m".format("Iterating over all robots"))

    mean_runtimes_per_env = np.zeros((len(robots), len(envs)))  # three robots, ten environments in total

    for i, robot in enumerate(robots_envs):
        for j, env in enumerate(envs):
            current_env = RobosuiteEnv(env, robot)
            mean_runtimes_per_env[i, j] = np.mean(current_env())

    return mean_runtimes_per_env


if __name__ == "__main__":
    times_buffer = []

    for _ in range(3):
        times = runtime_measurement_random_policy()
        times_buffer.append(times)

    all_runtimes_per_env = np.ones([3, 10])
    for robot in range(3):
        for env in range(10):
            all_runtimes_per_env[robot, env] = np.mean([times_buffer[0][robot][env], times_buffer[1][robot][env],
                                                        times_buffer[2][robot][env]])

    all_runtimes_per_env *= 1000

    bar_y_pos = np.arange(len(robots))
    plt.barh(bar_y_pos, np.mean(all_runtimes_per_env, axis=1), align='center', alpha=0.5)
    plt.yticks(bar_y_pos, robots)
    plt.title("Per-environment-step runtime averaged over all CRiSE 7 tasks", fontsize=12)
    plt.xlabel("Mean runtime (ms)", fontsize=10)
    plt.savefig('runtime_measurements_average.pgf')

    figure2, axis2 = plt.subplots()
    index = np.arange(len(envs))
    width_single_bar = 0.2
    bars_sawyer = plt.bar(index, all_runtimes_per_env[0, :], width_single_bar, color="r", label=robots[0])
    bars_iiwa = plt.bar(index + width_single_bar, all_runtimes_per_env[1, :], width_single_bar,
                        color="g", label=robots[1])
    bars_iiwa14 = plt.bar(index + 2 * width_single_bar, all_runtimes_per_env[2, :], width_single_bar,
                          color="y", label=robots[2])
    plt.xlabel("CRiSE 7 task environments", fontsize=10)
    plt.ylabel("Mean runtime (ms)", fontsize=10)
    plt.title("Mean per-environment-step runtime of all CRiSE 7 meta-train/test tasks", fontsize=12)
    plt.xticks(index + width_single_bar, name_of_envs)
    axis2.set_xticklabels(axis2.get_xticklabels(), rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_measurements_AllCRiSE7Tasks.pgf')
    # plt.show()
