"""Script for plotting all the experimental results.
The directories must be modified to the individual directory structure.
Proposed structure:

  CRiSE137
    ├──Experiment_Data (create this folder and store the experiment data in the respective sub-folders)
    │    ├──Robosuite_IIWA14_CRiSE1_LiftBlock
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_NutAssemblyMixed
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_NutAssemblyRound
    │    │    ├──crise1_maml_trpo
    │    │    ├──crise1_rl2_ppo
    │    │    └──crise1_rl2_ppo_OSCPOSE
    │    ├──Robosuite_IIWA14_CRiSE1_NutAssemblySquare
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_OpenDoor
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_PickPlaceBread
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_PickPlaceCan
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_PickPLaceCereal
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_PickPlaceMilk
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE1_StackBlocks
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE3
    │    │    ├──crise3_maml_trpo
    │    │    └──crise3_rl2_ppo
    │    ├──Robosuite_IIWA14_CRiSE7
    │    │    ├──crise7_maml_trpo
    │    │    └──crise7_rl2_ppo
    │    ├──Robosuite_Sawyer_CRiSE1_LiftBlock
    │    │    ├──crise1_maml_trpo
    │    │    └──crise1_rl2_ppo
    │    └──Robosuite_Sawyer_CRiSE7
    │        ├──crise7_maml_trpo
    │        └──crise7_rl2_ppo
    │
    ├──experiment_launchers
    │    ├──IIWA14_extended_nolinear
    │    └──Sawyer
    │
    ├──Robosuite_IIWA14_extended_nolinear
    │
    └──Robosuite_Sawyer

"""

import csv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Configure .pgf LaTex export for storage of figures
matplotlib.use("pgf")
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 6.0,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.figsize': (5.8476, 6),
    'figure.titlesize': 'medium',
    'axes.titlesize': 'xx-small',
    'legend.fontsize': 'x-small'
})

train_envs = ["LiftBlock", "NutAssemblyRound", "OpenDoor", "NutAssemblyMixed", "PickPlaceMilk",
              "PickPlaceBread", "PickPlaceCereal"]
test_envs = ["StackBlocks", "PickPlaceCan", "NutAssemblySquare"]
all_envs = train_envs + test_envs


def plot_all():
    # Rethink Robotics Sawyer in the LiftBlock task (CRiSE 1) with MAML_TRPO and RL2_PPO
    # --------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_CRiSE1_Blocklifting/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]
    average_numepisodes_mamltrpo = data_rows[:, header.index('Average/NumEpisodes')]

    # Get meta test average return, success rate and standard return MAML_TRPO
    metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')] * 100.0
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]

    # Compute standard deviation for success rates
    average_stdsuccess_mamltrpo = np.sqrt((average_successrate_mamltrpo / 100) *
                                          (1 - (average_successrate_mamltrpo / 100))) * 100

    # Get the number of the corresponding environment steps MAML_TRPO
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_CRiSE1_Blocklifting/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)
    average_numepisodes_rl2ppo = data_rows[:, header.index('Average/NumEpisodes')]

    # Compute standard deviation for success rates (binomial distribution --> Bernoulli experiment)
    average_stdsuccess_rl2ppo = np.sqrt(
        (average_successrate_rl2ppo / 100) * (1 - (average_successrate_rl2ppo / 100))) * 100

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
                                       .astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                     .astype(float))

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    lowerbound_return_mamltrpo = average_return_mamltrpo - average_stdreturn_mamltrpo
    upperbound_return_mamltrpo = average_return_mamltrpo + average_stdreturn_mamltrpo
    lowerbound_return_rl2ppo = average_return_rl2ppo - average_stdreturn_rl2ppo
    upperbound_return_rl2ppo = average_return_rl2ppo + average_stdreturn_rl2ppo

    lowerbound_return_mamltrpo = np.where(lowerbound_return_mamltrpo >= 0, lowerbound_return_mamltrpo, 0)
    upperbound_return_mamltrpo = np.where(upperbound_return_mamltrpo <= 500, upperbound_return_mamltrpo, 500)
    lowerbound_return_rl2ppo = np.where(lowerbound_return_rl2ppo >= 0, lowerbound_return_rl2ppo, 0)
    upperbound_return_rl2ppo = np.where(upperbound_return_rl2ppo <= 500, upperbound_return_rl2ppo, 500)

    lowerbound_success_mamltrpo = average_successrate_mamltrpo - average_stdsuccess_mamltrpo
    upperbound_success_mamltrpo = average_successrate_mamltrpo + average_stdsuccess_mamltrpo
    lowerbound_success_rl2ppo = average_successrate_rl2ppo - average_stdsuccess_rl2ppo
    upperbound_success_rl2ppo = average_successrate_rl2ppo + average_stdsuccess_rl2ppo

    lowerbound_success_mamltrpo = np.where(lowerbound_success_mamltrpo >= 0, lowerbound_success_mamltrpo, 0)
    upperbound_success_mamltrpo = np.where(upperbound_success_mamltrpo <= 100, upperbound_success_mamltrpo, 100)
    lowerbound_success_rl2ppo = np.where(lowerbound_success_rl2ppo >= 0, lowerbound_success_rl2ppo, 0)
    upperbound_success_rl2ppo = np.where(upperbound_success_rl2ppo <= 100, upperbound_success_rl2ppo, 100)

    # Plot everything
    fig, axis = plt.subplots(2, 1)
    fig_2, axis_2 = plt.subplots(2, 1)

    axis[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis[0].fill_between(total_env_steps_mamltrpo, lowerbound_return_mamltrpo,
                         upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axis[0].fill_between(total_env_steps_rl2ppo, lowerbound_return_rl2ppo,
                         upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axis[0].set_ylim([-5, 520])
    legend = axis[0].legend()
    legend.get_frame().set_facecolor('white')

    axis[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis[1].fill_between(total_env_steps_mamltrpo, lowerbound_success_mamltrpo,
                         upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axis[1].fill_between(total_env_steps_rl2ppo, lowerbound_success_rl2ppo,
                         upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axis[1].set_ylim([-5, 105])
    legend = axis[1].legend()
    legend.get_frame().set_facecolor('white')

    axis_2[0].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis_2[0].set_ylim([-5, 520])
    legend = axis_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis_2[1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, '+', color='red',
                   label='MAML-TRPO')
    axis_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, 'x', color='green',
                   label='RL2-PPO')
    axis_2[1].set_ylim([-5, 105])
    legend = axis_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis[0], ylabel='Average Return')
    plt.setp(axis[1], ylabel='Success Rate (%)')
    plt.setp(axis_2[0], ylabel='Return')
    plt.setp(axis_2[1], ylabel='Success Rate (%)')
    plt.setp(axis[1], xlabel='Total Environment Steps')
    plt.setp(axis_2[1], xlabel='Total Environment Steps')
    fig.suptitle('CRiSE 1 - Lift Block with Sawyer (Meta-Training)', fontsize=14)
    fig_2.suptitle('CRiSE 1 - Lift Block with Sawyer (Meta-Test)', fontsize=14)
    fig.savefig('CRiSE1_Sawyer_LiftBlock_Train.pgf')
    fig_2.savefig('CRiSE1_Sawyer_LiftBlock_Test.pgf')


    # Kuka IIWA14 with no linear axes in the LiftBlock task (CRiSE 1) with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_LiftBlock/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

    # Get meta test average return, success rate and standard return MAML_TRPO
    metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')] * 100.0
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]

    # Compute standard deviation for success rates
    average_stdsuccess_mamltrpo = np.sqrt(
        (average_successrate_mamltrpo / 100) * (1 - (average_successrate_mamltrpo / 100))) * 100

    # Get the number of the corresponding environment steps MAML_TRPO
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_LiftBlock/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
                                       .astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                     .astype(float))

    # Compute standard deviation for success rates
    average_stdsuccess_rl2ppo = np.sqrt(
        (average_successrate_rl2ppo / 100) * (1 - (average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    lowerbound_return_mamltrpo = average_return_mamltrpo - average_stdreturn_mamltrpo
    upperbound_return_mamltrpo = average_return_mamltrpo + average_stdreturn_mamltrpo
    lowerbound_return_rl2ppo = average_return_rl2ppo - average_stdreturn_rl2ppo
    upperbound_return_rl2ppo = average_return_rl2ppo + average_stdreturn_rl2ppo

    lowerbound_return_mamltrpo = np.where(lowerbound_return_mamltrpo >= 0, lowerbound_return_mamltrpo, 0)
    upperbound_return_mamltrpo = np.where(upperbound_return_mamltrpo <= 500, upperbound_return_mamltrpo, 500)
    lowerbound_return_rl2ppo = np.where(lowerbound_return_rl2ppo >= 0, lowerbound_return_rl2ppo, 0)
    upperbound_return_rl2ppo = np.where(upperbound_return_rl2ppo <= 500, upperbound_return_rl2ppo, 500)

    lowerbound_success_mamltrpo = average_successrate_mamltrpo - average_stdsuccess_mamltrpo
    upperbound_success_mamltrpo = average_successrate_mamltrpo + average_stdsuccess_mamltrpo
    lowerbound_success_rl2ppo = average_successrate_rl2ppo - average_stdsuccess_rl2ppo
    upperbound_success_rl2ppo = average_successrate_rl2ppo + average_stdsuccess_rl2ppo

    lowerbound_success_mamltrpo = np.where(lowerbound_success_mamltrpo >= 0, lowerbound_success_mamltrpo, 0)
    upperbound_success_mamltrpo = np.where(upperbound_success_mamltrpo <= 100, upperbound_success_mamltrpo, 100)
    lowerbound_success_rl2ppo = np.where(lowerbound_success_rl2ppo >= 0, lowerbound_success_rl2ppo, 0)
    upperbound_success_rl2ppo = np.where(upperbound_success_rl2ppo <= 100, upperbound_success_rl2ppo, 100)

    # Plot everything
    fig3, axis3 = plt.subplots(2, 1)
    fig3_2, axis3_2 = plt.subplots(2, 1)

    axis3[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis3[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis3[0].fill_between(total_env_steps_mamltrpo, lowerbound_return_mamltrpo,
                          upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axis3[0].fill_between(total_env_steps_rl2ppo, lowerbound_return_rl2ppo,
                          upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axis3[0].set_ylim([-5, 520])
    legend = axis3[0].legend()
    legend.get_frame().set_facecolor('white')

    axis3[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis3[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis3[1].fill_between(total_env_steps_mamltrpo, lowerbound_success_mamltrpo,
                          upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axis3[1].fill_between(total_env_steps_rl2ppo, lowerbound_success_rl2ppo,
                          upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axis3[1].set_ylim([-5, 105])
    legend = axis3[1].legend()
    legend.get_frame().set_facecolor('white')

    axis3_2[0].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis3_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis3_2[0].set_ylim([-5, 520])
    legend = axis3_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis3_2[1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, '+', color='red',
                    label='MAML-TRPO')
    axis3_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, 'x', color='green',
                    label='RL2-PPO')
    axis3_2[1].set_ylim([-5, 105])
    legend = axis3_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis3[0], ylabel='Average Return')
    plt.setp(axis3[1], ylabel='Success Rate (%)')
    plt.setp(axis3_2[0], ylabel='Return')
    plt.setp(axis3_2[1], ylabel='Success Rate (%)')
    plt.setp(axis3[1], xlabel='Total Environment Steps')
    plt.setp(axis3_2[1], xlabel='Total Environment Steps')
    fig3.suptitle('CRiSE 1 - Lift Block with IIWA14 (Meta-Training)', fontsize=14)
    fig3_2.suptitle('CRiSE 1 - Lift Block with IIWA14 (Meta-Test)', fontsize=14)
    fig3.savefig('CRiSE1_IIWA14_LiftBlock_Train.pgf')
    fig3_2.savefig('CRiSE1_IIWA14_LiftBlock_Test.pgf')

    # Kuka IIWA14 with no linear axes in ALL CRiSE 7 tasks (in CRiSE 1 MRL-context) with MAML_TRPO and RL2_PPO
    # --------------------------------------------------------------------------------------------------------

    # LiftBlock
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_LiftBlock/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    blocklift_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    blocklift_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    blocklift_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

    # Get meta test average return, success rate and standard return MAML_TRPO
    blocklift_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    blocklift_metatest_avg_return_mamltrpo = blocklift_metatest_avg_return_mamltrpo[0::5]
    blocklift_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')] * 100.0
    blocklift_metatest_avg_successrate_mamltrpo = blocklift_metatest_avg_successrate_mamltrpo[0::5]
    blocklift_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    blocklift_metatest_avg_stdreturn_mamltrpo = blocklift_metatest_avg_stdreturn_mamltrpo[0::5]

    # Get the number of the corresponding environment steps MAML_TRPO
    blocklift_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    blocklift_total_testenv_steps_mamltrpo = blocklift_total_env_steps_mamltrpo[0::5]

    # Compute standard deviation for success rates
    blocklift_average_stdsuccess_mamltrpo = np.sqrt(
        (blocklift_average_successrate_mamltrpo / 100) * (1 - (blocklift_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_LiftBlock/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    blocklift_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    blocklift_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    blocklift_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    blocklift_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    blocklift_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    blocklift_total_testenv_steps_rl2ppo = blocklift_total_env_steps_rl2ppo[np.where(
        blocklift_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    blocklift_metatest_avg_return_rl2ppo = blocklift_metatest_avg_return_rl2ppo[
        np.where(blocklift_metatest_avg_return_rl2ppo != '')].astype(float)
    blocklift_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    blocklift_metatest_avg_successrate_rl2ppo = (
            blocklift_metatest_avg_successrate_rl2ppo[np.where(blocklift_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    blocklift_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    blocklift_metatest_avg_stdreturn_rl2ppo = (
        blocklift_metatest_avg_stdreturn_rl2ppo[np.where(blocklift_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    blocklift_average_stdsuccess_rl2ppo = np.sqrt(
        (blocklift_average_successrate_rl2ppo / 100) * (1 - (blocklift_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    blocklift_lowerbound_return_mamltrpo = blocklift_average_return_mamltrpo - blocklift_average_stdreturn_mamltrpo
    blocklift_upperbound_return_mamltrpo = blocklift_average_return_mamltrpo + blocklift_average_stdreturn_mamltrpo
    blocklift_lowerbound_return_rl2ppo = blocklift_average_return_rl2ppo - blocklift_average_stdreturn_rl2ppo
    blocklift_upperbound_return_rl2ppo = blocklift_average_return_rl2ppo + blocklift_average_stdreturn_rl2ppo

    blocklift_lowerbound_return_mamltrpo = np.where(blocklift_lowerbound_return_mamltrpo >= 0,
                                                    blocklift_lowerbound_return_mamltrpo, 0)
    blocklift_upperbound_return_mamltrpo = np.where(blocklift_upperbound_return_mamltrpo <= 500,
                                                    blocklift_upperbound_return_mamltrpo, 500)
    blocklift_lowerbound_return_rl2ppo = np.where(blocklift_lowerbound_return_rl2ppo >= 0,
                                                  blocklift_lowerbound_return_rl2ppo, 0)
    blocklift_upperbound_return_rl2ppo = np.where(blocklift_upperbound_return_rl2ppo <= 500,
                                                  blocklift_upperbound_return_rl2ppo, 500)

    blocklift_lowerbound_success_mamltrpo = blocklift_average_successrate_mamltrpo - blocklift_average_stdsuccess_mamltrpo
    blocklift_upperbound_success_mamltrpo = blocklift_average_successrate_mamltrpo + blocklift_average_stdsuccess_mamltrpo
    blocklift_lowerbound_success_rl2ppo = blocklift_average_successrate_rl2ppo - blocklift_average_stdsuccess_rl2ppo
    blocklift_upperbound_success_rl2ppo = blocklift_average_successrate_rl2ppo + blocklift_average_stdsuccess_rl2ppo

    blocklift_lowerbound_success_mamltrpo = np.where(blocklift_lowerbound_success_mamltrpo >= 0,
                                                     blocklift_lowerbound_success_mamltrpo, 0)
    blocklift_upperbound_success_mamltrpo = np.where(blocklift_upperbound_success_mamltrpo <= 100,
                                                     blocklift_upperbound_success_mamltrpo, 100)
    blocklift_lowerbound_success_rl2ppo = np.where(blocklift_lowerbound_success_rl2ppo >= 0,
                                                   blocklift_lowerbound_success_rl2ppo, 0)
    blocklift_upperbound_success_rl2ppo = np.where(blocklift_upperbound_success_rl2ppo <= 100,
                                                   blocklift_upperbound_success_rl2ppo, 100)

    # NutAssemblyRound

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblyRound/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    naround_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    naround_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    naround_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    naround_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    naround_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    naround_total_testenv_steps_mamltrpo = naround_total_env_steps_mamltrpo[np.where(
        naround_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    naround_metatest_avg_return_mamltrpo = naround_metatest_avg_return_mamltrpo[
        np.where(naround_metatest_avg_return_mamltrpo != '')].astype(float)
    naround_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    naround_metatest_avg_successrate_mamltrpo = (
            naround_metatest_avg_successrate_mamltrpo[np.where(naround_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    naround_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    naround_metatest_avg_stdreturn_mamltrpo = (
        naround_metatest_avg_stdreturn_mamltrpo[np.where(naround_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    naround_average_stdsuccess_mamltrpo = np.sqrt(
        (naround_average_successrate_mamltrpo / 100) * (1 - (naround_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblyRound/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    naround_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    naround_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    naround_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    naround_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    naround_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    naround_total_testenv_steps_rl2ppo = naround_total_env_steps_rl2ppo[np.where(
        naround_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    naround_metatest_avg_return_rl2ppo = naround_metatest_avg_return_rl2ppo[
        np.where(naround_metatest_avg_return_rl2ppo != '')].astype(float)
    naround_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    naround_metatest_avg_successrate_rl2ppo = (
            naround_metatest_avg_successrate_rl2ppo[np.where(naround_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    naround_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    naround_metatest_avg_stdreturn_rl2ppo = (
        naround_metatest_avg_stdreturn_rl2ppo[np.where(naround_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    naround_average_stdsuccess_rl2ppo = np.sqrt(
        (naround_average_successrate_rl2ppo / 100) * (1 - (naround_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    naround_lowerbound_return_mamltrpo = naround_average_return_mamltrpo - naround_average_stdreturn_mamltrpo
    naround_upperbound_return_mamltrpo = naround_average_return_mamltrpo + naround_average_stdreturn_mamltrpo
    naround_lowerbound_return_rl2ppo = naround_average_return_rl2ppo - naround_average_stdreturn_rl2ppo
    naround_upperbound_return_rl2ppo = naround_average_return_rl2ppo + naround_average_stdreturn_rl2ppo

    naround_lowerbound_return_mamltrpo = np.where(naround_lowerbound_return_mamltrpo >= 0,
                                                  naround_lowerbound_return_mamltrpo, 0)
    naround_upperbound_return_mamltrpo = np.where(naround_upperbound_return_mamltrpo <= 500,
                                                  naround_upperbound_return_mamltrpo, 500)
    naround_lowerbound_return_rl2ppo = np.where(naround_lowerbound_return_rl2ppo >= 0,
                                                naround_lowerbound_return_rl2ppo, 0)
    naround_upperbound_return_rl2ppo = np.where(naround_upperbound_return_rl2ppo <= 500,
                                                naround_upperbound_return_rl2ppo, 500)

    naround_lowerbound_success_mamltrpo = naround_average_successrate_mamltrpo - naround_average_stdsuccess_mamltrpo
    naround_upperbound_success_mamltrpo = naround_average_successrate_mamltrpo + naround_average_stdsuccess_mamltrpo
    naround_lowerbound_success_rl2ppo = naround_average_successrate_rl2ppo - naround_average_stdsuccess_rl2ppo
    naround_upperbound_success_rl2ppo = naround_average_successrate_rl2ppo + naround_average_stdsuccess_rl2ppo

    naround_lowerbound_success_mamltrpo = np.where(naround_lowerbound_success_mamltrpo >= 0,
                                                   naround_lowerbound_success_mamltrpo, 0)
    naround_upperbound_success_mamltrpo = np.where(naround_upperbound_success_mamltrpo <= 100,
                                                   naround_upperbound_success_mamltrpo, 100)
    naround_lowerbound_success_rl2ppo = np.where(naround_lowerbound_success_rl2ppo >= 0,
                                                 naround_lowerbound_success_rl2ppo, 0)
    naround_upperbound_success_rl2ppo = np.where(naround_upperbound_success_rl2ppo <= 100,
                                                 naround_upperbound_success_rl2ppo, 100)

    # OpenDoor

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_OpenDoor/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    opendoor_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    opendoor_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    opendoor_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    opendoor_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    opendoor_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    opendoor_total_testenv_steps_mamltrpo = opendoor_total_env_steps_mamltrpo[np.where(
        opendoor_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    opendoor_metatest_avg_return_mamltrpo = opendoor_metatest_avg_return_mamltrpo[
        np.where(opendoor_metatest_avg_return_mamltrpo != '')].astype(float)
    opendoor_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    opendoor_metatest_avg_successrate_mamltrpo = (
            opendoor_metatest_avg_successrate_mamltrpo[np.where(opendoor_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    opendoor_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    opendoor_metatest_avg_stdreturn_mamltrpo = (
        opendoor_metatest_avg_stdreturn_mamltrpo[np.where(opendoor_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    opendoor_average_stdsuccess_mamltrpo = np.sqrt(
        (opendoor_average_successrate_mamltrpo / 100) * (1 - (opendoor_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_Door-Open/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    opendoor_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    opendoor_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    opendoor_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    opendoor_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    opendoor_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    opendoor_total_testenv_steps_rl2ppo = opendoor_total_env_steps_rl2ppo[np.where(
        opendoor_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    opendoor_metatest_avg_return_rl2ppo = opendoor_metatest_avg_return_rl2ppo[
        np.where(opendoor_metatest_avg_return_rl2ppo != '')].astype(float)
    opendoor_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    opendoor_metatest_avg_successrate_rl2ppo = (
            opendoor_metatest_avg_successrate_rl2ppo[np.where(opendoor_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    opendoor_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    opendoor_metatest_avg_stdreturn_rl2ppo = (
        opendoor_metatest_avg_stdreturn_rl2ppo[np.where(opendoor_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    opendoor_average_stdsuccess_rl2ppo = np.sqrt(
        (opendoor_average_successrate_rl2ppo / 100) * (1 - (opendoor_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    opendoor_lowerbound_return_mamltrpo = opendoor_average_return_mamltrpo - opendoor_average_stdreturn_mamltrpo
    opendoor_upperbound_return_mamltrpo = opendoor_average_return_mamltrpo + opendoor_average_stdreturn_mamltrpo
    opendoor_lowerbound_return_rl2ppo = opendoor_average_return_rl2ppo - opendoor_average_stdreturn_rl2ppo
    opendoor_upperbound_return_rl2ppo = opendoor_average_return_rl2ppo + opendoor_average_stdreturn_rl2ppo

    opendoor_lowerbound_return_mamltrpo = np.where(opendoor_lowerbound_return_mamltrpo >= 0,
                                                   opendoor_lowerbound_return_mamltrpo, 0)
    opendoor_upperbound_return_mamltrpo = np.where(opendoor_upperbound_return_mamltrpo <= 500,
                                                   opendoor_upperbound_return_mamltrpo, 500)
    opendoor_lowerbound_return_rl2ppo = np.where(opendoor_lowerbound_return_rl2ppo >= 0,
                                                 opendoor_lowerbound_return_rl2ppo, 0)
    opendoor_upperbound_return_rl2ppo = np.where(opendoor_upperbound_return_rl2ppo <= 500,
                                                 opendoor_upperbound_return_rl2ppo, 500)

    opendoor_lowerbound_success_mamltrpo = opendoor_average_successrate_mamltrpo - opendoor_average_stdsuccess_mamltrpo
    opendoor_upperbound_success_mamltrpo = opendoor_average_successrate_mamltrpo + opendoor_average_stdsuccess_mamltrpo
    opendoor_lowerbound_success_rl2ppo = opendoor_average_successrate_rl2ppo - opendoor_average_stdsuccess_rl2ppo
    opendoor_upperbound_success_rl2ppo = opendoor_average_successrate_rl2ppo + opendoor_average_stdsuccess_rl2ppo

    opendoor_lowerbound_success_mamltrpo = np.where(opendoor_lowerbound_success_mamltrpo >= 0,
                                                    opendoor_lowerbound_success_mamltrpo, 0)
    opendoor_upperbound_success_mamltrpo = np.where(opendoor_upperbound_success_mamltrpo <= 100,
                                                    opendoor_upperbound_success_mamltrpo, 100)
    opendoor_lowerbound_success_rl2ppo = np.where(opendoor_lowerbound_success_rl2ppo >= 0,
                                                  opendoor_lowerbound_success_rl2ppo, 0)
    opendoor_upperbound_success_rl2ppo = np.where(opendoor_upperbound_success_rl2ppo <= 100,
                                                  opendoor_upperbound_success_rl2ppo, 100)

    # NutAssemblyMixed

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblyMixed/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    namixed_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    namixed_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    namixed_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    namixed_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    namixed_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    namixed_total_testenv_steps_mamltrpo = namixed_total_env_steps_mamltrpo[np.where(
        namixed_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    namixed_metatest_avg_return_mamltrpo = namixed_metatest_avg_return_mamltrpo[
        np.where(namixed_metatest_avg_return_mamltrpo != '')].astype(float)
    namixed_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    namixed_metatest_avg_successrate_mamltrpo = (
            namixed_metatest_avg_successrate_mamltrpo[np.where(namixed_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    namixed_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    namixed_metatest_avg_stdreturn_mamltrpo = (
        namixed_metatest_avg_stdreturn_mamltrpo[np.where(namixed_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    namixed_average_stdsuccess_mamltrpo = np.sqrt(
        (namixed_average_successrate_mamltrpo / 100) * (1 - (namixed_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblyMixed/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    namixed_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    namixed_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    namixed_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    namixed_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    namixed_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    namixed_total_testenv_steps_rl2ppo = namixed_total_env_steps_rl2ppo[np.where(
        namixed_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    namixed_metatest_avg_return_rl2ppo = namixed_metatest_avg_return_rl2ppo[
        np.where(namixed_metatest_avg_return_rl2ppo != '')].astype(float)
    namixed_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    namixed_metatest_avg_successrate_rl2ppo = (
            namixed_metatest_avg_successrate_rl2ppo[np.where(namixed_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    namixed_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    namixed_metatest_avg_stdreturn_rl2ppo = (
        namixed_metatest_avg_stdreturn_rl2ppo[np.where(namixed_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    namixed_average_stdsuccess_rl2ppo = np.sqrt(
        (namixed_average_successrate_rl2ppo / 100) * (1 - (namixed_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    namixed_lowerbound_return_mamltrpo = namixed_average_return_mamltrpo - namixed_average_stdreturn_mamltrpo
    namixed_upperbound_return_mamltrpo = namixed_average_return_mamltrpo + namixed_average_stdreturn_mamltrpo
    namixed_lowerbound_return_rl2ppo = namixed_average_return_rl2ppo - namixed_average_stdreturn_rl2ppo
    namixed_upperbound_return_rl2ppo = namixed_average_return_rl2ppo + namixed_average_stdreturn_rl2ppo

    namixed_lowerbound_return_mamltrpo = np.where(namixed_lowerbound_return_mamltrpo >= 0,
                                                  namixed_lowerbound_return_mamltrpo, 0)
    namixed_upperbound_return_mamltrpo = np.where(namixed_upperbound_return_mamltrpo <= 500,
                                                  namixed_upperbound_return_mamltrpo, 500)
    namixed_lowerbound_return_rl2ppo = np.where(namixed_lowerbound_return_rl2ppo >= 0,
                                                namixed_lowerbound_return_rl2ppo, 0)
    namixed_upperbound_return_rl2ppo = np.where(namixed_upperbound_return_rl2ppo <= 500,
                                                namixed_upperbound_return_rl2ppo, 500)

    namixed_lowerbound_success_mamltrpo = namixed_average_successrate_mamltrpo - namixed_average_stdsuccess_mamltrpo
    namixed_upperbound_success_mamltrpo = namixed_average_successrate_mamltrpo + namixed_average_stdsuccess_mamltrpo
    namixed_lowerbound_success_rl2ppo = namixed_average_successrate_rl2ppo - namixed_average_stdsuccess_rl2ppo
    namixed_upperbound_success_rl2ppo = namixed_average_successrate_rl2ppo + namixed_average_stdsuccess_rl2ppo

    namixed_lowerbound_success_mamltrpo = np.where(namixed_lowerbound_success_mamltrpo >= 0,
                                                   namixed_lowerbound_success_mamltrpo, 0)
    namixed_upperbound_success_mamltrpo = np.where(namixed_upperbound_success_mamltrpo <= 100,
                                                   namixed_upperbound_success_mamltrpo, 100)
    namixed_lowerbound_success_rl2ppo = np.where(namixed_lowerbound_success_rl2ppo >= 0,
                                                 namixed_lowerbound_success_rl2ppo, 0)
    namixed_upperbound_success_rl2ppo = np.where(namixed_upperbound_success_rl2ppo <= 100,
                                                 namixed_upperbound_success_rl2ppo, 100)

    # PickPlaceMilk

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceMilk/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    ppmilk_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppmilk_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppmilk_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    ppmilk_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    ppmilk_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppmilk_total_testenv_steps_mamltrpo = ppmilk_total_env_steps_mamltrpo[np.where(
        ppmilk_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppmilk_metatest_avg_return_mamltrpo = ppmilk_metatest_avg_return_mamltrpo[
        np.where(ppmilk_metatest_avg_return_mamltrpo != '')].astype(float)
    ppmilk_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppmilk_metatest_avg_successrate_mamltrpo = (
            ppmilk_metatest_avg_successrate_mamltrpo[np.where(ppmilk_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    ppmilk_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppmilk_metatest_avg_stdreturn_mamltrpo = (
        ppmilk_metatest_avg_stdreturn_mamltrpo[np.where(ppmilk_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppmilk_average_stdsuccess_mamltrpo = np.sqrt(
        (ppmilk_average_successrate_mamltrpo / 100) * (1 - (ppmilk_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceMilk/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    ppmilk_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppmilk_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppmilk_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    ppmilk_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppmilk_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    ppmilk_total_testenv_steps_rl2ppo = ppmilk_total_env_steps_rl2ppo[np.where(
        ppmilk_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppmilk_metatest_avg_return_rl2ppo = ppmilk_metatest_avg_return_rl2ppo[
        np.where(ppmilk_metatest_avg_return_rl2ppo != '')].astype(float)
    ppmilk_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppmilk_metatest_avg_successrate_rl2ppo = (
            ppmilk_metatest_avg_successrate_rl2ppo[np.where(ppmilk_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    ppmilk_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppmilk_metatest_avg_stdreturn_rl2ppo = (
        ppmilk_metatest_avg_stdreturn_rl2ppo[np.where(ppmilk_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppmilk_average_stdsuccess_rl2ppo = np.sqrt(
        (ppmilk_average_successrate_rl2ppo / 100) * (1 - (ppmilk_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    ppmilk_lowerbound_return_mamltrpo = ppmilk_average_return_mamltrpo - ppmilk_average_stdreturn_mamltrpo
    ppmilk_upperbound_return_mamltrpo = ppmilk_average_return_mamltrpo + ppmilk_average_stdreturn_mamltrpo
    ppmilk_lowerbound_return_rl2ppo = ppmilk_average_return_rl2ppo - ppmilk_average_stdreturn_rl2ppo
    ppmilk_upperbound_return_rl2ppo = ppmilk_average_return_rl2ppo + ppmilk_average_stdreturn_rl2ppo

    ppmilk_lowerbound_return_mamltrpo = np.where(ppmilk_lowerbound_return_mamltrpo >= 0,
                                                 ppmilk_lowerbound_return_mamltrpo, 0)
    ppmilk_upperbound_return_mamltrpo = np.where(ppmilk_upperbound_return_mamltrpo <= 500,
                                                 ppmilk_upperbound_return_mamltrpo, 500)
    ppmilk_lowerbound_return_rl2ppo = np.where(ppmilk_lowerbound_return_rl2ppo >= 0,
                                               ppmilk_lowerbound_return_rl2ppo, 0)
    ppmilk_upperbound_return_rl2ppo = np.where(ppmilk_upperbound_return_rl2ppo <= 500,
                                               ppmilk_upperbound_return_rl2ppo, 500)

    ppmilk_lowerbound_success_mamltrpo = ppmilk_average_successrate_mamltrpo - ppmilk_average_stdsuccess_mamltrpo
    ppmilk_upperbound_success_mamltrpo = ppmilk_average_successrate_mamltrpo + ppmilk_average_stdsuccess_mamltrpo
    ppmilk_lowerbound_success_rl2ppo = ppmilk_average_successrate_rl2ppo - ppmilk_average_stdsuccess_rl2ppo
    ppmilk_upperbound_success_rl2ppo = ppmilk_average_successrate_rl2ppo + ppmilk_average_stdsuccess_rl2ppo

    ppmilk_lowerbound_success_mamltrpo = np.where(ppmilk_lowerbound_success_mamltrpo >= 0,
                                                  ppmilk_lowerbound_success_mamltrpo, 0)
    ppmilk_upperbound_success_mamltrpo = np.where(ppmilk_upperbound_success_mamltrpo <= 100,
                                                  ppmilk_upperbound_success_mamltrpo, 100)
    ppmilk_lowerbound_success_rl2ppo = np.where(ppmilk_lowerbound_success_rl2ppo >= 0,
                                                ppmilk_lowerbound_success_rl2ppo, 0)
    ppmilk_upperbound_success_rl2ppo = np.where(ppmilk_upperbound_success_rl2ppo <= 100,
                                                ppmilk_upperbound_success_rl2ppo, 100)

    # NEXT FIGURES WITH THE REMAINING FIVE CRiSE 7 TASKS
    # ----------------------------------------

    # PickPlaceBread

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceBread/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    ppbread_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppbread_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppbread_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    ppbread_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    ppbread_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppbread_total_testenv_steps_mamltrpo = ppbread_total_env_steps_mamltrpo[np.where(
        ppbread_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppbread_metatest_avg_return_mamltrpo = ppbread_metatest_avg_return_mamltrpo[
        np.where(ppbread_metatest_avg_return_mamltrpo != '')].astype(float)
    ppbread_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppbread_metatest_avg_successrate_mamltrpo = (
            ppbread_metatest_avg_successrate_mamltrpo[np.where(ppbread_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    ppbread_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppbread_metatest_avg_stdreturn_mamltrpo = (
        ppbread_metatest_avg_stdreturn_mamltrpo[np.where(ppbread_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppbread_average_stdsuccess_mamltrpo = np.sqrt(
        (ppbread_average_successrate_mamltrpo / 100) * (1 - (ppbread_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceBread/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    ppbread_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppbread_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppbread_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    ppbread_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppbread_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    ppbread_total_testenv_steps_rl2ppo = ppbread_total_env_steps_rl2ppo[np.where(
        ppbread_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppbread_metatest_avg_return_rl2ppo = ppbread_metatest_avg_return_rl2ppo[
        np.where(ppbread_metatest_avg_return_rl2ppo != '')].astype(float)
    ppbread_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppbread_metatest_avg_successrate_rl2ppo = (
            ppbread_metatest_avg_successrate_rl2ppo[np.where(ppbread_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    ppbread_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppbread_metatest_avg_stdreturn_rl2ppo = (
        ppbread_metatest_avg_stdreturn_rl2ppo[np.where(ppbread_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppbread_average_stdsuccess_rl2ppo = np.sqrt(
        (ppbread_average_successrate_rl2ppo / 100) * (1 - (ppbread_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    ppbread_lowerbound_return_mamltrpo = ppbread_average_return_mamltrpo - ppbread_average_stdreturn_mamltrpo
    ppbread_upperbound_return_mamltrpo = ppbread_average_return_mamltrpo + ppbread_average_stdreturn_mamltrpo
    ppbread_lowerbound_return_rl2ppo = ppbread_average_return_rl2ppo - ppbread_average_stdreturn_rl2ppo
    ppbread_upperbound_return_rl2ppo = ppbread_average_return_rl2ppo + ppbread_average_stdreturn_rl2ppo

    ppbread_lowerbound_return_mamltrpo = np.where(ppbread_lowerbound_return_mamltrpo >= 0,
                                                  ppbread_lowerbound_return_mamltrpo, 0)
    ppbread_upperbound_return_mamltrpo = np.where(ppbread_upperbound_return_mamltrpo <= 500,
                                                  ppbread_upperbound_return_mamltrpo, 500)
    ppbread_lowerbound_return_rl2ppo = np.where(ppbread_lowerbound_return_rl2ppo >= 0,
                                                ppbread_lowerbound_return_rl2ppo, 0)
    ppbread_upperbound_return_rl2ppo = np.where(ppbread_upperbound_return_rl2ppo <= 500,
                                                ppbread_upperbound_return_rl2ppo, 500)

    ppbread_lowerbound_success_mamltrpo = ppbread_average_successrate_mamltrpo - ppbread_average_stdsuccess_mamltrpo
    ppbread_upperbound_success_mamltrpo = ppbread_average_successrate_mamltrpo + ppbread_average_stdsuccess_mamltrpo
    ppbread_lowerbound_success_rl2ppo = ppbread_average_successrate_rl2ppo - ppbread_average_stdsuccess_rl2ppo
    ppbread_upperbound_success_rl2ppo = ppbread_average_successrate_rl2ppo + ppbread_average_stdsuccess_rl2ppo

    ppbread_lowerbound_success_mamltrpo = np.where(ppbread_lowerbound_success_mamltrpo >= 0,
                                                   ppbread_lowerbound_success_mamltrpo, 0)
    ppbread_upperbound_success_mamltrpo = np.where(ppbread_upperbound_success_mamltrpo <= 100,
                                                   ppbread_upperbound_success_mamltrpo, 100)
    ppbread_lowerbound_success_rl2ppo = np.where(ppbread_lowerbound_success_rl2ppo >= 0,
                                                 ppbread_lowerbound_success_rl2ppo, 0)
    ppbread_upperbound_success_rl2ppo = np.where(ppbread_upperbound_success_rl2ppo <= 100,
                                                 ppbread_upperbound_success_rl2ppo, 100)

    # PickPlaceCereal

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceCereal/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    ppcereal_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppcereal_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppcereal_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    ppcereal_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    ppcereal_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppcereal_total_testenv_steps_mamltrpo = ppcereal_total_env_steps_mamltrpo[np.where(
        ppcereal_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppcereal_metatest_avg_return_mamltrpo = ppcereal_metatest_avg_return_mamltrpo[
        np.where(ppcereal_metatest_avg_return_mamltrpo != '')].astype(float)
    ppcereal_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppcereal_metatest_avg_successrate_mamltrpo = (
            ppcereal_metatest_avg_successrate_mamltrpo[np.where(ppcereal_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    ppcereal_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppcereal_metatest_avg_stdreturn_mamltrpo = (
        ppcereal_metatest_avg_stdreturn_mamltrpo[np.where(ppcereal_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppcereal_average_stdsuccess_mamltrpo = np.sqrt(
        (ppcereal_average_successrate_mamltrpo / 100) * (1 - (ppcereal_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceCereal/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    ppcereal_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppcereal_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppcereal_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    ppcereal_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppcereal_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    ppcereal_total_testenv_steps_rl2ppo = ppcereal_total_env_steps_rl2ppo[np.where(
        ppcereal_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppcereal_metatest_avg_return_rl2ppo = ppcereal_metatest_avg_return_rl2ppo[
        np.where(ppcereal_metatest_avg_return_rl2ppo != '')].astype(float)
    ppcereal_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppcereal_metatest_avg_successrate_rl2ppo = (
            ppcereal_metatest_avg_successrate_rl2ppo[np.where(ppcereal_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    ppcereal_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppcereal_metatest_avg_stdreturn_rl2ppo = (
        ppcereal_metatest_avg_stdreturn_rl2ppo[np.where(ppcereal_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppcereal_average_stdsuccess_rl2ppo = np.sqrt(
        (ppcereal_average_successrate_rl2ppo / 100) * (1 - (ppcereal_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    ppcereal_lowerbound_return_mamltrpo = ppcereal_average_return_mamltrpo - ppcereal_average_stdreturn_mamltrpo
    ppcereal_upperbound_return_mamltrpo = ppcereal_average_return_mamltrpo + ppcereal_average_stdreturn_mamltrpo
    ppcereal_lowerbound_return_rl2ppo = ppcereal_average_return_rl2ppo - ppcereal_average_stdreturn_rl2ppo
    ppcereal_upperbound_return_rl2ppo = ppcereal_average_return_rl2ppo + ppcereal_average_stdreturn_rl2ppo

    ppcereal_lowerbound_return_mamltrpo = np.where(ppcereal_lowerbound_return_mamltrpo >= 0,
                                                   ppcereal_lowerbound_return_mamltrpo, 0)
    ppcereal_upperbound_return_mamltrpo = np.where(ppcereal_upperbound_return_mamltrpo <= 500,
                                                   ppcereal_upperbound_return_mamltrpo, 500)
    ppcereal_lowerbound_return_rl2ppo = np.where(ppcereal_lowerbound_return_rl2ppo >= 0,
                                                 ppcereal_lowerbound_return_rl2ppo, 0)
    ppcereal_upperbound_return_rl2ppo = np.where(ppcereal_upperbound_return_rl2ppo <= 500,
                                                 ppcereal_upperbound_return_rl2ppo, 500)

    ppcereal_lowerbound_success_mamltrpo = ppcereal_average_successrate_mamltrpo - ppcereal_average_stdsuccess_mamltrpo
    ppcereal_upperbound_success_mamltrpo = ppcereal_average_successrate_mamltrpo + ppcereal_average_stdsuccess_mamltrpo
    ppcereal_lowerbound_success_rl2ppo = ppcereal_average_successrate_rl2ppo - ppcereal_average_stdsuccess_rl2ppo
    ppcereal_upperbound_success_rl2ppo = ppcereal_average_successrate_rl2ppo + ppcereal_average_stdsuccess_rl2ppo

    ppcereal_lowerbound_success_mamltrpo = np.where(ppcereal_lowerbound_success_mamltrpo >= 0,
                                                    ppcereal_lowerbound_success_mamltrpo, 0)
    ppcereal_upperbound_success_mamltrpo = np.where(ppcereal_upperbound_success_mamltrpo <= 100,
                                                    ppcereal_upperbound_success_mamltrpo, 100)
    ppcereal_lowerbound_success_rl2ppo = np.where(ppcereal_lowerbound_success_rl2ppo >= 0,
                                                  ppcereal_lowerbound_success_rl2ppo, 0)
    ppcereal_upperbound_success_rl2ppo = np.where(ppcereal_upperbound_success_rl2ppo <= 100,
                                                  ppcereal_upperbound_success_rl2ppo, 100)

    # StackBlocks

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_StackBlocks/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    stackblocks_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    stackblocks_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    stackblocks_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    stackblocks_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    stackblocks_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    stackblocks_total_testenv_steps_mamltrpo = stackblocks_total_env_steps_mamltrpo[np.where(
        stackblocks_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    stackblocks_metatest_avg_return_mamltrpo = stackblocks_metatest_avg_return_mamltrpo[
        np.where(stackblocks_metatest_avg_return_mamltrpo != '')].astype(float)
    stackblocks_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    stackblocks_metatest_avg_successrate_mamltrpo = (
            stackblocks_metatest_avg_successrate_mamltrpo[np.where(stackblocks_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    stackblocks_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    stackblocks_metatest_avg_stdreturn_mamltrpo = (
        stackblocks_metatest_avg_stdreturn_mamltrpo[np.where(stackblocks_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    stackblocks_average_stdsuccess_mamltrpo = np.sqrt(
        (stackblocks_average_successrate_mamltrpo / 100) * (1 - (stackblocks_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_StackBlocks/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    stackblocks_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    stackblocks_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    stackblocks_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    stackblocks_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    stackblocks_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    stackblocks_total_testenv_steps_rl2ppo = stackblocks_total_env_steps_rl2ppo[np.where(
        stackblocks_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    stackblocks_metatest_avg_return_rl2ppo = stackblocks_metatest_avg_return_rl2ppo[
        np.where(stackblocks_metatest_avg_return_rl2ppo != '')].astype(float)
    stackblocks_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    stackblocks_metatest_avg_successrate_rl2ppo = (
            stackblocks_metatest_avg_successrate_rl2ppo[np.where(stackblocks_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    stackblocks_metatest_avg_successrate_mamltrpo[
        np.where(stackblocks_metatest_avg_successrate_mamltrpo != 0.0)[0]] = 0.0
    stackblocks_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    stackblocks_metatest_avg_stdreturn_rl2ppo = (
        stackblocks_metatest_avg_stdreturn_rl2ppo[np.where(stackblocks_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    stackblocks_average_stdsuccess_rl2ppo = np.sqrt(
        (stackblocks_average_successrate_rl2ppo / 100) * (1 - (stackblocks_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    stackblocks_lowerbound_return_mamltrpo = stackblocks_average_return_mamltrpo - stackblocks_average_stdreturn_mamltrpo
    stackblocks_upperbound_return_mamltrpo = stackblocks_average_return_mamltrpo + stackblocks_average_stdreturn_mamltrpo
    stackblocks_lowerbound_return_rl2ppo = stackblocks_average_return_rl2ppo - stackblocks_average_stdreturn_rl2ppo
    stackblocks_upperbound_return_rl2ppo = stackblocks_average_return_rl2ppo + stackblocks_average_stdreturn_rl2ppo

    stackblocks_lowerbound_return_mamltrpo = np.where(stackblocks_lowerbound_return_mamltrpo >= 0,
                                                      stackblocks_lowerbound_return_mamltrpo, 0)
    stackblocks_upperbound_return_mamltrpo = np.where(stackblocks_upperbound_return_mamltrpo <= 500,
                                                      stackblocks_upperbound_return_mamltrpo, 500)
    stackblocks_lowerbound_return_rl2ppo = np.where(stackblocks_lowerbound_return_rl2ppo >= 0,
                                                    stackblocks_lowerbound_return_rl2ppo, 0)
    stackblocks_upperbound_return_rl2ppo = np.where(stackblocks_upperbound_return_rl2ppo <= 500,
                                                    stackblocks_upperbound_return_rl2ppo, 500)

    stackblocks_lowerbound_success_mamltrpo = stackblocks_average_successrate_mamltrpo - stackblocks_average_stdsuccess_mamltrpo
    stackblocks_upperbound_success_mamltrpo = stackblocks_average_successrate_mamltrpo + stackblocks_average_stdsuccess_mamltrpo
    stackblocks_lowerbound_success_rl2ppo = stackblocks_average_successrate_rl2ppo - stackblocks_average_stdsuccess_rl2ppo
    stackblocks_upperbound_success_rl2ppo = stackblocks_average_successrate_rl2ppo + stackblocks_average_stdsuccess_rl2ppo

    stackblocks_lowerbound_success_mamltrpo = np.where(stackblocks_lowerbound_success_mamltrpo >= 0,
                                                       stackblocks_lowerbound_success_mamltrpo, 0)
    stackblocks_upperbound_success_mamltrpo = np.where(stackblocks_upperbound_success_mamltrpo <= 100,
                                                       stackblocks_upperbound_success_mamltrpo, 100)
    stackblocks_lowerbound_success_rl2ppo = np.where(stackblocks_lowerbound_success_rl2ppo >= 0,
                                                     stackblocks_lowerbound_success_rl2ppo, 0)
    stackblocks_upperbound_success_rl2ppo = np.where(stackblocks_upperbound_success_rl2ppo <= 100,
                                                     stackblocks_upperbound_success_rl2ppo, 100)

    # PickPlaceCan

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceCan/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    ppcan_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppcan_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppcan_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    ppcan_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    ppcan_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppcan_total_testenv_steps_mamltrpo = ppcan_total_env_steps_mamltrpo[np.where(
        ppcan_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppcan_metatest_avg_return_mamltrpo = ppcan_metatest_avg_return_mamltrpo[
        np.where(ppcan_metatest_avg_return_mamltrpo != '')].astype(float)
    ppcan_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppcan_metatest_avg_successrate_mamltrpo = (
            ppcan_metatest_avg_successrate_mamltrpo[np.where(ppcan_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    ppcan_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppcan_metatest_avg_stdreturn_mamltrpo = (
        ppcan_metatest_avg_stdreturn_mamltrpo[np.where(ppcan_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppcan_average_stdsuccess_mamltrpo = np.sqrt(
        (ppcan_average_successrate_mamltrpo / 100) * (1 - (ppcan_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_PickPlaceCan/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    ppcan_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    ppcan_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    ppcan_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    ppcan_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    ppcan_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    ppcan_total_testenv_steps_rl2ppo = ppcan_total_env_steps_rl2ppo[np.where(
        ppcan_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    ppcan_metatest_avg_return_rl2ppo = ppcan_metatest_avg_return_rl2ppo[
        np.where(ppcan_metatest_avg_return_rl2ppo != '')].astype(float)
    ppcan_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    ppcan_metatest_avg_successrate_rl2ppo = (
            ppcan_metatest_avg_successrate_rl2ppo[np.where(ppcan_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    ppcan_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    ppcan_metatest_avg_stdreturn_rl2ppo = (
        ppcan_metatest_avg_stdreturn_rl2ppo[np.where(ppcan_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    ppcan_average_stdsuccess_rl2ppo = np.sqrt(
        (ppcan_average_successrate_rl2ppo / 100) * (1 - (ppcan_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    ppcan_lowerbound_return_mamltrpo = ppcan_average_return_mamltrpo - ppcan_average_stdreturn_mamltrpo
    ppcan_upperbound_return_mamltrpo = ppcan_average_return_mamltrpo + ppcan_average_stdreturn_mamltrpo
    ppcan_lowerbound_return_rl2ppo = ppcan_average_return_rl2ppo - ppcan_average_stdreturn_rl2ppo
    ppcan_upperbound_return_rl2ppo = ppcan_average_return_rl2ppo + ppcan_average_stdreturn_rl2ppo

    ppcan_lowerbound_return_mamltrpo = np.where(ppcan_lowerbound_return_mamltrpo >= 0,
                                                ppcan_lowerbound_return_mamltrpo, 0)
    ppcan_upperbound_return_mamltrpo = np.where(ppcan_upperbound_return_mamltrpo <= 500,
                                                ppcan_upperbound_return_mamltrpo, 500)
    ppcan_lowerbound_return_rl2ppo = np.where(ppcan_lowerbound_return_rl2ppo >= 0,
                                              ppcan_lowerbound_return_rl2ppo, 0)
    ppcan_upperbound_return_rl2ppo = np.where(ppcan_upperbound_return_rl2ppo <= 500,
                                              ppcan_upperbound_return_rl2ppo, 500)

    ppcan_lowerbound_success_mamltrpo = ppcan_average_successrate_mamltrpo - ppcan_average_stdsuccess_mamltrpo
    ppcan_upperbound_success_mamltrpo = ppcan_average_successrate_mamltrpo + ppcan_average_stdsuccess_mamltrpo
    ppcan_lowerbound_success_rl2ppo = ppcan_average_successrate_rl2ppo - ppcan_average_stdsuccess_rl2ppo
    ppcan_upperbound_success_rl2ppo = ppcan_average_successrate_rl2ppo + ppcan_average_stdsuccess_rl2ppo

    ppcan_lowerbound_success_mamltrpo = np.where(ppcan_lowerbound_success_mamltrpo >= 0,
                                                 ppcan_lowerbound_success_mamltrpo, 0)
    ppcan_upperbound_success_mamltrpo = np.where(ppcan_upperbound_success_mamltrpo <= 100,
                                                 ppcan_upperbound_success_mamltrpo, 100)
    ppcan_lowerbound_success_rl2ppo = np.where(ppcan_lowerbound_success_rl2ppo >= 0,
                                               ppcan_lowerbound_success_rl2ppo, 0)
    ppcan_upperbound_success_rl2ppo = np.where(ppcan_upperbound_success_rl2ppo <= 100,
                                               ppcan_upperbound_success_rl2ppo, 100)

    # NutAssemblySquare

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblySquare/crise1_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    nasquare_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    nasquare_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    nasquare_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    nasquare_total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    nasquare_metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    nasquare_total_testenv_steps_mamltrpo = nasquare_total_env_steps_mamltrpo[np.where(
        nasquare_metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    nasquare_metatest_avg_return_mamltrpo = nasquare_metatest_avg_return_mamltrpo[
        np.where(nasquare_metatest_avg_return_mamltrpo != '')].astype(float)
    nasquare_metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    nasquare_metatest_avg_successrate_mamltrpo = (
            nasquare_metatest_avg_successrate_mamltrpo[np.where(nasquare_metatest_avg_successrate_mamltrpo != '')]
            .astype(float) * 100.0)
    nasquare_metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    nasquare_metatest_avg_stdreturn_mamltrpo = (
        nasquare_metatest_avg_stdreturn_mamltrpo[np.where(nasquare_metatest_avg_stdreturn_mamltrpo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    nasquare_average_stdsuccess_mamltrpo = np.sqrt(
        (nasquare_average_successrate_mamltrpo / 100) * (1 - (nasquare_average_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblySquare/crise1_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    nasquare_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    nasquare_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    nasquare_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    nasquare_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    nasquare_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    nasquare_total_testenv_steps_rl2ppo = nasquare_total_env_steps_rl2ppo[np.where(
        nasquare_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    nasquare_metatest_avg_return_rl2ppo = nasquare_metatest_avg_return_rl2ppo[
        np.where(nasquare_metatest_avg_return_rl2ppo != '')].astype(float)
    nasquare_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    nasquare_metatest_avg_successrate_rl2ppo = (
            nasquare_metatest_avg_successrate_rl2ppo[np.where(nasquare_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    nasquare_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    nasquare_metatest_avg_stdreturn_rl2ppo = (
        nasquare_metatest_avg_stdreturn_rl2ppo[np.where(nasquare_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    nasquare_average_stdsuccess_rl2ppo = np.sqrt(
        (nasquare_average_successrate_rl2ppo / 100) * (1 - (nasquare_average_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    nasquare_lowerbound_return_mamltrpo = nasquare_average_return_mamltrpo - nasquare_average_stdreturn_mamltrpo
    nasquare_upperbound_return_mamltrpo = nasquare_average_return_mamltrpo + nasquare_average_stdreturn_mamltrpo
    nasquare_lowerbound_return_rl2ppo = nasquare_average_return_rl2ppo - nasquare_average_stdreturn_rl2ppo
    nasquare_upperbound_return_rl2ppo = nasquare_average_return_rl2ppo + nasquare_average_stdreturn_rl2ppo

    nasquare_lowerbound_return_mamltrpo = np.where(nasquare_lowerbound_return_mamltrpo >= 0,
                                                   nasquare_lowerbound_return_mamltrpo, 0)
    nasquare_upperbound_return_mamltrpo = np.where(nasquare_upperbound_return_mamltrpo <= 500,
                                                   nasquare_upperbound_return_mamltrpo, 500)
    nasquare_lowerbound_return_rl2ppo = np.where(nasquare_lowerbound_return_rl2ppo >= 0,
                                                 nasquare_lowerbound_return_rl2ppo, 0)
    nasquare_upperbound_return_rl2ppo = np.where(nasquare_upperbound_return_rl2ppo <= 500,
                                                 nasquare_upperbound_return_rl2ppo, 500)

    nasquare_lowerbound_success_mamltrpo = nasquare_average_successrate_mamltrpo - nasquare_average_stdsuccess_mamltrpo
    nasquare_upperbound_success_mamltrpo = nasquare_average_successrate_mamltrpo + nasquare_average_stdsuccess_mamltrpo
    nasquare_lowerbound_success_rl2ppo = nasquare_average_successrate_rl2ppo - nasquare_average_stdsuccess_rl2ppo
    nasquare_upperbound_success_rl2ppo = nasquare_average_successrate_rl2ppo + nasquare_average_stdsuccess_rl2ppo

    nasquare_lowerbound_success_mamltrpo = np.where(nasquare_lowerbound_success_mamltrpo >= 0,
                                                    nasquare_lowerbound_success_mamltrpo, 0)
    nasquare_upperbound_success_mamltrpo = np.where(nasquare_upperbound_success_mamltrpo <= 100,
                                                    nasquare_upperbound_success_mamltrpo, 100)
    nasquare_lowerbound_success_rl2ppo = np.where(nasquare_lowerbound_success_rl2ppo >= 0,
                                                  nasquare_lowerbound_success_rl2ppo, 0)
    nasquare_upperbound_success_rl2ppo = np.where(nasquare_upperbound_success_rl2ppo <= 100,
                                                  nasquare_upperbound_success_rl2ppo, 100)

    # Plot everything
    fig, axis = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))  # (8.81036, 5.8476)
    fig_2, axis_2 = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))
    fig_3, axis_3 = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))
    fig_4, axis_4 = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))

    # LiftBlock
    axis[0, 0].plot(blocklift_total_env_steps_mamltrpo[0::2], blocklift_average_return_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[0, 0].plot(blocklift_total_env_steps_rl2ppo[0::2], blocklift_average_return_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[0, 0].fill_between(blocklift_total_env_steps_mamltrpo[0::2], blocklift_lowerbound_return_mamltrpo[0::2],
                            blocklift_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[0, 0].fill_between(blocklift_total_env_steps_rl2ppo[0::2], blocklift_lowerbound_return_rl2ppo[0::2],
                            blocklift_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)
    axis[0, 0].set_ylim([-5, 520])
    # legend = axis[0, 0].legend()
    # legend.get_frame().set_facecolor('white')

    axis[1, 0].plot(blocklift_total_env_steps_mamltrpo[0::2], blocklift_average_successrate_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[1, 0].plot(blocklift_total_env_steps_rl2ppo[0::2], blocklift_average_successrate_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[1, 0].fill_between(blocklift_total_env_steps_mamltrpo[0::2], blocklift_lowerbound_success_mamltrpo[0::2],
                            blocklift_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[1, 0].fill_between(blocklift_total_env_steps_rl2ppo[0::2], blocklift_lowerbound_success_rl2ppo[0::2],
                            blocklift_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)
    axis[1, 0].set_ylim([-5, 105])

    # NutAssemblyRound
    axis[0, 1].plot(naround_total_env_steps_mamltrpo[0::2], naround_average_return_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[0, 1].plot(naround_total_env_steps_rl2ppo[0::2], naround_average_return_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[0, 1].fill_between(naround_total_env_steps_mamltrpo[0::2], naround_lowerbound_return_mamltrpo[0::2],
                            naround_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[0, 1].fill_between(naround_total_env_steps_rl2ppo[0::2], naround_lowerbound_return_rl2ppo[0::2],
                            naround_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis[1, 1].plot(naround_total_env_steps_mamltrpo[0::2], naround_average_successrate_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[1, 1].plot(naround_total_env_steps_rl2ppo[0::2], naround_average_successrate_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[1, 1].fill_between(naround_total_env_steps_mamltrpo[0::2], naround_lowerbound_success_mamltrpo[0::2],
                            naround_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[1, 1].fill_between(naround_total_env_steps_rl2ppo[0::2], naround_lowerbound_success_rl2ppo[0::2],
                            naround_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)
    legend = axis[0, 1].legend()
    legend.get_frame().set_facecolor('white')
    legend = axis[1, 1].legend()
    legend.get_frame().set_facecolor('white')

    # OpenDoor
    axis[0, 2].plot(opendoor_total_env_steps_mamltrpo[0::2], opendoor_average_return_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[0, 2].plot(opendoor_total_env_steps_rl2ppo[0::2], opendoor_average_return_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[0, 2].fill_between(opendoor_total_env_steps_mamltrpo[0::2], opendoor_lowerbound_return_mamltrpo[0::2],
                            opendoor_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[0, 2].fill_between(opendoor_total_env_steps_rl2ppo[0::2], opendoor_lowerbound_return_rl2ppo[0::2],
                            opendoor_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis[1, 2].plot(opendoor_total_env_steps_mamltrpo[0::2], opendoor_average_successrate_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[1, 2].plot(opendoor_total_env_steps_rl2ppo[0::2], opendoor_average_successrate_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[1, 2].fill_between(opendoor_total_env_steps_mamltrpo[0::2], opendoor_lowerbound_success_mamltrpo[0::2],
                            opendoor_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[1, 2].fill_between(opendoor_total_env_steps_rl2ppo[0::2], opendoor_lowerbound_success_rl2ppo[0::2],
                            opendoor_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)
    # axis[1, 2].xaxis.get_offset_text().set_visible(False)

    # NutAssemblyMixed
    axis[0, 3].plot(namixed_total_env_steps_mamltrpo[0::2], namixed_average_return_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[0, 3].plot(namixed_total_env_steps_rl2ppo[0::2], namixed_average_return_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[0, 3].fill_between(namixed_total_env_steps_mamltrpo[0::2], namixed_lowerbound_return_mamltrpo[0::2],
                            namixed_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[0, 3].fill_between(namixed_total_env_steps_rl2ppo[0::2], namixed_lowerbound_return_rl2ppo[0::2],
                            namixed_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis[1, 3].plot(namixed_total_env_steps_mamltrpo[0::2], namixed_average_successrate_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[1, 3].plot(namixed_total_env_steps_rl2ppo[0::2], namixed_average_successrate_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[1, 3].fill_between(namixed_total_env_steps_mamltrpo[0::2], namixed_lowerbound_success_mamltrpo[0::2],
                            namixed_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[1, 3].fill_between(namixed_total_env_steps_rl2ppo[0::2], namixed_lowerbound_success_rl2ppo[0::2],
                            namixed_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)

    # PickPlaceMilk
    axis[0, 4].plot(ppmilk_total_env_steps_mamltrpo[0::2], ppmilk_average_return_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[0, 4].plot(ppmilk_total_env_steps_rl2ppo[0::2], ppmilk_average_return_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[0, 4].fill_between(ppmilk_total_env_steps_mamltrpo[0::2], ppmilk_lowerbound_return_mamltrpo[0::2],
                            ppmilk_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[0, 4].fill_between(ppmilk_total_env_steps_rl2ppo[0::2], ppmilk_lowerbound_return_rl2ppo[0::2],
                            ppmilk_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis[1, 4].plot(ppmilk_total_env_steps_mamltrpo[0::2], ppmilk_average_successrate_mamltrpo[0::2], color='red',
                    label='MAML-TRPO')
    axis[1, 4].plot(ppmilk_total_env_steps_rl2ppo[0::2], ppmilk_average_successrate_rl2ppo[0::2], color='green',
                    label='RL2-PPO')
    axis[1, 4].fill_between(ppmilk_total_env_steps_mamltrpo[0::2], ppmilk_lowerbound_success_mamltrpo[0::2],
                            ppmilk_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis[1, 4].fill_between(ppmilk_total_env_steps_rl2ppo[0::2], ppmilk_lowerbound_success_rl2ppo[0::2],
                            ppmilk_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)

    # PickPlaceBread
    axis_3[0, 0].plot(ppbread_total_env_steps_mamltrpo[0::2], ppbread_average_return_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[0, 0].plot(ppbread_total_env_steps_rl2ppo[0::2], ppbread_average_return_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[0, 0].fill_between(ppbread_total_env_steps_mamltrpo[0::2], ppbread_lowerbound_return_mamltrpo[0::2],
                              ppbread_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[0, 0].fill_between(ppbread_total_env_steps_rl2ppo[0::2], ppbread_lowerbound_return_rl2ppo[0::2],
                              ppbread_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)
    axis_3[0, 0].set_ylim([-5, 520])
    legend = axis_3[0, 0].legend()
    legend.get_frame().set_facecolor('white')

    axis_3[1, 0].plot(ppbread_total_env_steps_mamltrpo[0::2], ppbread_average_successrate_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[1, 0].plot(ppbread_total_env_steps_rl2ppo[0::2], ppbread_average_successrate_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[1, 0].fill_between(ppbread_total_env_steps_mamltrpo[0::2], ppbread_lowerbound_success_mamltrpo[0::2],
                              ppbread_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[1, 0].fill_between(ppbread_total_env_steps_rl2ppo[0::2], ppbread_lowerbound_success_rl2ppo[0::2],
                              ppbread_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)
    axis_3[1, 0].set_ylim([-5, 105])
    legend = axis_3[1, 0].legend()
    legend.get_frame().set_facecolor('white')

    # PickPlaceCereal
    axis_3[0, 1].plot(ppcereal_total_env_steps_mamltrpo[0::2], ppcereal_average_return_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[0, 1].plot(ppcereal_total_env_steps_rl2ppo[0::2], ppcereal_average_return_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[0, 1].fill_between(ppcereal_total_env_steps_mamltrpo[0::2], ppcereal_lowerbound_return_mamltrpo[0::2],
                              ppcereal_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[0, 1].fill_between(ppcereal_total_env_steps_rl2ppo[0::2], ppcereal_lowerbound_return_rl2ppo[0::2],
                              ppcereal_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis_3[1, 1].plot(ppcereal_total_env_steps_mamltrpo[0::2], ppcereal_average_successrate_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[1, 1].plot(ppcereal_total_env_steps_rl2ppo[0::2], ppcereal_average_successrate_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[1, 1].fill_between(ppcereal_total_env_steps_mamltrpo[0::2], ppcereal_lowerbound_success_mamltrpo[0::2],
                              ppcereal_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[1, 1].fill_between(ppcereal_total_env_steps_rl2ppo[0::2], ppcereal_lowerbound_success_rl2ppo[0::2],
                              ppcereal_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)

    # StackBlocks
    axis_3[0, 2].plot(stackblocks_total_env_steps_mamltrpo[0::2], stackblocks_average_return_mamltrpo[0::2],
                      color='red',
                      label='MAML-TRPO')
    axis_3[0, 2].plot(stackblocks_total_env_steps_rl2ppo[0::2], stackblocks_average_return_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[0, 2].fill_between(stackblocks_total_env_steps_mamltrpo[0::2], stackblocks_lowerbound_return_mamltrpo[0::2],
                              stackblocks_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[0, 2].fill_between(stackblocks_total_env_steps_rl2ppo[0::2], stackblocks_lowerbound_return_rl2ppo[0::2],
                              stackblocks_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis_3[1, 2].plot(stackblocks_total_env_steps_mamltrpo[0::2], stackblocks_average_successrate_mamltrpo[0::2],
                      color='red',
                      label='MAML-TRPO')
    axis_3[1, 2].plot(stackblocks_total_env_steps_rl2ppo[0::2], stackblocks_average_successrate_rl2ppo[0::2],
                      color='green',
                      label='RL2-PPO')
    axis_3[1, 2].fill_between(stackblocks_total_env_steps_mamltrpo[0::2], stackblocks_lowerbound_success_mamltrpo[0::2],
                              stackblocks_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[1, 2].fill_between(stackblocks_total_env_steps_rl2ppo[0::2], stackblocks_lowerbound_success_rl2ppo[0::2],
                              stackblocks_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)
    # axis_3[1, 2].xaxis.get_offset_text().set_visible(False)

    # PickPlaceCan
    axis_3[0, 3].plot(ppcan_total_env_steps_mamltrpo[0::2], ppcan_average_return_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[0, 3].plot(ppcan_total_env_steps_rl2ppo[0::2], ppcan_average_return_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[0, 3].fill_between(ppcan_total_env_steps_mamltrpo[0::2], ppcan_lowerbound_return_mamltrpo[0::2],
                              ppcan_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[0, 3].fill_between(ppcan_total_env_steps_rl2ppo[0::2], ppcan_lowerbound_return_rl2ppo[0::2],
                              ppcan_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis_3[1, 3].plot(ppcan_total_env_steps_mamltrpo[0::2], ppcan_average_successrate_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[1, 3].plot(ppcan_total_env_steps_rl2ppo[0::2], ppcan_average_successrate_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[1, 3].fill_between(ppcan_total_env_steps_mamltrpo[0::2], ppcan_lowerbound_success_mamltrpo[0::2],
                              ppcan_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[1, 3].fill_between(ppcan_total_env_steps_rl2ppo[0::2], ppcan_lowerbound_success_rl2ppo[0::2],
                              ppcan_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)

    # NutAssemblySquare
    axis_3[0, 4].plot(nasquare_total_env_steps_mamltrpo[0::2], nasquare_average_return_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[0, 4].plot(nasquare_total_env_steps_rl2ppo[0::2], nasquare_average_return_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[0, 4].fill_between(nasquare_total_env_steps_mamltrpo[0::2], nasquare_lowerbound_return_mamltrpo[0::2],
                              nasquare_upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[0, 4].fill_between(nasquare_total_env_steps_rl2ppo[0::2], nasquare_lowerbound_return_rl2ppo[0::2],
                              nasquare_upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)

    axis_3[1, 4].plot(nasquare_total_env_steps_mamltrpo[0::2], nasquare_average_successrate_mamltrpo[0::2], color='red',
                      label='MAML-TRPO')
    axis_3[1, 4].plot(nasquare_total_env_steps_rl2ppo[0::2], nasquare_average_successrate_rl2ppo[0::2], color='green',
                      label='RL2-PPO')
    axis_3[1, 4].fill_between(nasquare_total_env_steps_mamltrpo[0::2], nasquare_lowerbound_success_mamltrpo[0::2],
                              nasquare_upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axis_3[1, 4].fill_between(nasquare_total_env_steps_rl2ppo[0::2], nasquare_lowerbound_success_rl2ppo[0::2],
                              nasquare_upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)

    # LiftBlock meta test
    axis_2[0, 0].plot(blocklift_total_testenv_steps_mamltrpo, blocklift_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_2[0, 0].plot(blocklift_total_testenv_steps_rl2ppo, blocklift_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')
    axis_2[0, 0].set_ylim([-5, 520])
    # legend = axis_2[0, 0].legend()
    # legend.get_frame().set_facecolor('white')

    axis_2[1, 0].plot(blocklift_total_testenv_steps_mamltrpo, blocklift_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_2[1, 0].plot(blocklift_total_testenv_steps_rl2ppo, blocklift_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    axis_2[1, 0].set_ylim([-5, 105])

    # NutAssemblyRound meta test
    axis_2[0, 1].plot(naround_total_testenv_steps_mamltrpo, naround_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_2[0, 1].plot(naround_total_testenv_steps_rl2ppo, naround_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_2[1, 1].plot(naround_total_testenv_steps_mamltrpo, naround_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_2[1, 1].plot(naround_total_testenv_steps_rl2ppo, naround_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    legend = axis_2[0, 1].legend()
    legend.get_frame().set_facecolor('white')
    legend = axis_2[1, 1].legend()
    legend.get_frame().set_facecolor('white')

    # OpenDoor meta test
    axis_2[0, 2].plot(opendoor_total_testenv_steps_mamltrpo, opendoor_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_2[0, 2].plot(opendoor_total_testenv_steps_rl2ppo, opendoor_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_2[1, 2].plot(opendoor_total_testenv_steps_mamltrpo, opendoor_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_2[1, 2].plot(opendoor_total_testenv_steps_rl2ppo, opendoor_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    # axis_2[1, 2].xaxis.get_offset_text().set_visible(False)

    # NutAssemblyMixed meta test
    axis_2[0, 3].plot(namixed_total_testenv_steps_mamltrpo, namixed_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_2[0, 3].plot(namixed_total_testenv_steps_rl2ppo, namixed_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_2[1, 3].plot(namixed_total_testenv_steps_mamltrpo, namixed_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_2[1, 3].plot(namixed_total_testenv_steps_rl2ppo, namixed_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')

    # PickPlaceMilk meta test
    axis_2[0, 4].plot(ppmilk_total_testenv_steps_mamltrpo, ppmilk_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_2[0, 4].plot(ppmilk_total_testenv_steps_rl2ppo, ppmilk_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_2[1, 4].plot(ppmilk_total_testenv_steps_mamltrpo, ppmilk_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_2[1, 4].plot(ppmilk_total_testenv_steps_rl2ppo, ppmilk_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')

    # PickPlaceBread meta test
    axis_4[0, 0].plot(ppbread_total_testenv_steps_mamltrpo, ppbread_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_4[0, 0].plot(ppbread_total_testenv_steps_rl2ppo, ppbread_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')
    axis_4[0, 0].set_ylim([-5, 520])
    legend = axis_4[0, 0].legend()
    legend.get_frame().set_facecolor('white')

    axis_4[1, 0].plot(ppbread_total_testenv_steps_mamltrpo, ppbread_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_4[1, 0].plot(ppbread_total_testenv_steps_rl2ppo, ppbread_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    axis_4[1, 0].set_ylim([-5, 105])
    legend = axis_4[1, 0].legend()
    legend.get_frame().set_facecolor('white')

    # PickPlaceCereal meta test
    axis_4[0, 1].plot(ppcereal_total_testenv_steps_mamltrpo, ppcereal_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_4[0, 1].plot(ppcereal_total_testenv_steps_rl2ppo, ppcereal_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_4[1, 1].plot(ppcereal_total_testenv_steps_mamltrpo, ppcereal_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_4[1, 1].plot(ppcereal_total_testenv_steps_rl2ppo, ppcereal_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')

    # StackBlocks meta test
    axis_4[0, 2].plot(stackblocks_total_testenv_steps_mamltrpo, stackblocks_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_4[0, 2].plot(stackblocks_total_testenv_steps_rl2ppo, stackblocks_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_4[1, 2].plot(stackblocks_total_testenv_steps_mamltrpo, stackblocks_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_4[1, 2].plot(stackblocks_total_testenv_steps_rl2ppo, stackblocks_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    # axis_4[1, 2].xaxis.get_offset_text().set_visible(False)
    # axis_4[1, 2].annotate(r'$\times$10$^{8}$', xy=(75, -4), xycoords='axes points')
    # axis_4[1, 2].text(0.74, -0.1, r'$\times$10$^{8}$', transform=axis_4[1, 2].transAxes)

    # PickPlaceCan meta test
    axis_4[0, 3].plot(ppcan_total_testenv_steps_mamltrpo, ppcan_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_4[0, 3].plot(ppcan_total_testenv_steps_rl2ppo, ppcan_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_4[1, 3].plot(ppcan_total_testenv_steps_mamltrpo, ppcan_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_4[1, 3].plot(ppcan_total_testenv_steps_rl2ppo, ppcan_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')

    # NutAssemblySquare meta test
    axis_4[0, 4].plot(nasquare_total_testenv_steps_mamltrpo, nasquare_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_4[0, 4].plot(nasquare_total_testenv_steps_rl2ppo, nasquare_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    axis_4[1, 4].plot(nasquare_total_testenv_steps_mamltrpo, nasquare_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_4[1, 4].plot(nasquare_total_testenv_steps_rl2ppo, nasquare_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')

    for i in range(5):
        axis[0, i].title.set_text(all_envs[i])
        # plt.setp(axis[0, i], xlabel=all_envs[i])

    for i in range(5):
        axis_2[0, i].title.set_text(all_envs[i])
        # plt.setp(axis_2[0, i], xlabel=all_envs[i])

    for i in range(5):
        axis_3[0, i].title.set_text(all_envs[i + 5])
        # plt.setp(axis_3[0, i], xlabel=all_envs[i])

    for i in range(5):
        axis_4[0, i].title.set_text(all_envs[i + 5])
        # plt.setp(axis_4[0, i], xlabel=all_envs[i])

    plt.setp(axis[0, 0], ylabel='Average Return')
    plt.setp(axis[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis_2[0, 0], ylabel='Return')
    plt.setp(axis_2[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis_3[0, 0], ylabel='Average Return')
    plt.setp(axis_3[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis_4[0, 0], ylabel='Return')
    plt.setp(axis_4[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis[1, 2], xlabel='Total Environment Steps')
    axis[1, 2].xaxis.set_label_coords(.5, -.2)
    plt.setp(axis_2[1, 2], xlabel='Total Environment Steps')
    axis_2[1, 2].xaxis.set_label_coords(.5, -.2)
    plt.setp(axis_3[1, 2], xlabel='Total Environment Steps')
    axis_3[1, 2].xaxis.set_label_coords(.5, -.2)
    plt.setp(axis_4[1, 2], xlabel='Total Environment Steps')
    axis_4[1, 2].xaxis.set_label_coords(.5, -.2)
    # fig.suptitle('Meta 1 - All Tasks with IIWA14 (Meta Training)', fontsize=14)
    # fig_2.suptitle('Meta 1 - All Tasks with IIWA14 (Meta Test)', fontsize=14)
    fig.tight_layout()
    fig_2.tight_layout()
    fig_3.tight_layout()
    fig_4.tight_layout()
    fig.savefig('CRiSE1_IIWA14_AllTasks_Train.pgf')
    fig_3.savefig('CRiSE1_IIWA14_AllTasks_Train2.pgf')
    fig_2.savefig('CRiSE1_IIWA14_AllTasks_Test.pgf')
    fig_4.savefig('CRiSE1_IIWA14_AllTasks_Test2.pgf')

    # Kuka IIWA14 with locked linear axes in CRiSE 7 with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE7/crise7_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO (meta testing in Meta 7 MAML_TRPO is executed every
    # five epochs, therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices
    # in meta test data and store them separately)
    metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_mamltrpo = total_env_steps_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    metatest_avg_return_mamltrpo = metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(
        float)
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(metatest_avg_successrate_mamltrpo
                                                                                    != '')].astype(float) * 100.0)
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_mamltrpo = (metatest_avg_stdreturn_mamltrpo[np.where(metatest_avg_stdreturn_mamltrpo != '')]
                                       .astype(float))
    average_stdsuccess_mamltrpo = np.sqrt(
        (average_successrate_mamltrpo / 100) * (1 - (average_successrate_mamltrpo / 100))) * 100
    metatest_avg_stdsuccess_mamltrpo = np.sqrt(
        (metatest_avg_successrate_mamltrpo / 100) * (1 - (metatest_avg_successrate_mamltrpo / 100))) * 100

    # Max success and return bar chart CRiSE 7 train/test tasks
    # CRiSE 7 test tasks MAML:
    metatest_nutassemblysquare_success_MAML = data_rows[:, header.index('MetaTest/nut-assembly-square/SuccessRate')]
    metatest_nutassemblysquare_success_MAML = metatest_nutassemblysquare_success_MAML[
        np.where(metatest_nutassemblysquare_success_MAML != '')].astype(float)
    metatest_pickplacecan_success_MAML = data_rows[:, header.index('MetaTest/pick-place-can/SuccessRate')]
    metatest_pickplacecan_success_MAML = (metatest_pickplacecan_success_MAML[np.where(metatest_pickplacecan_success_MAML
                                                                                      != '')].astype(float))
    metatest_stack_success_MAML = data_rows[:, header.index('MetaTest/stack-blocks/SuccessRate')]
    metatest_stack_success_MAML = metatest_stack_success_MAML[np.where(metatest_stack_success_MAML != '')].astype(float)
    metatest_nutassemblysquare_return_MAML = data_rows[:, header.index('MetaTest/nut-assembly-square/MaxReturn')]
    metatest_nutassemblysquare_return_MAML = (metatest_nutassemblysquare_return_MAML[
                                                  np.where(metatest_nutassemblysquare_return_MAML != '')].astype(float))
    metatest_pickplacecan_return_MAML = data_rows[:, header.index('MetaTest/pick-place-can/MaxReturn')]
    metatest_pickplacecan_return_MAML = metatest_pickplacecan_return_MAML[np.where(metatest_pickplacecan_return_MAML
                                                                                   != '')].astype(float)
    metatest_stack_return_MAML = data_rows[:, header.index('MetaTest/stack-blocks/MaxReturn')]
    metatest_stack_return_MAML = metatest_stack_return_MAML[np.where(metatest_stack_return_MAML != '')].astype(float)

    # CRiSE 7 train tasks:
    lift_success_MAML = data_rows[:, header.index('blocklifting/SuccessRate')].astype(float)
    door_success_MAML = data_rows[:, header.index('door-open/SuccessRate')].astype(float)
    nutassembly_success_MAML = data_rows[:, header.index('nut-assembly-mixed/SuccessRate')].astype(float)
    nutassemblyround_success_MAML = data_rows[:, header.index('nut-assembly-round/SuccessRate')].astype(float)
    pickplacebread_success_MAML = data_rows[:, header.index('pick-place-bread/SuccessRate')].astype(float)
    pickplacecereal_success_MAML = data_rows[:, header.index('pick-place-cereal/SuccessRate')].astype(float)
    pickplacemilk_success_MAML = data_rows[:, header.index('pick-place-milk/SuccessRate')].astype(float)
    lift_return_MAML = data_rows[:, header.index('blocklifting/MaxReturn')].astype(float)
    door_return_MAML = data_rows[:, header.index('door-open/MaxReturn')].astype(float)
    nutassembly_return_MAML = data_rows[:, header.index('nut-assembly-mixed/MaxReturn')].astype(float)
    nutassemblyround_return_MAML = data_rows[:, header.index('nut-assembly-round/MaxReturn')].astype(float)
    pickplacebread_return_MAML = data_rows[:, header.index('pick-place-bread/MaxReturn')].astype(float)
    pickplacecereal_return_MAML = data_rows[:, header.index('pick-place-cereal/MaxReturn')].astype(float)
    pickplacemilk_return_MAML = data_rows[:, header.index('pick-place-milk/MaxReturn')].astype(float)

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE7/crise7_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
                                       .astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                     .astype(float))
    average_stdsuccess_rl2ppo = np.sqrt(
        (average_successrate_rl2ppo / 100) * (1 - (average_successrate_rl2ppo / 100))) * 100
    metatest_avg_stdsuccess_rl2ppo = np.sqrt(
        (metatest_avg_successrate_rl2ppo / 100) * (1 - (metatest_avg_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    lowerbound_return_mamltrpo = average_return_mamltrpo - average_stdreturn_mamltrpo
    upperbound_return_mamltrpo = average_return_mamltrpo + average_stdreturn_mamltrpo
    lowerbound_return_rl2ppo = average_return_rl2ppo - average_stdreturn_rl2ppo
    upperbound_return_rl2ppo = average_return_rl2ppo + average_stdreturn_rl2ppo

    lowerbound_return_mamltrpo = np.where(lowerbound_return_mamltrpo >= 0, lowerbound_return_mamltrpo, 0)
    upperbound_return_mamltrpo = np.where(upperbound_return_mamltrpo <= 500, upperbound_return_mamltrpo, 500)
    lowerbound_return_rl2ppo = np.where(lowerbound_return_rl2ppo >= 0, lowerbound_return_rl2ppo, 0)
    upperbound_return_rl2ppo = np.where(upperbound_return_rl2ppo <= 500, upperbound_return_rl2ppo, 500)

    lowerbound_success_mamltrpo = average_successrate_mamltrpo - average_stdsuccess_mamltrpo
    upperbound_success_mamltrpo = average_successrate_mamltrpo + average_stdsuccess_mamltrpo
    lowerbound_success_rl2ppo = average_successrate_rl2ppo - average_stdsuccess_rl2ppo
    upperbound_success_rl2ppo = average_successrate_rl2ppo + average_stdsuccess_rl2ppo

    lowerbound_success_mamltrpo = np.where(lowerbound_success_mamltrpo >= 0, lowerbound_success_mamltrpo, 0)
    upperbound_success_mamltrpo = np.where(upperbound_success_mamltrpo <= 100, upperbound_success_mamltrpo, 100)
    lowerbound_success_rl2ppo = np.where(lowerbound_success_rl2ppo >= 0, lowerbound_success_rl2ppo, 0)
    upperbound_success_rl2ppo = np.where(upperbound_success_rl2ppo <= 100, upperbound_success_rl2ppo, 100)

    # Meta test standard deviation
    metatest_lowerbound_return_mamltrpo = metatest_avg_return_mamltrpo - metatest_avg_stdreturn_mamltrpo
    metatest_upperbound_return_mamltrpo = metatest_avg_return_mamltrpo + metatest_avg_stdreturn_mamltrpo
    metatest_lowerbound_return_rl2ppo = metatest_avg_return_rl2ppo - metatest_avg_stdreturn_rl2ppo
    metatest_upperbound_return_rl2ppo = metatest_avg_return_rl2ppo + metatest_avg_stdreturn_rl2ppo

    metatest_lowerbound_return_mamltrpo = np.where(metatest_lowerbound_return_mamltrpo >= 0,
                                                   metatest_lowerbound_return_mamltrpo, 0)
    metatest_upperbound_return_mamltrpo = np.where(metatest_upperbound_return_mamltrpo <= 500,
                                                   metatest_upperbound_return_mamltrpo, 500)
    metatest_lowerbound_return_rl2ppo = np.where(metatest_lowerbound_return_rl2ppo >= 0,
                                                 metatest_lowerbound_return_rl2ppo, 0)
    metatest_upperbound_return_rl2ppo = np.where(metatest_upperbound_return_rl2ppo <= 500,
                                                 metatest_upperbound_return_rl2ppo, 500)

    metatest_lowerbound_success_mamltrpo = metatest_avg_successrate_mamltrpo - metatest_avg_stdsuccess_mamltrpo
    metatest_upperbound_success_mamltrpo = metatest_avg_successrate_mamltrpo + metatest_avg_stdsuccess_mamltrpo
    metatest_lowerbound_success_rl2ppo = metatest_avg_successrate_rl2ppo - metatest_avg_stdsuccess_rl2ppo
    metatest_upperbound_success_rl2ppo = metatest_avg_successrate_rl2ppo + metatest_avg_stdsuccess_rl2ppo

    metatest_lowerbound_success_mamltrpo = np.where(metatest_lowerbound_success_mamltrpo >= 0,
                                                    metatest_lowerbound_success_mamltrpo, 0)
    metatest_upperbound_success_mamltrpo = np.where(metatest_upperbound_success_mamltrpo <= 100,
                                                    metatest_upperbound_success_mamltrpo, 100)
    metatest_lowerbound_success_rl2ppo = np.where(metatest_lowerbound_success_rl2ppo >= 0,
                                                  metatest_lowerbound_success_rl2ppo, 0)
    metatest_upperbound_success_rl2ppo = np.where(metatest_upperbound_success_rl2ppo <= 100,
                                                  metatest_upperbound_success_rl2ppo, 100)

    # Plot everything
    fig4, axis4 = plt.subplots(2, 1)
    fig4_2, axis4_2 = plt.subplots(2, 1)

    axis4[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis4[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis4[0].fill_between(total_env_steps_mamltrpo, lowerbound_return_mamltrpo,
                          upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axis4[0].fill_between(total_env_steps_rl2ppo, lowerbound_return_rl2ppo,
                          upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axis4[0].set_ylim([0, 500])
    legend = axis4[0].legend()
    legend.get_frame().set_facecolor('white')

    axis4[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis4[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis4[1].fill_between(total_env_steps_mamltrpo, lowerbound_success_mamltrpo,
                          upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axis4[1].fill_between(total_env_steps_rl2ppo, lowerbound_success_rl2ppo,
                          upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axis4[1].set_ylim([0, 100])
    legend = axis4[1].legend()
    legend.get_frame().set_facecolor('white')

    axis4_2[0].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis4_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis4_2[0].fill_between(total_testenv_steps_mamltrpo, metatest_lowerbound_return_mamltrpo,
                            metatest_upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axis4_2[0].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_return_rl2ppo,
                            metatest_upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axis4_2[0].set_ylim([0, 500])
    legend = axis4_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis4_2[1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis4_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis4_2[1].fill_between(total_testenv_steps_mamltrpo, metatest_lowerbound_success_mamltrpo,
                            metatest_upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axis4_2[1].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_success_rl2ppo,
                            metatest_upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axis4_2[1].set_ylim([0, 100])
    legend = axis4_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis4[0], ylabel='Average Return')
    plt.setp(axis4[1], ylabel='Success Rate (%)')
    plt.setp(axis4_2[0], ylabel='Average Return')
    plt.setp(axis4_2[1], ylabel='Success Rate (%)')
    plt.setp(axis4[1], xlabel='Total Environment Steps')
    plt.setp(axis4_2[1], xlabel='Total Environment Steps')
    fig4.suptitle('CRiSE 7 with IIWA14 (Meta-Training)', fontsize=14)
    fig4_2.suptitle('CRiSE 7 with IIWA14 (Meta-Test)', fontsize=14)
    fig4.savefig('CRiSE7_IIWA14_Train.pgf')
    fig4_2.savefig('CRiSE7_IIWA14_Test.pgf')

    # Max success and return bar chart CRiSE 7 train/test tasks
    # CRiSE 7 test tasks RL2:
    metatest_nutassemblysquare_success_RL2 = data_rows[:, header.index('MetaTest/nut-assembly-square/SuccessRate')]
    metatest_nutassemblysquare_success_RL2 = metatest_nutassemblysquare_success_RL2[np.where(
        metatest_nutassemblysquare_success_RL2 != '')].astype(float)
    metatest_pickplacecan_success_RL2 = data_rows[:, header.index('MetaTest/pick-place-can/SuccessRate')]
    metatest_pickplacecan_success_RL2 = (metatest_pickplacecan_success_RL2[np.where(metatest_pickplacecan_success_RL2
                                                                                    != '')].astype(float))
    metatest_stack_success_RL2 = data_rows[:, header.index('MetaTest/stack-blocks/SuccessRate')]
    metatest_stack_success_RL2 = metatest_stack_success_RL2[np.where(metatest_stack_success_RL2 != '')].astype(float)
    metatest_nutassemblysquare_return_RL2 = data_rows[:, header.index('MetaTest/nut-assembly-square/MaxReturn')]
    metatest_nutassemblysquare_return_RL2 = (metatest_nutassemblysquare_return_RL2[np.where(
        metatest_nutassemblysquare_return_RL2 != '')].astype(float))
    metatest_pickplacecan_return_RL2 = data_rows[:, header.index('MetaTest/pick-place-can/MaxReturn')]
    metatest_pickplacecan_return_RL2 = (
        metatest_pickplacecan_return_RL2[np.where(metatest_pickplacecan_return_RL2 != '')]
        .astype(float))
    metatest_stack_return_RL2 = data_rows[:, header.index('MetaTest/stack-blocks/MaxReturn')]
    metatest_stack_return_RL2 = metatest_stack_return_RL2[np.where(metatest_stack_return_RL2 != '')].astype(float)

    # CRiSE 7 train tasks:
    lift_success_RL2 = data_rows[:, header.index('blocklifting/SuccessRate')].astype(float)
    door_success_RL2 = data_rows[:, header.index('door-open/SuccessRate')].astype(float)
    nutassembly_success_RL2 = data_rows[:, header.index('nut-assembly-mixed/SuccessRate')].astype(float)
    nutassemblyround_success_RL2 = data_rows[:, header.index('nut-assembly-round/SuccessRate')].astype(float)
    pickplacebread_success_RL2 = data_rows[:, header.index('pick-place-bread/SuccessRate')].astype(float)
    pickplacecereal_success_RL2 = data_rows[:, header.index('pick-place-cereal/SuccessRate')].astype(float)
    pickplacemilk_success_RL2 = data_rows[:, header.index('pick-place-milk/SuccessRate')].astype(float)
    lift_return_RL2 = data_rows[:, header.index('blocklifting/MaxReturn')].astype(float)
    door_return_RL2 = data_rows[:, header.index('door-open/MaxReturn')].astype(float)
    nutassembly_return_RL2 = data_rows[:, header.index('nut-assembly-mixed/MaxReturn')].astype(float)
    nutassemblyround_return_RL2 = data_rows[:, header.index('nut-assembly-round/MaxReturn')].astype(float)
    pickplacebread_return_RL2 = data_rows[:, header.index('pick-place-bread/MaxReturn')].astype(float)
    pickplacecereal_return_RL2 = data_rows[:, header.index('pick-place-cereal/MaxReturn')].astype(float)
    pickplacemilk_return_RL2 = data_rows[:, header.index('pick-place-milk/MaxReturn')].astype(float)

    train_envs = ["OpenDoor", "LiftBlock", "NutAssemblyRound", "NutAssemblyMixed", "PickPlaceMilk",
                  "PickPlaceCereal", "PickPlaceBread"]
    test_envs = ["PickPlaceCan", "NutAssemblySquare", "StackBlocks"]

    max_success_train_envs_rl2 = [np.amax(door_success_RL2), np.amax(lift_success_RL2),
                                  np.amax(nutassemblyround_success_RL2), np.amax(nutassembly_success_RL2),
                                  np.amax(pickplacemilk_success_RL2), np.amax(pickplacecereal_success_RL2),
                                  np.amax(pickplacebread_success_RL2)]

    max_success_train_envs_maml = [np.amax(door_success_MAML), np.amax(lift_success_MAML),
                                   np.amax(nutassemblyround_success_MAML), np.amax(nutassembly_success_MAML),
                                   np.amax(pickplacemilk_success_MAML), np.amax(pickplacecereal_success_MAML),
                                   np.amax(pickplacebread_success_MAML)]

    max_success_test_envs_rl2 = [np.amax(metatest_pickplacecan_success_RL2),
                                 np.amax(metatest_nutassemblysquare_success_RL2),
                                 np.amax(metatest_stack_success_RL2)]

    max_success_test_envs_maml = [np.amax(metatest_pickplacecan_success_MAML),
                                  np.amax(metatest_nutassemblysquare_success_MAML),
                                  np.amax(metatest_stack_success_MAML)]

    max_success_train_envs_rl2 = [element * 100.0 for element in max_success_train_envs_rl2]
    max_success_test_envs_rl2 = [element * 100.0 for element in max_success_test_envs_rl2]
    max_success_train_envs_maml = [element * 100.0 for element in max_success_train_envs_maml]
    max_success_test_envs_maml = [element * 100.0 for element in max_success_test_envs_maml]

    max_return_train_envs_rl2 = [np.amax(door_return_RL2), np.amax(lift_return_RL2),
                                 np.amax(nutassemblyround_return_RL2),
                                 np.amax(nutassembly_return_RL2), np.amax(pickplacemilk_return_RL2),
                                 np.amax(pickplacecereal_return_RL2), np.amax(pickplacebread_return_RL2)]

    max_return_test_envs_rl2 = [np.amax(metatest_pickplacecan_return_RL2),
                                np.amax(metatest_nutassemblysquare_return_RL2),
                                np.amax(metatest_stack_return_RL2)]

    max_return_train_envs_maml = [np.amax(door_return_MAML), np.amax(lift_return_MAML),
                                  np.amax(nutassemblyround_return_MAML), np.amax(nutassembly_return_MAML),
                                  np.amax(pickplacemilk_return_MAML), np.amax(pickplacecereal_return_MAML),
                                  np.amax(pickplacebread_return_MAML)]

    max_return_test_envs_maml = [np.amax(metatest_pickplacecan_return_MAML),
                                 np.amax(metatest_nutassemblysquare_return_MAML),
                                 np.amax(metatest_stack_return_MAML)]

    """
    width_single_bar = 0.4
    fig5, axis5 = plt.subplots(2, 1)
    fig5_2, axis5_2 = plt.subplots(2, 1)
    bar_y_pos1 = np.arange(7)
    bar_y_pos2 = np.arange(3)
    axis5[0].barh(bar_y_pos1, max_success_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    axis5[0].barh(bar_y_pos1 + width_single_bar, max_success_train_envs_maml, height=-width_single_bar,
                  alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5[0].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis5[0].legend()
    legend.get_frame().set_facecolor('white')
    axis5[0].set_xlim([0, 100])
    axis5[0].set_title('Max Success Rate (%)', fontsize=12)
    axis5[1].set_title('Max Return', fontsize=12)
    axis5[1].barh(bar_y_pos1, max_return_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                     label='RL2-PPO', align='edge')
    axis5[1].barh(bar_y_pos1 + width_single_bar, max_return_train_envs_maml, height=-width_single_bar,
                     alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5[1].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis5[1].legend()
    legend.get_frame().set_facecolor('white')
    axis5[1].set_xlim([0, 500])
    axis5_2[0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    axis5_2[0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
                  alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5_2[0].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis5_2[0].legend()
    legend.get_frame().set_facecolor('white')
    axis5_2[0].set_xlim([0, 100])
    axis5_2[0].set_title('Max Success Rate (%)', fontsize=12)
    axis5_2[1].set_title('Max Return', fontsize=12)
    axis5_2[1].barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                    label='RL2-PPO', align='edge')
    axis5_2[1].barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
                    alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5_2[1].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis5_2[1].legend()
    legend.get_frame().set_facecolor('white')
    axis5_2[1].set_xlim([0, 500])
    # plt.setp(axis5[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis5[1], xlabel='Max Return')
    # plt.setp(axis5_2[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis5_2[1], xlabel='Max Return')
    fig5.suptitle('CRiSE 7 - Max Returns/Success Rates per Meta-Train Task (IIWA14)', fontsize=13)
    fig5_2.suptitle('CRiSE 7 - Max Returns/Success Rates per Meta-Test Task (IIWA14)', fontsize=13)
    fig5.tight_layout()
    fig5_2.tight_layout()
    fig5.savefig('CRiSE7_IIWA14_SuccessReturns_MetaTrain.pgf')
    fig5_2.savefig('CRiSE7_IIWA14_SuccessReturns_MetaTest.pgf')
    """
    width_single_bar = 0.4
    fig5, axis5 = plt.subplots(2, 1)
    fig5_2, axis5_2 = plt.subplots(1, 1)
    bar_y_pos1 = np.arange(7)
    bar_y_pos2 = np.arange(3)
    axis5[0].barh(bar_y_pos1, max_success_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    axis5[0].barh(bar_y_pos1 + width_single_bar, max_success_train_envs_maml, height=-width_single_bar,
                  alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5[0].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis5[0].legend()
    legend.get_frame().set_facecolor('white')
    axis5[0].set_xlim([0, 100])
    axis5[0].set_title('Max Success Rate (%)', fontsize=12)
    axis5[1].set_title('Max Return', fontsize=12)
    axis5[1].barh(bar_y_pos1, max_return_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    axis5[1].barh(bar_y_pos1 + width_single_bar, max_return_train_envs_maml, height=-width_single_bar,
                  alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5[1].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis5[1].legend()
    legend.get_frame().set_facecolor('white')
    axis5[1].set_xlim([0, 500])
    # axis5_2[0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
    #               label='RL2-PPO', align='edge')
    # axis5_2[0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
    #               alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    # axis5_2[0].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    # legend = axis5_2[0].legend()
    # legend.get_frame().set_facecolor('white')
    # axis5_2[0].set_xlim([0, 100])
    # axis5_2[0].set_title('Max Success Rate (%)', fontsize=12)
    axis5_2.set_title('Max Return', fontsize=12)
    axis5_2.barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
    axis5_2.barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis5_2.set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis5_2.legend()
    legend.get_frame().set_facecolor('white')
    axis5_2.set_xlim([0, 500])
    # plt.setp(axis5[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis5[1], xlabel='Max Return')
    # plt.setp(axis5_2[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis5_2[1], xlabel='Max Return')
    fig5.suptitle('CRiSE 7 - Max Returns/Success Rates per Meta-Train Task (IIWA14)', fontsize=13)
    fig5_2.suptitle('CRiSE 7 - Max Returns per Meta-Test Task (IIWA14)', fontsize=13)
    fig5.tight_layout()
    fig5_2.tight_layout()
    fig5.savefig('CRiSE7_IIWA14_SuccessReturns_MetaTrain.pgf')
    fig5_2.savefig('CRiSE7_IIWA14_SuccessReturns_MetaTest.pgf')

    # Rethink Robotics Sawyer in CRiSE 7 with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_CRiSE7/crise7_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO (meta testing in Meta 7 MAML_TRPO is executed every
    # ten epochs, therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices
    # in meta test data and store them separately)
    metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_mamltrpo = total_env_steps_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    metatest_avg_return_mamltrpo = metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(
        float)
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(metatest_avg_successrate_mamltrpo
                                                                                    != '')].astype(float) * 100.0)
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_mamltrpo = (metatest_avg_stdreturn_mamltrpo[np.where(metatest_avg_stdreturn_mamltrpo != '')]
                                       .astype(float))
    average_stdsuccess_mamltrpo = np.sqrt(
        (average_successrate_mamltrpo / 100) * (1 - (average_successrate_mamltrpo / 100))) * 100
    metatest_avg_stdsuccess_mamltrpo = np.sqrt(
        (metatest_avg_successrate_mamltrpo / 100) * (1 - (metatest_avg_successrate_mamltrpo / 100))) * 100

    # Max success and return bar chart CRiSE 7 train/test tasks
    # CRiSE 7 test tasks MAML:
    metatest_nutassemblysquare_success_MAML = data_rows[:, header.index('MetaTest/nut-assembly-square/SuccessRate')]
    metatest_nutassemblysquare_success_MAML = metatest_nutassemblysquare_success_MAML[np.where(
        metatest_nutassemblysquare_success_MAML != '')].astype(float)
    metatest_pickplacecan_success_MAML = data_rows[:, header.index('MetaTest/pick-place-can/SuccessRate')]
    metatest_pickplacecan_success_MAML = (metatest_pickplacecan_success_MAML[np.where(metatest_pickplacecan_success_MAML
                                                                                      != '')].astype(float))
    metatest_stack_success_MAML = data_rows[:, header.index('MetaTest/stack-blocks/SuccessRate')]
    metatest_stack_success_MAML = metatest_stack_success_MAML[np.where(metatest_stack_success_MAML != '')].astype(float)
    metatest_nutassemblysquare_return_MAML = data_rows[:, header.index('MetaTest/nut-assembly-square/MaxReturn')]
    metatest_nutassemblysquare_return_MAML = (metatest_nutassemblysquare_return_MAML[np.where(
        metatest_nutassemblysquare_return_MAML != '')].astype(float))
    metatest_pickplacecan_return_MAML = data_rows[:, header.index('MetaTest/pick-place-can/MaxReturn')]
    metatest_pickplacecan_return_MAML = (metatest_pickplacecan_return_MAML[np.where(metatest_pickplacecan_return_MAML
                                                                                    != '')].astype(float))
    metatest_stack_return_MAML = data_rows[:, header.index('MetaTest/stack-blocks/MaxReturn')]
    metatest_stack_return_MAML = metatest_stack_return_MAML[np.where(metatest_stack_return_MAML != '')].astype(float)

    # CRiSE 7 train tasks:
    lift_success_MAML = data_rows[:, header.index('blocklifting/SuccessRate')].astype(float)
    door_success_MAML = data_rows[:, header.index('door-open/SuccessRate')].astype(float)
    nutassembly_success_MAML = data_rows[:, header.index('nut-assembly-mixed/SuccessRate')].astype(float)
    nutassemblyround_success_MAML = data_rows[:, header.index('nut-assembly-round/SuccessRate')].astype(float)
    pickplacebread_success_MAML = data_rows[:, header.index('pick-place-bread/SuccessRate')].astype(float)
    pickplacecereal_success_MAML = data_rows[:, header.index('pick-place-cereal/SuccessRate')].astype(float)
    pickplacemilk_success_MAML = data_rows[:, header.index('pick-place-milk/SuccessRate')].astype(float)
    lift_return_MAML = data_rows[:, header.index('blocklifting/MaxReturn')].astype(float)
    door_return_MAML = data_rows[:, header.index('door-open/MaxReturn')].astype(float)
    nutassembly_return_MAML = data_rows[:, header.index('nut-assembly-mixed/MaxReturn')].astype(float)
    nutassemblyround_return_MAML = data_rows[:, header.index('nut-assembly-round/MaxReturn')].astype(float)
    pickplacebread_return_MAML = data_rows[:, header.index('pick-place-bread/MaxReturn')].astype(float)
    pickplacecereal_return_MAML = data_rows[:, header.index('pick-place-cereal/MaxReturn')].astype(float)
    pickplacemilk_return_MAML = data_rows[:, header.index('pick-place-milk/MaxReturn')].astype(float)

    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_CRiSE7/crise7_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
                                       .astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                     .astype(float))
    average_stdsuccess_rl2ppo = np.sqrt(
        (average_successrate_rl2ppo / 100) * (1 - (average_successrate_rl2ppo / 100))) * 100
    metatest_avg_stdsuccess_rl2ppo = np.sqrt(
        (metatest_avg_successrate_rl2ppo / 100) * (1 - (metatest_avg_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    lowerbound_return_mamltrpo = average_return_mamltrpo - average_stdreturn_mamltrpo
    upperbound_return_mamltrpo = average_return_mamltrpo + average_stdreturn_mamltrpo
    lowerbound_return_rl2ppo = average_return_rl2ppo - average_stdreturn_rl2ppo
    upperbound_return_rl2ppo = average_return_rl2ppo + average_stdreturn_rl2ppo

    lowerbound_return_mamltrpo = np.where(lowerbound_return_mamltrpo >= 0, lowerbound_return_mamltrpo, 0)
    upperbound_return_mamltrpo = np.where(upperbound_return_mamltrpo <= 500, upperbound_return_mamltrpo, 500)
    lowerbound_return_rl2ppo = np.where(lowerbound_return_rl2ppo >= 0, lowerbound_return_rl2ppo, 0)
    upperbound_return_rl2ppo = np.where(upperbound_return_rl2ppo <= 500, upperbound_return_rl2ppo, 500)

    lowerbound_success_mamltrpo = average_successrate_mamltrpo - average_stdsuccess_mamltrpo
    upperbound_success_mamltrpo = average_successrate_mamltrpo + average_stdsuccess_mamltrpo
    lowerbound_success_rl2ppo = average_successrate_rl2ppo - average_stdsuccess_rl2ppo
    upperbound_success_rl2ppo = average_successrate_rl2ppo + average_stdsuccess_rl2ppo

    lowerbound_success_mamltrpo = np.where(lowerbound_success_mamltrpo >= 0, lowerbound_success_mamltrpo, 0)
    upperbound_success_mamltrpo = np.where(upperbound_success_mamltrpo <= 100, upperbound_success_mamltrpo, 100)
    lowerbound_success_rl2ppo = np.where(lowerbound_success_rl2ppo >= 0, lowerbound_success_rl2ppo, 0)
    upperbound_success_rl2ppo = np.where(upperbound_success_rl2ppo <= 100, upperbound_success_rl2ppo, 100)

    # Meta test standard deviation
    metatest_lowerbound_return_mamltrpo = metatest_avg_return_mamltrpo - metatest_avg_stdreturn_mamltrpo
    metatest_upperbound_return_mamltrpo = metatest_avg_return_mamltrpo + metatest_avg_stdreturn_mamltrpo
    metatest_lowerbound_return_rl2ppo = metatest_avg_return_rl2ppo - metatest_avg_stdreturn_rl2ppo
    metatest_upperbound_return_rl2ppo = metatest_avg_return_rl2ppo + metatest_avg_stdreturn_rl2ppo

    metatest_lowerbound_return_mamltrpo = np.where(metatest_lowerbound_return_mamltrpo >= 0,
                                                   metatest_lowerbound_return_mamltrpo, 0)
    metatest_upperbound_return_mamltrpo = np.where(metatest_upperbound_return_mamltrpo <= 500,
                                                   metatest_upperbound_return_mamltrpo, 500)
    metatest_lowerbound_return_rl2ppo = np.where(metatest_lowerbound_return_rl2ppo >= 0,
                                                 metatest_lowerbound_return_rl2ppo, 0)
    metatest_upperbound_return_rl2ppo = np.where(metatest_upperbound_return_rl2ppo <= 500,
                                                 metatest_upperbound_return_rl2ppo, 500)

    metatest_lowerbound_success_mamltrpo = metatest_avg_successrate_mamltrpo - metatest_avg_stdsuccess_mamltrpo
    metatest_upperbound_success_mamltrpo = metatest_avg_successrate_mamltrpo + metatest_avg_stdsuccess_mamltrpo
    metatest_lowerbound_success_rl2ppo = metatest_avg_successrate_rl2ppo - metatest_avg_stdsuccess_rl2ppo
    metatest_upperbound_success_rl2ppo = metatest_avg_successrate_rl2ppo + metatest_avg_stdsuccess_rl2ppo

    metatest_lowerbound_success_mamltrpo = np.where(metatest_lowerbound_success_mamltrpo >= 0,
                                                    metatest_lowerbound_success_mamltrpo, 0)
    metatest_upperbound_success_mamltrpo = np.where(metatest_upperbound_success_mamltrpo <= 100,
                                                    metatest_upperbound_success_mamltrpo, 100)
    metatest_lowerbound_success_rl2ppo = np.where(metatest_lowerbound_success_rl2ppo >= 0,
                                                  metatest_lowerbound_success_rl2ppo, 0)
    metatest_upperbound_success_rl2ppo = np.where(metatest_upperbound_success_rl2ppo <= 100,
                                                  metatest_upperbound_success_rl2ppo, 100)

    # Plot everything
    fig6, axis6 = plt.subplots(2, 1)
    fig6_2, axis6_2 = plt.subplots(2, 1)

    axis6[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis6[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis6[0].fill_between(total_env_steps_mamltrpo, lowerbound_return_mamltrpo,
                          upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axis6[0].fill_between(total_env_steps_rl2ppo, lowerbound_return_rl2ppo,
                          upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axis6[0].set_ylim([0, 500])
    legend = axis6[0].legend()
    legend.get_frame().set_facecolor('white')

    axis6[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis6[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis6[1].fill_between(total_env_steps_mamltrpo, lowerbound_success_mamltrpo,
                          upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axis6[1].fill_between(total_env_steps_rl2ppo, lowerbound_success_rl2ppo,
                          upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axis6[1].set_ylim([0, 100])
    legend = axis6[1].legend()
    legend.get_frame().set_facecolor('white')

    axis6_2[0].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis6_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis6_2[0].fill_between(total_testenv_steps_mamltrpo, metatest_lowerbound_return_mamltrpo,
                            metatest_upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axis6_2[0].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_return_rl2ppo,
                            metatest_upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axis6_2[0].set_ylim([0, 500])
    legend = axis6_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis6_2[1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis6_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis6_2[1].fill_between(total_testenv_steps_mamltrpo, metatest_lowerbound_success_mamltrpo,
                            metatest_upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axis6_2[1].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_success_rl2ppo,
                            metatest_upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axis6_2[1].set_ylim([0, 100])
    legend = axis6_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis6[0], ylabel='Average Return')
    plt.setp(axis6[1], ylabel='Success Rate (%)')
    plt.setp(axis6_2[0], ylabel='Average Return')
    plt.setp(axis6_2[1], ylabel='Success Rate (%)')
    plt.setp(axis6[1], xlabel='Total Environment Steps')
    plt.setp(axis6_2[1], xlabel='Total Environment Steps')
    fig6.suptitle('CRiSE 7 with Sawyer (Meta-Training)', fontsize=14)
    fig6_2.suptitle('CRiSE 7 with Sawyer (Meta-Test)', fontsize=14)
    fig6.savefig('CRiSE7_Sawyer_Train.pgf')
    fig6_2.savefig('CRiSE7_Sawyer_Test.pgf')

    # Max success and return bar chart CRiSE 7 train/test tasks
    # CRiSE 7 test tasks RL2:
    metatest_nutassemblysquare_success_RL2 = data_rows[:, header.index('MetaTest/nut-assembly-square/SuccessRate')]
    metatest_nutassemblysquare_success_RL2 = metatest_nutassemblysquare_success_RL2[np.where(
        metatest_nutassemblysquare_success_RL2 != '')].astype(float)
    metatest_pickplacecan_success_RL2 = data_rows[:, header.index('MetaTest/pick-place-can/SuccessRate')]
    metatest_pickplacecan_success_RL2 = (metatest_pickplacecan_success_RL2[np.where(metatest_pickplacecan_success_RL2
                                                                                    != '')].astype(float))
    metatest_stack_success_RL2 = data_rows[:, header.index('MetaTest/stack-blocks/SuccessRate')]
    metatest_stack_success_RL2 = metatest_stack_success_RL2[np.where(metatest_stack_success_RL2 != '')].astype(float)
    metatest_nutassemblysquare_return_RL2 = data_rows[:, header.index('MetaTest/nut-assembly-square/MaxReturn')]
    metatest_nutassemblysquare_return_RL2 = (metatest_nutassemblysquare_return_RL2[np.where(
        metatest_nutassemblysquare_return_RL2 != '')].astype(float))
    metatest_pickplacecan_return_RL2 = data_rows[:, header.index('MetaTest/pick-place-can/MaxReturn')]
    metatest_pickplacecan_return_RL2 = (
        metatest_pickplacecan_return_RL2[np.where(metatest_pickplacecan_return_RL2 != '')]
        .astype(float))
    metatest_stack_return_RL2 = data_rows[:, header.index('MetaTest/stack-blocks/MaxReturn')]
    metatest_stack_return_RL2 = metatest_stack_return_RL2[np.where(metatest_stack_return_RL2 != '')].astype(float)

    # CRiSE 7 train tasks:
    lift_success_RL2 = data_rows[:, header.index('blocklifting/SuccessRate')].astype(float)
    door_success_RL2 = data_rows[:, header.index('door-open/SuccessRate')].astype(float)
    nutassembly_success_RL2 = data_rows[:, header.index('nut-assembly-mixed/SuccessRate')].astype(float)
    nutassemblyround_success_RL2 = data_rows[:, header.index('nut-assembly-round/SuccessRate')].astype(float)
    pickplacebread_success_RL2 = data_rows[:, header.index('pick-place-bread/SuccessRate')].astype(float)
    pickplacecereal_success_RL2 = data_rows[:, header.index('pick-place-cereal/SuccessRate')].astype(float)
    pickplacemilk_success_RL2 = data_rows[:, header.index('pick-place-milk/SuccessRate')].astype(float)
    lift_return_RL2 = data_rows[:, header.index('blocklifting/MaxReturn')].astype(float)
    door_return_RL2 = data_rows[:, header.index('door-open/MaxReturn')].astype(float)
    nutassembly_return_RL2 = data_rows[:, header.index('nut-assembly-mixed/MaxReturn')].astype(float)
    nutassemblyround_return_RL2 = data_rows[:, header.index('nut-assembly-round/MaxReturn')].astype(float)
    pickplacebread_return_RL2 = data_rows[:, header.index('pick-place-bread/MaxReturn')].astype(float)
    pickplacecereal_return_RL2 = data_rows[:, header.index('pick-place-cereal/MaxReturn')].astype(float)
    pickplacemilk_return_RL2 = data_rows[:, header.index('pick-place-milk/MaxReturn')].astype(float)

    train_envs = ["OpenDoor", "LiftBlock", "NutAssemblyRound", "NutAssembly", "PickPlaceMilk",
                  "PickPlaceCereal", "PickPlaceBread"]
    test_envs = ["PickPlaceCan", "NutAssemblySquare", "StackBlocks"]

    max_success_train_envs_rl2 = [np.amax(door_success_RL2), np.amax(lift_success_RL2),
                                  np.amax(nutassemblyround_success_RL2), np.amax(nutassembly_success_RL2),
                                  np.amax(pickplacemilk_success_RL2), np.amax(pickplacecereal_success_RL2),
                                  np.amax(pickplacebread_success_RL2)]

    max_success_train_envs_maml = [np.amax(door_success_MAML), np.amax(lift_success_MAML),
                                   np.amax(nutassemblyround_success_MAML), np.amax(nutassembly_success_MAML),
                                   np.amax(pickplacemilk_success_MAML), np.amax(pickplacecereal_success_MAML),
                                   np.amax(pickplacebread_success_MAML)]

    max_success_test_envs_rl2 = [np.amax(metatest_pickplacecan_success_RL2),
                                 np.amax(metatest_nutassemblysquare_success_RL2),
                                 np.amax(metatest_stack_success_RL2)]

    max_success_test_envs_maml = [np.amax(metatest_pickplacecan_success_MAML),
                                  np.amax(metatest_nutassemblysquare_success_MAML),
                                  np.amax(metatest_stack_success_MAML)]

    max_success_train_envs_rl2 = [element * 100.0 for element in max_success_train_envs_rl2]
    max_success_test_envs_rl2 = [element * 100.0 for element in max_success_test_envs_rl2]
    max_success_train_envs_maml = [element * 100.0 for element in max_success_train_envs_maml]
    max_success_test_envs_maml = [element * 100.0 for element in max_success_test_envs_maml]

    max_return_train_envs_rl2 = [np.amax(door_return_RL2), np.amax(lift_return_RL2),
                                 np.amax(nutassemblyround_return_RL2),
                                 np.amax(nutassembly_return_RL2), np.amax(pickplacemilk_return_RL2),
                                 np.amax(pickplacecereal_return_RL2), np.amax(pickplacebread_return_RL2)]

    max_return_test_envs_rl2 = [np.amax(metatest_pickplacecan_return_RL2),
                                np.amax(metatest_nutassemblysquare_return_RL2),
                                np.amax(metatest_stack_return_RL2)]

    max_return_train_envs_maml = [np.amax(door_return_MAML), np.amax(lift_return_MAML),
                                  np.amax(nutassemblyround_return_MAML), np.amax(nutassembly_return_MAML),
                                  np.amax(pickplacemilk_return_MAML), np.amax(pickplacecereal_return_MAML),
                                  np.amax(pickplacebread_return_MAML)]

    max_return_test_envs_maml = [np.amax(metatest_pickplacecan_return_MAML),
                                 np.amax(metatest_nutassemblysquare_return_MAML),
                                 np.amax(metatest_stack_return_MAML)]

    width_single_bar = 0.4
    fig7, axis7 = plt.subplots(2, 1)
    fig7_2, axis7_2 = plt.subplots(1, 1)
    bar_y_pos1 = np.arange(7)
    bar_y_pos2 = np.arange(3)
    axis7[0].barh(bar_y_pos1, max_success_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    axis7[0].barh(bar_y_pos1 + width_single_bar, max_success_train_envs_maml, height=-width_single_bar,
                  alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7[0].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis7[0].legend()
    legend.get_frame().set_facecolor('white')
    axis7[0].set_xlim([0, 100])
    axis7[0].set_title('Max Success Rate (%)', fontsize=12)
    axis7[1].set_title('Max Return', fontsize=12)
    axis7[1].barh(bar_y_pos1, max_return_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    axis7[1].barh(bar_y_pos1 + width_single_bar, max_return_train_envs_maml, height=-width_single_bar,
                  alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7[1].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis7[1].legend()
    legend.get_frame().set_facecolor('white')
    axis7[1].set_xlim([0, 500])
    # axis7_2[0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
    #                 label='RL2-PPO', align='edge')
    # axis7_2[0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
    #                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    # axis7_2[0].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    # legend = axis7_2[0].legend()
    # legend.get_frame().set_facecolor('white')
    # axis7_2[0].set_xlim([0, 100])
    # axis7_2[0].set_title('Max Success Rate (%)', fontsize=12)
    axis7_2.set_title('Max Return', fontsize=12)
    axis7_2.barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                    label='RL2-PPO', align='edge')
    axis7_2.barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
                    alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7_2.set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis7_2.legend()
    legend.get_frame().set_facecolor('white')
    axis7_2.set_xlim([0, 500])
    # plt.setp(axis7[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis7[1], xlabel='Max Return')
    # plt.setp(axis7_2[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis7_2[1], xlabel='Max Return')
    fig7.suptitle('CRiSE 7 - Max Returns/Success Rates per Meta-Train Task (Sawyer)', fontsize=13)
    fig7_2.suptitle('CRiSE 7 - Max Returns per Meta-Test Task (Sawyer)', fontsize=13)
    fig7.tight_layout()
    fig7_2.tight_layout()
    fig7.savefig('CRiSE7_Sawyer_SuccessReturns_MetaTrain.pgf')
    fig7_2.savefig('CRiSE7_Sawyer_SuccessReturns_MetaTest.pgf')

    # Kuka IIWA14 with locked linear axes AND OSC_POSE controller in the NutAssemblyRound task
    # (in CRiSE 1 MRL-context, RL2_PPO)
    # ----------------------------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE1_NutAssemblyRound/crise1_rl2_ppo_OSCPOSE/progress.csv',
              'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
                                       .astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                     .astype(float))

    # Compute standard deviation for success rates
    average_stdsuccess_rl2ppo = np.sqrt(
        (average_successrate_rl2ppo / 100) * (1 - (average_successrate_rl2ppo / 100))) * 100
    metatest_avg_stdsuccess_rl2ppo = np.sqrt(
        (metatest_avg_successrate_rl2ppo / 100) * (1 - (metatest_avg_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    lowerbound_return_rl2ppo = average_return_rl2ppo - average_stdreturn_rl2ppo
    upperbound_return_rl2ppo = average_return_rl2ppo + average_stdreturn_rl2ppo

    lowerbound_return_rl2ppo = np.where(lowerbound_return_rl2ppo >= 0, lowerbound_return_rl2ppo, 0)
    upperbound_return_rl2ppo = np.where(upperbound_return_rl2ppo <= 500, upperbound_return_rl2ppo, 500)

    lowerbound_success_rl2ppo = average_successrate_rl2ppo - average_stdsuccess_rl2ppo
    upperbound_success_rl2ppo = average_successrate_rl2ppo + average_stdsuccess_rl2ppo

    lowerbound_success_rl2ppo = np.where(lowerbound_success_rl2ppo >= 0, lowerbound_success_rl2ppo, 0)
    upperbound_success_rl2ppo = np.where(upperbound_success_rl2ppo <= 100, upperbound_success_rl2ppo, 100)

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_NutAssemblyRound/singleml_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    naround_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    naround_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    naround_average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    naround_metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    naround_total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    naround_total_testenv_steps_rl2ppo = naround_total_env_steps_rl2ppo[np.where(
        naround_metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    naround_metatest_avg_return_rl2ppo = naround_metatest_avg_return_rl2ppo[
        np.where(naround_metatest_avg_return_rl2ppo != '')].astype(float)
    naround_metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    naround_metatest_avg_successrate_rl2ppo = (
            naround_metatest_avg_successrate_rl2ppo[np.where(naround_metatest_avg_successrate_rl2ppo != '')]
            .astype(float) * 100.0)
    naround_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    naround_metatest_avg_stdreturn_rl2ppo = (
        naround_metatest_avg_stdreturn_rl2ppo[np.where(naround_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Compute standard deviation for success rates
    naround_average_stdsuccess_rl2ppo = np.sqrt(
        (naround_average_successrate_rl2ppo / 100) * (1 - (naround_average_successrate_rl2ppo / 100))) * 100
    naround_metatest_avg_stdsuccess_rl2ppo = np.sqrt(
        (naround_metatest_avg_successrate_rl2ppo / 100) * (1 - (naround_metatest_avg_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    naround_lowerbound_return_rl2ppo = naround_average_return_rl2ppo - naround_average_stdreturn_rl2ppo
    naround_upperbound_return_rl2ppo = naround_average_return_rl2ppo + naround_average_stdreturn_rl2ppo

    naround_lowerbound_return_rl2ppo = np.where(naround_lowerbound_return_rl2ppo >= 0,
                                                naround_lowerbound_return_rl2ppo, 0)
    naround_upperbound_return_rl2ppo = np.where(naround_upperbound_return_rl2ppo <= 500,
                                                naround_upperbound_return_rl2ppo, 500)

    naround_lowerbound_success_rl2ppo = naround_average_successrate_rl2ppo - naround_average_stdsuccess_rl2ppo
    naround_upperbound_success_rl2ppo = naround_average_successrate_rl2ppo + naround_average_stdsuccess_rl2ppo

    naround_lowerbound_success_rl2ppo = np.where(naround_lowerbound_success_rl2ppo >= 0,
                                                 naround_lowerbound_success_rl2ppo, 0)
    naround_upperbound_success_rl2ppo = np.where(naround_upperbound_success_rl2ppo <= 100,
                                                 naround_upperbound_success_rl2ppo, 100)

    # Meta test standard deviation
    metatest_lowerbound_return_rl2ppo = metatest_avg_return_rl2ppo - metatest_avg_stdreturn_rl2ppo
    metatest_upperbound_return_rl2ppo = metatest_avg_return_rl2ppo + metatest_avg_stdreturn_rl2ppo
    naround_metatest_lowerbound_return_rl2ppo = naround_metatest_avg_return_rl2ppo - naround_metatest_avg_stdreturn_rl2ppo
    naround_metatest_upperbound_return_rl2ppo = naround_metatest_avg_return_rl2ppo + naround_metatest_avg_stdreturn_rl2ppo

    metatest_lowerbound_return_rl2ppo = np.where(metatest_lowerbound_return_rl2ppo >= 0,
                                                 metatest_lowerbound_return_rl2ppo, 0)
    metatest_upperbound_return_rl2ppo = np.where(metatest_upperbound_return_rl2ppo <= 500,
                                                 metatest_upperbound_return_rl2ppo, 500)
    naround_metatest_lowerbound_return_rl2ppo = np.where(naround_metatest_lowerbound_return_rl2ppo >= 0,
                                                         naround_metatest_lowerbound_return_rl2ppo, 0)
    naround_metatest_upperbound_return_rl2ppo = np.where(naround_metatest_upperbound_return_rl2ppo <= 500,
                                                         naround_metatest_upperbound_return_rl2ppo, 500)

    metatest_lowerbound_success_rl2ppo = metatest_avg_successrate_rl2ppo - metatest_avg_stdsuccess_rl2ppo
    metatest_upperbound_success_rl2ppo = metatest_avg_successrate_rl2ppo + metatest_avg_stdsuccess_rl2ppo
    naround_metatest_lowerbound_success_rl2ppo = naround_metatest_avg_successrate_rl2ppo - naround_metatest_avg_stdsuccess_rl2ppo
    naround_metatest_upperbound_success_rl2ppo = naround_metatest_avg_successrate_rl2ppo + naround_metatest_avg_stdsuccess_rl2ppo

    metatest_lowerbound_success_rl2ppo = np.where(metatest_lowerbound_success_rl2ppo >= 0,
                                                  metatest_lowerbound_success_rl2ppo, 0)
    metatest_upperbound_success_rl2ppo = np.where(metatest_upperbound_success_rl2ppo <= 100,
                                                  metatest_upperbound_success_rl2ppo, 100)
    naround_metatest_lowerbound_success_rl2ppo = np.where(naround_metatest_lowerbound_success_rl2ppo >= 0,
                                                          naround_metatest_lowerbound_success_rl2ppo, 0)
    naround_metatest_upperbound_success_rl2ppo = np.where(naround_metatest_upperbound_success_rl2ppo <= 100,
                                                          naround_metatest_upperbound_success_rl2ppo, 100)

    # Plot everything
    figx, axisx = plt.subplots(2, 1)
    figx_2, axisx_2 = plt.subplots(2, 1)

    axisx[0].plot(total_env_steps_rl2ppo[0::2], average_return_rl2ppo[0::2], color='green', label='OSC-POSE')
    axisx[0].fill_between(total_env_steps_rl2ppo[0::2], lowerbound_return_rl2ppo[0::2], upperbound_return_rl2ppo[0::2],
                          facecolor='green', alpha=0.1)
    # naround_total_env_steps_rl2ppo = naround_total_env_steps_rl2ppo[0::2]
    # naround_average_return_rl2ppo = naround_average_return_rl2ppo[0::2]

    # Cut off excess data
    limit_j_vel_train = np.where(naround_total_env_steps_rl2ppo <= 8e7)
    limit_j_vel_test = np.where(naround_total_testenv_steps_rl2ppo <= 8e7)

    # naround_lowerbound_return_rl2ppo = naround_lowerbound_return_rl2ppo[0::2]
    # naround_upperbound_return_rl2ppo = naround_upperbound_return_rl2ppo[0::2]

    naround_total_env_steps_rl2ppo = naround_total_env_steps_rl2ppo[limit_j_vel_train]
    naround_average_return_rl2ppo = naround_average_return_rl2ppo[limit_j_vel_train]
    naround_lowerbound_return_rl2ppo = naround_lowerbound_return_rl2ppo[limit_j_vel_train]
    naround_upperbound_return_rl2ppo = naround_upperbound_return_rl2ppo[limit_j_vel_train]
    naround_average_successrate_rl2ppo = naround_average_successrate_rl2ppo[limit_j_vel_train]
    naround_lowerbound_success_rl2ppo = naround_lowerbound_success_rl2ppo[limit_j_vel_train]
    naround_upperbound_success_rl2ppo = naround_upperbound_success_rl2ppo[limit_j_vel_train]
    naround_metatest_lowerbound_return_rl2ppo = naround_metatest_lowerbound_return_rl2ppo[limit_j_vel_test]
    naround_metatest_upperbound_return_rl2ppo = naround_metatest_upperbound_return_rl2ppo[limit_j_vel_test]
    naround_metatest_lowerbound_success_rl2ppo = naround_metatest_lowerbound_success_rl2ppo[limit_j_vel_test]
    naround_metatest_upperbound_success_rl2ppo = naround_metatest_upperbound_success_rl2ppo[limit_j_vel_test]
    naround_metatest_avg_return_rl2ppo = naround_metatest_avg_return_rl2ppo[limit_j_vel_test]
    naround_metatest_avg_successrate_rl2ppo = naround_metatest_avg_successrate_rl2ppo[limit_j_vel_test]

    axisx[0].plot(naround_total_env_steps_rl2ppo, naround_average_return_rl2ppo,
                  color='orange', label='J-VEL')
    axisx[0].fill_between(naround_total_env_steps_rl2ppo,
                          naround_lowerbound_return_rl2ppo,
                          naround_upperbound_return_rl2ppo, facecolor='orange', alpha=0.1)
    axisx[0].set_ylim([-5, 520])
    legend = axisx[0].legend()
    legend.get_frame().set_facecolor('white')

    # naround_average_successrate_rl2ppo = naround_average_successrate_rl2ppo[0::2]
    # naround_lowerbound_success_rl2ppo = naround_lowerbound_success_rl2ppo[0::2]
    # naround_upperbound_success_rl2ppo = naround_upperbound_success_rl2ppo[0::2]
    axisx[1].plot(naround_total_env_steps_rl2ppo,
                  naround_average_successrate_rl2ppo, color='orange', label='J-VEL')
    axisx[1].fill_between(naround_total_env_steps_rl2ppo,
                          naround_lowerbound_success_rl2ppo,
                          naround_upperbound_success_rl2ppo, facecolor='orange', alpha=0.1)
    axisx[1].plot(total_env_steps_rl2ppo[0::2], average_successrate_rl2ppo[0::2], color='green', label='OSC-POSE')
    axisx[1].fill_between(total_env_steps_rl2ppo[0::2], lowerbound_success_rl2ppo[0::2],
                          upperbound_success_rl2ppo[0::2],
                          facecolor='green', alpha=0.1)
    axisx[1].set_ylim([-5, 105])
    legend = axisx[1].legend()
    legend.get_frame().set_facecolor('white')

    axisx_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='OSC-POSE')
    axisx_2[0].plot(naround_total_testenv_steps_rl2ppo[limit_j_vel_test],
                    naround_metatest_avg_return_rl2ppo, color='orange', label='J-VEL')
    axisx_2[0].fill_between(naround_total_testenv_steps_rl2ppo[limit_j_vel_test],
                            naround_metatest_lowerbound_return_rl2ppo,
                            naround_metatest_upperbound_return_rl2ppo, facecolor='orange', alpha=0.1)
    axisx_2[0].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_return_rl2ppo,
                            metatest_upperbound_return_rl2ppo,
                            facecolor='green', alpha=0.1)
    axisx_2[0].set_ylim([-5, 520])
    legend = axisx_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axisx_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', label='OSC-POSE')
    axisx_2[1].plot(naround_total_testenv_steps_rl2ppo[limit_j_vel_test],
                    naround_metatest_avg_successrate_rl2ppo, color='orange', label='J-VEL')
    axisx_2[1].fill_between(naround_total_testenv_steps_rl2ppo[limit_j_vel_test],
                            naround_metatest_lowerbound_success_rl2ppo,
                            naround_metatest_upperbound_success_rl2ppo, facecolor='orange', alpha=0.1)
    axisx_2[1].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_success_rl2ppo,
                            metatest_upperbound_success_rl2ppo,
                            facecolor='green', alpha=0.1)
    axisx_2[1].set_ylim([-5, 105])
    legend = axisx_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axisx[0], ylabel='Average Return')
    plt.setp(axisx[1], ylabel='Success Rate (%)')
    plt.setp(axisx_2[0], ylabel='Average Return')
    plt.setp(axisx_2[1], ylabel='Success Rate (%)')
    plt.setp(axisx[1], xlabel='Total Environment Steps')
    plt.setp(axisx_2[1], xlabel='Total Environment Steps')
    figx.suptitle('CRiSE 1 - OSC-POSE vs. J-VEL on IIWA14 (Meta-Training)', fontsize=14)
    figx_2.suptitle('CRiSE 1 - OSC-POSE vs. J-VEL on IIWA14 (Meta-Test)', fontsize=14)
    figx.savefig('CRiSE1_IIWA14_NARoundOSCPOSE_Train.pgf')
    figx_2.savefig('CRiSE1_IIWA14_NARoundOSCPOSE_Test.pgf')

    # Kuka IIWA14 with locked linear axes in CRiSE 3 with adapted grasp reward function (MAML_TRPO and RL2_PPO)
    # ---------------------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE3/crise3_maml_trpo/progress.csv',
              'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return MAML_TRPO
    average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps MAML_TRPO
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_testenv_steps_mamltrpo = total_env_steps_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(int)

    # Get meta test average return, success rate and standard return MAML_TRPO
    metatest_avg_return_mamltrpo = metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(
        float)
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_mamltrpo = metatest_avg_successrate_mamltrpo[np.where(
        metatest_avg_successrate_mamltrpo != '')].astype(float) * 100.0
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_mamltrpo = metatest_avg_stdreturn_mamltrpo[np.where(
        metatest_avg_stdreturn_mamltrpo != '')].astype(float)

    # Compute standard deviation for success rates
    average_stdsuccess_mamltrpo = np.sqrt(
        (average_successrate_mamltrpo / 100) * (1 - (average_successrate_mamltrpo / 100))) * 100
    metatest_avg_stdsuccess_mamltrpo = np.sqrt(
        (metatest_avg_successrate_mamltrpo / 100) * (1 - (metatest_avg_successrate_mamltrpo / 100))) * 100

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_CRiSE3/CRiSE3_rl2_ppo/progress.csv',
              'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (meta testing in RL2PPO is executed every ten epochs,
    # therefore blank entries in the csv exist which are not castable to float values -> get "valid" indices in meta test
    # data and store them separately)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
                                       .astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                     .astype(float))

    average_stdsuccess_rl2ppo = np.sqrt(
        (average_successrate_rl2ppo / 100) * (1 - (average_successrate_rl2ppo / 100))) * 100
    metatest_avg_stdsuccess_rl2ppo = np.sqrt(
        (metatest_avg_successrate_rl2ppo / 100) * (1 - (metatest_avg_successrate_rl2ppo / 100))) * 100

    # Define upper and lower bounds for printing the standard deviation in the success and return plots, limit the
    # values to [0, 100]
    lowerbound_return_mamltrpo = average_return_mamltrpo - average_stdreturn_mamltrpo
    upperbound_return_mamltrpo = average_return_mamltrpo + average_stdreturn_mamltrpo
    lowerbound_return_rl2ppo = average_return_rl2ppo - average_stdreturn_rl2ppo
    upperbound_return_rl2ppo = average_return_rl2ppo + average_stdreturn_rl2ppo

    lowerbound_return_mamltrpo = np.where(lowerbound_return_mamltrpo >= 0, lowerbound_return_mamltrpo, 0)
    upperbound_return_mamltrpo = np.where(upperbound_return_mamltrpo <= 500, upperbound_return_mamltrpo, 500)
    lowerbound_return_rl2ppo = np.where(lowerbound_return_rl2ppo >= 0, lowerbound_return_rl2ppo, 0)
    upperbound_return_rl2ppo = np.where(upperbound_return_rl2ppo <= 500, upperbound_return_rl2ppo, 500)

    lowerbound_success_mamltrpo = average_successrate_mamltrpo - average_stdsuccess_mamltrpo
    upperbound_success_mamltrpo = average_successrate_mamltrpo + average_stdsuccess_mamltrpo
    lowerbound_success_rl2ppo = average_successrate_rl2ppo - average_stdsuccess_rl2ppo
    upperbound_success_rl2ppo = average_successrate_rl2ppo + average_stdsuccess_rl2ppo

    lowerbound_success_mamltrpo = np.where(lowerbound_success_mamltrpo >= 0, lowerbound_success_mamltrpo, 0)
    upperbound_success_mamltrpo = np.where(upperbound_success_mamltrpo <= 100, upperbound_success_mamltrpo, 100)
    lowerbound_success_rl2ppo = np.where(lowerbound_success_rl2ppo >= 0, lowerbound_success_rl2ppo, 0)
    upperbound_success_rl2ppo = np.where(upperbound_success_rl2ppo <= 100, upperbound_success_rl2ppo, 100)

    # Meta test standard deviation
    metatest_lowerbound_return_mamltrpo = metatest_avg_return_mamltrpo - metatest_avg_stdreturn_mamltrpo
    metatest_upperbound_return_mamltrpo = metatest_avg_return_mamltrpo + metatest_avg_stdreturn_mamltrpo
    metatest_lowerbound_return_rl2ppo = metatest_avg_return_rl2ppo - metatest_avg_stdreturn_rl2ppo
    metatest_upperbound_return_rl2ppo = metatest_avg_return_rl2ppo + metatest_avg_stdreturn_rl2ppo

    metatest_lowerbound_return_mamltrpo = np.where(metatest_lowerbound_return_mamltrpo >= 0,
                                                   metatest_lowerbound_return_mamltrpo, 0)
    metatest_upperbound_return_mamltrpo = np.where(metatest_upperbound_return_mamltrpo <= 500,
                                                   metatest_upperbound_return_mamltrpo, 500)
    metatest_lowerbound_return_rl2ppo = np.where(metatest_lowerbound_return_rl2ppo >= 0,
                                                 metatest_lowerbound_return_rl2ppo, 0)
    metatest_upperbound_return_rl2ppo = np.where(metatest_upperbound_return_rl2ppo <= 500,
                                                 metatest_upperbound_return_rl2ppo, 500)

    metatest_lowerbound_success_mamltrpo = metatest_avg_successrate_mamltrpo - metatest_avg_stdsuccess_mamltrpo
    metatest_upperbound_success_mamltrpo = metatest_avg_successrate_mamltrpo + metatest_avg_stdsuccess_mamltrpo
    metatest_lowerbound_success_rl2ppo = metatest_avg_successrate_rl2ppo - metatest_avg_stdsuccess_rl2ppo
    metatest_upperbound_success_rl2ppo = metatest_avg_successrate_rl2ppo + metatest_avg_stdsuccess_rl2ppo

    metatest_lowerbound_success_mamltrpo = np.where(metatest_lowerbound_success_mamltrpo >= 0,
                                                    metatest_lowerbound_success_mamltrpo, 0)
    metatest_upperbound_success_mamltrpo = np.where(metatest_upperbound_success_mamltrpo <= 100,
                                                    metatest_upperbound_success_mamltrpo, 100)
    metatest_lowerbound_success_rl2ppo = np.where(metatest_lowerbound_success_rl2ppo >= 0,
                                                  metatest_lowerbound_success_rl2ppo, 0)
    metatest_upperbound_success_rl2ppo = np.where(metatest_upperbound_success_rl2ppo <= 100,
                                                  metatest_upperbound_success_rl2ppo, 100)

    # Plot everything
    figy, axisy = plt.subplots(2, 1)
    figy_2, axisy_2 = plt.subplots(2, 1)

    axisy[0].plot(total_env_steps_mamltrpo[0::2], average_return_mamltrpo[0::2], color='red', label='MAML-TRPO')
    axisy[0].plot(total_env_steps_rl2ppo[0::2], average_return_rl2ppo[0::2], color='green', label='RL2-PPO')
    axisy[0].fill_between(total_env_steps_mamltrpo[0::2], lowerbound_return_mamltrpo[0::2],
                          upperbound_return_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axisy[0].fill_between(total_env_steps_rl2ppo[0::2], lowerbound_return_rl2ppo[0::2],
                          upperbound_return_rl2ppo[0::2], facecolor='green', alpha=0.1)
    axisy[0].set_ylim([-5, 520])
    legend = axisy[0].legend()
    legend.get_frame().set_facecolor('white')

    axisy[1].plot(total_env_steps_mamltrpo[0::2], average_successrate_mamltrpo[0::2], color='red', label='MAML-TRPO')
    axisy[1].plot(total_env_steps_rl2ppo[0::2], average_successrate_rl2ppo[0::2], color='green', label='RL2-PPO')
    axisy[1].fill_between(total_env_steps_mamltrpo[0::2], lowerbound_success_mamltrpo[0::2],
                          upperbound_success_mamltrpo[0::2], facecolor='red', alpha=0.1)
    axisy[1].fill_between(total_env_steps_rl2ppo[0::2], lowerbound_success_rl2ppo[0::2],
                          upperbound_success_rl2ppo[0::2], facecolor='green', alpha=0.1)
    axisy[1].set_ylim([-5, 105])
    legend = axisy[1].legend()
    legend.get_frame().set_facecolor('white')

    axisy_2[0].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axisy_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axisy_2[0].fill_between(total_testenv_steps_mamltrpo, metatest_lowerbound_return_mamltrpo,
                            metatest_upperbound_return_mamltrpo, facecolor='red', alpha=0.1)
    axisy_2[0].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_return_rl2ppo,
                            metatest_upperbound_return_rl2ppo, facecolor='green', alpha=0.1)
    axisy_2[0].set_ylim([-5, 520])
    legend = axisy_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axisy_2[1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axisy_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axisy_2[1].fill_between(total_testenv_steps_mamltrpo, metatest_lowerbound_success_mamltrpo,
                            metatest_upperbound_success_mamltrpo, facecolor='red', alpha=0.1)
    axisy_2[1].fill_between(total_testenv_steps_rl2ppo, metatest_lowerbound_success_rl2ppo,
                            metatest_upperbound_success_rl2ppo, facecolor='green', alpha=0.1)
    axisy_2[1].set_ylim([-5, 105])
    legend = axisy_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axisy[0], ylabel='Average Return')
    plt.setp(axisy[1], ylabel='Success Rate (%)')
    plt.setp(axisy_2[0], ylabel='Average Return')
    plt.setp(axisy_2[1], ylabel='Success Rate (%)')
    plt.setp(axisy[1], xlabel='Total Environment Steps')
    plt.setp(axisy_2[1], xlabel='Total Environment Steps')
    figy.suptitle('Meta 3 - Adapted Reward Function (Meta-Training)', fontsize=14)
    figy_2.suptitle('Meta 3 - Adapted Reward Function (Meta-Test)', fontsize=14)
    figy.savefig('Meta3_IIWA14_newGraspReward_Train.pgf')
    figy_2.savefig('Meta3_IIWA14_newGraspReward_Test.pgf')
    # plt.show()


if __name__ == "__main__":
    with plt.style.context('ggplot'):
        plot_all()
