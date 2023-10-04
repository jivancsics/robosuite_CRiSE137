"""File for plotting all the experimental results"""

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

    # Rethink Robotics Sawyer in Robosuite's blocklifting task (Meta 1) with MAML_TRPO and RL2_PPO
    # --------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_Meta1_Blocklifting/singleml_maml_trpo/progress.csv', 'r') as file:
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

    # Get the number of the corresponding environment steps MAML_TRPO
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)


    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_Meta1_Blocklifting/singleml_rl2_ppo/progress.csv', 'r') as file:
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


    # Plot everything
    fig, axis = plt.subplots(2, 1)
    fig_2, axis_2 = plt.subplots(2, 1)

    axis[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis[0].set_ylim([0, 500])
    legend = axis[0].legend()
    legend.get_frame().set_facecolor('white')

    axis[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis[1].set_ylim([0, 105])
    legend = axis[1].legend()
    legend.get_frame().set_facecolor('white')

    axis_2[0].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis_2[0].set_ylim([0, 500])
    legend = axis_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis_2[1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                   label='MAML-TRPO')
    axis_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                   label='RL2-PPO')
    axis_2[1].set_ylim([0, 105])
    legend = axis_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis[0], ylabel='Average Return')
    plt.setp(axis[1], ylabel='Success Rate (%)')
    plt.setp(axis_2[0], ylabel='Average Return')
    plt.setp(axis_2[1], ylabel='Success Rate (%)')
    plt.setp(axis[1], xlabel='Total Environment Steps')
    plt.setp(axis_2[1], xlabel='Total Environment Steps')
    fig.suptitle('Meta 1 - Lift Block with Sawyer (Meta Training)', fontsize=14)
    fig_2.suptitle('Meta 1 - Lift Block with Sawyer (Meta Test)', fontsize=14)
    fig.savefig('Meta1_Sawyer_LiftBlock_Train.pgf')
    fig_2.savefig('Meta1_Sawyer_LiftBlock_Test.pgf')


    # Sawyer in Meta-Worlds's ML10 with MAML_TRPO, RL2_PPO and PEARL
    # ------------------------------------------------------------------------
    
    data_rows = []
    
    with open('Experiment_Data/MetaWorld_Sawyer_ML10/ml10_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)
    
    data_rows = np.array(data_rows)
    
    # Get average return, success rate and standard return MAML_TRPO
    average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')].astype(float)
    
    # Get the number of the corresponding environment steps MAML_TRPO (number of results train != test !!!)
    metatest_avg_return_mamltrpo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_mamltrpo = total_env_steps_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(int)
    
    # Get meta test average return, success rate and standard return MAML_TRPO
    metatest_avg_return_mamltrpo = (metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')]
                                    .astype(float))
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(
        metatest_avg_successrate_mamltrpo != '')].astype(float)) * 100.0
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_mamltrpo = metatest_avg_stdreturn_mamltrpo[np.where(metatest_avg_stdreturn_mamltrpo != '')]
    
    
    # Get the latest ML10 RL2 experiment data stored in progress.csv

    data_rows = []

    with open('Experiment_Data/MetaWorld_Sawyer_ML10/ml10_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    average_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (number of results train != test !!!)
    env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    # Meta Test data is unforunately missing in the progress.csv (due to the power loss in the office)

    data_rows = []

    with open('Experiment_Data/MetaWorld_Sawyer_ML10/ml10_rl2_ppo/progress_pre1.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Get average return, success rate and standard return RL2_PPO
    total_average_return_rl2ppo = data_rows[:, header.index('Average/AverageReturn')].astype(float)
    total_average_successrate_rl2ppo = data_rows[:, header.index('Average/SuccessRate')].astype(float) * 100.0
    totalaverage_stdreturn_rl2ppo = data_rows[:, header.index('Average/StdReturn')].astype(float)

    # Get the number of the corresponding environment steps RL2_PPO (number of results train != test !!!)
    metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

    # Get meta test average return, success rate and standard return RL2_PPO
    total_metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(
        float)
    metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    total_metatest_avg_successrate_rl2ppo = \
        (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')].astype(float) * 100.0)
    metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    total_metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
                                           .astype(float))

    data_rows = []

    with open('Experiment_Data/MetaWorld_Sawyer_ML10/ml10_rl2_ppo/progress_pre2.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Append average return, success rate and standard return RL2_PPO to previously set np array
    total_average_return_rl2ppo = (
        np.append(total_average_return_rl2ppo, data_rows[:, header.index('Average/AverageReturn')].astype(float)))
    total_average_successrate_rl2ppo = (np.append(total_average_successrate_rl2ppo,
                                                  data_rows[:, header.index('Average/SuccessRate')].astype(
                                                      float) * 100.0))
    totalaverage_stdreturn_rl2ppo = (
        np.append(totalaverage_stdreturn_rl2ppo, data_rows[:, header.index('Average/StdReturn')].astype(float)))

    # Get the number of the corresponding environment steps RL2_PPO (number of results train != test !!!)
    buffer_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_env_steps_rl2ppo = np.append(total_env_steps_rl2ppo, buffer_env_steps_rl2ppo)

    # Meta Test data missing from here onwards; code saved for later
    # metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
    # total_testenv_steps_rl2ppo = np.append(total_testenv_steps_rl2ppo,
    #                                        buffer_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int))

    # Get meta test average return, success rate and standard return RL2_PPO
    # total_metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
    # metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    # total_metatest_avg_successrate_rl2ppo = np.append(total_metatest_avg_successrate_rl2ppo,
    #                                                   (metatest_avg_successrate_rl2ppo[np.where(
    #                                                       metatest_avg_successrate_rl2ppo != '')].astype(float) * 100.0))
    # metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    # total_metatest_avg_stdreturn_rl2ppo = np.append(total_metatest_avg_stdreturn_rl2ppo,
    #                                                 (metatest_avg_stdreturn_rl2ppo[np.where(
    #                                                     metatest_avg_stdreturn_rl2ppo != '')].astype(float)))

    data_rows = []

    with open('Experiment_Data/MetaWorld_Sawyer_ML10/ml10_rl2_ppo/progress_pre3.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows)

    # Append average return, success rate and standard return RL2_PPO to previously set np array
    total_average_return_rl2ppo = (
        np.append(total_average_return_rl2ppo, data_rows[:, header.index('Average/AverageReturn')].astype(float)))
    total_average_successrate_rl2ppo = (np.append(total_average_successrate_rl2ppo,
                                                  data_rows[:, header.index('Average/SuccessRate')].astype(
                                                      float) * 100.0))
    totalaverage_stdreturn_rl2ppo = (
        np.append(totalaverage_stdreturn_rl2ppo, data_rows[:, header.index('Average/StdReturn')].astype(float)))

    # Get the number of the corresponding environment steps RL2_PPO (number of results train != test !!!)
    buffer_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    total_env_steps_rl2ppo = np.append(total_env_steps_rl2ppo, buffer_env_steps_rl2ppo)

    # Concatenate last stored RL2 progress with previous progress
    total_average_return_rl2ppo = np.append(total_average_return_rl2ppo, average_return_rl2ppo)
    total_average_successrate_rl2ppo = (np.append(total_average_successrate_rl2ppo, average_successrate_rl2ppo))
    total_average_stdreturn_rl2ppo = np.append(totalaverage_stdreturn_rl2ppo, average_stdreturn_rl2ppo)
    total_env_steps_rl2ppo = np.append(total_env_steps_rl2ppo, env_steps_rl2ppo)
    
    data_rows = []
    
    with open('Experiment_Data/MetaWorld_Sawyer_ML10/ml10_pearl/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)
    
    data_rows = np.array(data_rows)
    
    
    # Get the number of the corresponding environment steps PEARL
    total_env_steps_pearl = data_rows[:, header.index('TotalEnvSteps')].astype(int)
    
    
    # Get meta test average return, success rate and standard return RL2_PPO
    metatest_avg_return_pearl = data_rows[:, header.index('MetaTest/Average/AverageReturn')].astype(float)
    metatest_avg_successrate_pearl = data_rows[:, header.index('MetaTest/Average/SuccessRate')].astype(float) * 100.0
    metatest_avg_stdreturn_pearl = data_rows[:, header.index('MetaTest/Average/StdReturn')].astype(float)
    
    
    # Plot everything
    fig2, axis2 = plt.subplots(2, 1)
    fig2_2, axis2_2 = plt.subplots(2, 1)

    axis2[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis2[0].plot(total_env_steps_rl2ppo, total_average_return_rl2ppo, color='green', label='RL2-PPO')
    axis2[0].set_ylim([0, 4000])
    legend = axis2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis2[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis2[1].plot(total_env_steps_rl2ppo, total_average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis2[1].set_ylim([0, 100])
    legend = axis2[1].legend()
    legend.get_frame().set_facecolor('white')

    axis2_2[0].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis2_2[0].plot(total_testenv_steps_rl2ppo, total_metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis2_2[0].plot(total_env_steps_pearl, metatest_avg_return_pearl, color='blue', label='PEARL')
    axis2_2[0].set_ylim([0, 4000])
    legend = axis2_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis2_2[1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis2_2[1].plot(total_testenv_steps_rl2ppo, total_metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis2_2[1].plot(total_env_steps_pearl, metatest_avg_successrate_pearl, color='blue', label='PEARL')
    axis2_2[1].set_ylim([0, 100])
    legend = axis2_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis2[0], ylabel='Average Return')
    plt.setp(axis2[1], ylabel='Success Rate (%)')
    plt.setp(axis2_2[0], ylabel='Average Return')
    plt.setp(axis2_2[1], ylabel='Success Rate (%)')
    plt.setp(axis2[1], xlabel='Total Environment Steps')
    plt.setp(axis2_2[1], xlabel='Total Environment Steps')
    fig2.suptitle('ML10 with Sawyer (Meta Training)', fontsize=14)
    fig2_2.suptitle('ML10 with Sawyer (Meta Test)', fontsize=14)
    fig2.savefig('ML10_MetaWorld_Sawyer_Train.pgf')
    fig2_2.savefig('ML10_MetaWorld_Sawyer_Test.pgf')


    # Kuka IIWA14 with no linear axis in Robosuite's blocklifting task (Meta 1) with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_Blocklifting/singleml_maml_trpo/progress.csv', 'r') as file:
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

    # Get the number of the corresponding environment steps MAML_TRPO
    total_env_steps_mamltrpo = data_rows[:, header.index('TotalEnvSteps')].astype(int)

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_Blocklifting/singleml_rl2_ppo/progress.csv', 'r') as file:
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

    # Plot everything
    fig3, axis3 = plt.subplots(2, 1)
    fig3_2, axis3_2 = plt.subplots(2, 1)

    axis3[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis3[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis3[0].set_ylim([0, 500])
    legend = axis3[0].legend()
    legend.get_frame().set_facecolor('white')

    axis3[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis3[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis3[1].set_ylim([0, 105])
    legend = axis3[1].legend()
    legend.get_frame().set_facecolor('white')

    axis3_2[0].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis3_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis3_2[0].set_ylim([0, 500])
    legend = axis3_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis3_2[1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis3_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis3_2[1].set_ylim([0, 105])
    legend = axis3_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis3[0], ylabel='Average Return')
    plt.setp(axis3[1], ylabel='Success Rate (%)')
    plt.setp(axis3_2[0], ylabel='Average Return')
    plt.setp(axis3_2[1], ylabel='Success Rate (%)')
    plt.setp(axis3[1], xlabel='Total Environment Steps')
    plt.setp(axis3_2[1], xlabel='Total Environment Steps')
    fig3.suptitle('Meta 1 - Lift Block with IIWA14 (Meta Training)', fontsize=14)
    fig3_2.suptitle('Meta 1 - Lift Block with IIWA14 (Meta Test)', fontsize=14)
    fig3.savefig('Meta1_IIWA14_LiftBlock_Train.pgf')
    fig3_2.savefig('Meta1_IIWA14_LiftBlock_Test.pgf')


    # Kuka IIWA14 with no linear axis in ALL Robosuite Meta 1 tasks with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    # Blocklifting
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_Blocklifting/singleml_maml_trpo/progress.csv', 'r') as file:
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

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_Blocklifting/singleml_rl2_ppo/progress.csv', 'r') as file:
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


    # Nut Assembly Round
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_NutAssemblyRound/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    naround_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    naround_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    naround_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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
    """

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

    # Open Door
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_Door-Open/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    opendoor_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    opendoor_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    opendoor_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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
    """

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_Door-Open/singleml_rl2_ppo/progress.csv', 'r') as file:
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

    # Nut assembly mixed
    """
    data_rows = []
    
    with open('Experiment_Data/Robosuite_IIWA14_Meta1_NutAssemblyMixed/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    namixed_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    namixed_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    namixed_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_NutAssemblyMixed/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    """

    # Pick place milk
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceMilk/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    ppmilk_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    ppmilk_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    ppmilk_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceMilk/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    """
    # NEXT FIGURES WITH THE SECOND FIVE TASKS
    #----------------------------------------

    # Pick place bread
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceBread/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    ppbread_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    ppbread_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    ppbread_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceBread/singleml_rl2_ppo/progress.csv', 'r') as file:
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

    # Pick place cereal
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceCereal/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    ppcereal_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    ppcereal_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    ppcereal_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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


    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceCereal/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    """

    # Stack blocks
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_StackBlocks/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    stackblocks_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    stackblocks_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    stackblocks_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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
    """

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_StackBlocks/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    stackblocks_metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    stackblocks_metatest_avg_stdreturn_rl2ppo = (
        stackblocks_metatest_avg_stdreturn_rl2ppo[np.where(stackblocks_metatest_avg_stdreturn_rl2ppo != '')]
        .astype(float))

    # Pick place can
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceCan/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    ppcan_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    ppcan_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    ppcan_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_PickPlaceCan/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    """

    # Nut assembly square
    """
    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_NutAssemblySquare/singleml_maml_trpo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for data_row in reader:
            data_rows.append(data_row)

    data_rows = np.array(data_rows).astype(float)

    # Get average return, success rate and standard return MAML_TRPO
    nasquare_average_return_mamltrpo = data_rows[:, header.index('Average/AverageReturn')]
    nasquare_average_successrate_mamltrpo = data_rows[:, header.index('Average/SuccessRate')] * 100.0
    nasquare_average_stdreturn_mamltrpo = data_rows[:, header.index('Average/StdReturn')]

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
    """

    data_rows = []

    with open('Experiment_Data/Robosuite_IIWA14_Meta1_NutAssemblySquare/singleml_rl2_ppo/progress.csv', 'r') as file:
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


    # Plot everything
    fig, axis = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))    # (8.81036, 5.8476)
    fig_2, axis_2 = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))
    fig_3, axis_3 = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))
    fig_4, axis_4 = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(9, 6))


    # Lift block
    axis[0, 0].plot(blocklift_total_env_steps_mamltrpo, blocklift_average_return_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis[0, 0].plot(blocklift_total_env_steps_rl2ppo, blocklift_average_return_rl2ppo, color='green', label='RL2-PPO')
    axis[0, 0].set_ylim([0, 520])
    legend = axis[0, 0].legend()
    legend.get_frame().set_facecolor('white')

    axis[1, 0].plot(blocklift_total_env_steps_mamltrpo, blocklift_average_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis[1, 0].plot(blocklift_total_env_steps_rl2ppo, blocklift_average_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis[1, 0].set_ylim([0, 105])
    legend = axis[1, 0].legend()
    legend.get_frame().set_facecolor('white')

    # Nut assembly round
    # axis[0, 1].plot(naround_total_env_steps_mamltrpo, naround_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis[0, 1].plot(naround_total_env_steps_rl2ppo, naround_average_return_rl2ppo, color='green', label='RL2-PPO')

    # axis[1, 1].plot(naround_total_env_steps_mamltrpo, naround_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis[1, 1].plot(naround_total_env_steps_rl2ppo, naround_average_successrate_rl2ppo, color='green',
                    label='RL2-PPO')

    # Open door
    # axis[0, 2].plot(opendoor_total_env_steps_mamltrpo, opendoor_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis[0, 2].plot(opendoor_total_env_steps_rl2ppo, opendoor_average_return_rl2ppo, color='green', label='RL2-PPO')

    # axis[1, 2].plot(opendoor_total_env_steps_mamltrpo, opendoor_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis[1, 2].plot(opendoor_total_env_steps_rl2ppo, opendoor_average_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis[1, 2].xaxis.get_offset_text().set_visible(False)

    # Nut assembly mixed
    # axis[0, 3].plot(namixed_total_env_steps_mamltrpo, namixed_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis[0, 3].plot(namixed_total_env_steps_rl2ppo, namixed_average_return_rl2ppo, color='green', label='RL2-PPO')

    # axis[1, 3].plot(namixed_total_env_steps_mamltrpo, namixed_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis[1, 3].plot(namixed_total_env_steps_rl2ppo, namixed_average_successrate_rl2ppo, color='green',
    #                 label='RL2-PPO')

    # Pick place milk
    # axis[0, 4].plot(ppmilk_total_env_steps_mamltrpo, ppmilk_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis[0, 4].plot(ppmilk_total_env_steps_rl2ppo, ppmilk_average_return_rl2ppo, color='green', label='RL2-PPO')
    #
    # axis[1, 4].plot(ppmilk_total_env_steps_mamltrpo, ppmilk_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis[1, 4].plot(ppmilk_total_env_steps_rl2ppo, ppmilk_average_successrate_rl2ppo, color='green',
    #                 label='RL2-PPO')

    # Pick place bread
    # axis_3[0, 0].plot(ppbread_total_env_steps_mamltrpo, ppbread_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis_3[0, 0].plot(ppbread_total_env_steps_rl2ppo, ppbread_average_return_rl2ppo, color='green', label='RL2-PPO')
    axis_3[0, 0].set_ylim([0, 520])
    legend = axis_3[0, 0].legend()
    legend.get_frame().set_facecolor('white')

    # axis_3[1, 0].plot(ppbread_total_env_steps_mamltrpo, ppbread_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis_3[1, 0].plot(ppbread_total_env_steps_rl2ppo, ppbread_average_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis_3[1, 0].set_ylim([0, 105])
    legend = axis_3[1, 0].legend()
    legend.get_frame().set_facecolor('white')

    # Pick place cereal
    # axis_3[0, 1].plot(ppcereal_total_env_steps_mamltrpo, ppcereal_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis_3[0, 1].plot(ppcereal_total_env_steps_rl2ppo, ppcereal_average_return_rl2ppo, color='green', label='RL2-PPO')
    #
    # axis_3[1, 1].plot(ppcereal_total_env_steps_mamltrpo, ppcereal_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis_3[1, 1].plot(ppcereal_total_env_steps_rl2ppo, ppcereal_average_successrate_rl2ppo, color='green',
    #                 label='RL2-PPO')

    # Stack blocks
    # axis_3[0, 2].plot(stackblocks_total_env_steps_mamltrpo, stackblocks_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis_3[0, 2].plot(stackblocks_total_env_steps_rl2ppo, stackblocks_average_return_rl2ppo, color='green', label='RL2-PPO')

    # axis_3[1, 2].plot(stackblocks_total_env_steps_mamltrpo, stackblocks_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis_3[1, 2].plot(stackblocks_total_env_steps_rl2ppo, stackblocks_average_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis_3[1, 2].xaxis.get_offset_text().set_visible(False)

    # Pick place can
    # axis_3[0, 3].plot(ppcan_total_env_steps_mamltrpo, ppcan_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis_3[0, 3].plot(ppcan_total_env_steps_rl2ppo, ppcan_average_return_rl2ppo, color='green', label='RL2-PPO')
    #
    # axis_3[1, 3].plot(ppcan_total_env_steps_mamltrpo, ppcan_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    # axis_3[1, 3].plot(ppcan_total_env_steps_rl2ppo, ppcan_average_successrate_rl2ppo, color='green',
    #                 label='RL2-PPO')

    # Nut assembly square
    # axis_3[0, 4].plot(nasquare_total_env_steps_mamltrpo, nasquare_average_return_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis_3[0, 4].plot(nasquare_total_env_steps_rl2ppo, nasquare_average_return_rl2ppo, color='green', label='RL2-PPO')

    # axis_3[1, 4].plot(nasquare_total_env_steps_mamltrpo, nasquare_average_successrate_mamltrpo, color='red',
    #                 label='MAML-TRPO')
    axis_3[1, 4].plot(nasquare_total_env_steps_rl2ppo, nasquare_average_successrate_rl2ppo, color='green',
                    label='RL2-PPO')


    # Lift block meta test
    axis_2[0, 0].plot(blocklift_total_testenv_steps_mamltrpo, blocklift_metatest_avg_return_mamltrpo, color='red',
                      label='MAML-TRPO')
    axis_2[0, 0].plot(blocklift_total_testenv_steps_rl2ppo, blocklift_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')
    axis_2[0, 0].set_ylim([0, 520])
    legend = axis_2[0, 0].legend()
    legend.get_frame().set_facecolor('white')

    axis_2[1, 0].plot(blocklift_total_testenv_steps_mamltrpo, blocklift_metatest_avg_successrate_mamltrpo, '+',
                      color='red', label='MAML-TRPO')
    axis_2[1, 0].plot(blocklift_total_testenv_steps_rl2ppo, blocklift_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    axis_2[1, 0].set_ylim([-5, 105])
    legend = axis_2[1, 0].legend()
    legend.get_frame().set_facecolor('white')

    # Nut assembly round meta test
    # axis_2[0, 1].plot(naround_total_testenv_steps_mamltrpo, naround_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    axis_2[0, 1].plot(naround_total_testenv_steps_rl2ppo, naround_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    # axis_2[1, 1].plot(naround_total_testenv_steps_mamltrpo, naround_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    axis_2[1, 1].plot(naround_total_testenv_steps_rl2ppo, naround_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')

    # Open door meta test
    # axis_2[0, 2].plot(opendoor_total_testenv_steps_mamltrpo, opendoor_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    axis_2[0, 2].plot(opendoor_total_testenv_steps_rl2ppo, opendoor_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    # axis_2[1, 2].plot(opendoor_total_testenv_steps_mamltrpo, opendoor_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    axis_2[1, 2].plot(opendoor_total_testenv_steps_rl2ppo, opendoor_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    axis_2[1, 2].xaxis.get_offset_text().set_visible(False)

    # Nut assembly mixed meta test
    # axis_2[0, 3].plot(namixed_total_testenv_steps_mamltrpo, namixed_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    # axis_2[0, 3].plot(namixed_total_testenv_steps_rl2ppo, namixed_metatest_avg_return_rl2ppo, color='green',
    #                   label='RL2-PPO')

    # axis_2[1, 3].plot(namixed_total_testenv_steps_mamltrpo, namixed_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    # axis_2[1, 3].plot(namixed_total_testenv_steps_rl2ppo, namixed_metatest_avg_successrate_rl2ppo, 'x',
    #                   color='green', label='RL2-PPO')

    # Pick place milk meta test
    # axis_2[0, 4].plot(ppmilk_total_testenv_steps_mamltrpo, ppmilk_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    # axis_2[0, 4].plot(ppmilk_total_testenv_steps_rl2ppo, ppmilk_metatest_avg_return_rl2ppo, color='green',
    #                   label='RL2-PPO')
    #
    # axis_2[1, 4].plot(ppmilk_total_testenv_steps_mamltrpo, ppmilk_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    # axis_2[1, 4].plot(ppmilk_total_testenv_steps_rl2ppo, ppmilk_metatest_avg_successrate_rl2ppo, 'x',
    #                   color='green', label='RL2-PPO')

    # Pick place bread meta test
    # axis_4[0, 0].plot(ppbread_total_testenv_steps_mamltrpo, ppbread_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    axis_4[0, 0].plot(ppbread_total_testenv_steps_rl2ppo, ppbread_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')
    axis_4[0, 0].set_ylim([0, 520])
    legend = axis_4[0, 0].legend()
    legend.get_frame().set_facecolor('white')

    # axis_4[1, 0].plot(ppbread_total_testenv_steps_mamltrpo, ppbread_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    axis_4[1, 0].plot(ppbread_total_testenv_steps_rl2ppo, ppbread_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    axis_4[1, 0].set_ylim([-5, 105])
    legend = axis_4[1, 0].legend()
    legend.get_frame().set_facecolor('white')

    # Pick place cereal meta test
    # axis_4[0, 1].plot(ppcereal_total_testenv_steps_mamltrpo, ppcereal_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    # axis_4[0, 1].plot(ppcereal_total_testenv_steps_rl2ppo, ppcereal_metatest_avg_return_rl2ppo, color='green',
    #                   label='RL2-PPO')
    #
    # axis_4[1, 1].plot(ppcereal_total_testenv_steps_mamltrpo, ppcereal_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    # axis_4[1, 1].plot(ppcereal_total_testenv_steps_rl2ppo, ppcereal_metatest_avg_successrate_rl2ppo, 'x',
    #                   color='green', label='RL2-PPO')

    # Stack blocks meta test
    # axis_4[0, 2].plot(stackblocks_total_testenv_steps_mamltrpo, stackblocks_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    axis_4[0, 2].plot(stackblocks_total_testenv_steps_rl2ppo, stackblocks_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    # axis_4[1, 2].plot(stackblocks_total_testenv_steps_mamltrpo, stackblocks_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    axis_4[1, 2].plot(stackblocks_total_testenv_steps_rl2ppo, stackblocks_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')
    axis_4[1, 2].xaxis.get_offset_text().set_visible(False)

    # Pick place can meta test
    # axis_4[0, 3].plot(ppcan_total_testenv_steps_mamltrpo, ppcan_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    # axis_4[0, 3].plot(ppcan_total_testenv_steps_rl2ppo, ppcan_metatest_avg_return_rl2ppo, color='green',
    #                   label='RL2-PPO')
    #
    # axis_4[1, 3].plot(ppcan_total_testenv_steps_mamltrpo, ppcan_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    # axis_4[1, 3].plot(ppcan_total_testenv_steps_rl2ppo, ppcan_metatest_avg_successrate_rl2ppo, 'x',
    #                   color='green', label='RL2-PPO')

    # Nut assembly square meta test
    # axis_4[0, 4].plot(nasquare_total_testenv_steps_mamltrpo, nasquare_metatest_avg_return_mamltrpo, color='red',
    #                   label='MAML-TRPO')
    axis_4[0, 4].plot(nasquare_total_testenv_steps_rl2ppo, nasquare_metatest_avg_return_rl2ppo, color='green',
                      label='RL2-PPO')

    # axis_4[1, 4].plot(nasquare_total_testenv_steps_mamltrpo, nasquare_metatest_avg_successrate_mamltrpo, '+',
    #                   color='red', label='MAML-TRPO')
    axis_4[1, 4].plot(nasquare_total_testenv_steps_rl2ppo, nasquare_metatest_avg_successrate_rl2ppo, 'x',
                      color='green', label='RL2-PPO')


    for i in range(5):
        axis[0, i].title.set_text(all_envs[i])
        # plt.setp(axis[0, i], xlabel=all_envs[i])

    for i in range(5):
        axis_2[0, i].title.set_text(all_envs[i])
        # plt.setp(axis_2[0, i], xlabel=all_envs[i])

    for i in range(5):
        axis_3[0, i].title.set_text(all_envs[i+5])
        # plt.setp(axis_3[0, i], xlabel=all_envs[i])

    for i in range(5):
        axis_4[0, i].title.set_text(all_envs[i+5])
        # plt.setp(axis_4[0, i], xlabel=all_envs[i])



    plt.setp(axis[0, 0], ylabel='Average Return')
    plt.setp(axis[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis_2[0, 0], ylabel='Average Return')
    plt.setp(axis_2[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis_3[0, 0], ylabel='Average Return')
    plt.setp(axis_3[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis_4[0, 0], ylabel='Average Return')
    plt.setp(axis_4[1, 0], ylabel='Success Rate (%)')
    plt.setp(axis[1, 2], xlabel='Total Environment Steps')
    plt.setp(axis_2[1, 2], xlabel='Total Environment Steps')
    plt.setp(axis_3[1, 2], xlabel='Total Environment Steps')
    plt.setp(axis_4[1, 2], xlabel='Total Environment Steps')
    # fig.suptitle('Meta 1 - All Tasks with IIWA14 (Meta Training)', fontsize=14)
    # fig_2.suptitle('Meta 1 - All Tasks with IIWA14 (Meta Test)', fontsize=14)
    fig.tight_layout()
    fig_2.tight_layout()
    fig_3.tight_layout()
    fig_4.tight_layout()
    fig.savefig('Meta1_IIWA14_AllTasks_Train.pgf')
    fig_3.savefig('Meta1_IIWA14_AllTasks_Train2.pgf')
    fig_2.savefig('Meta1_IIWA14_AllTasks_Test.pgf')
    fig_4.savefig('Meta1_IIWA14_AllTasks_Test2.pgf')

    
    # Kuka IIWA14 with no linear axis in Meta 7 with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    data_rows = []
    
    with open('Experiment_Data/Robosuite_IIWA14_Meta7/ml_maml_trpo/progress.csv', 'r') as file:
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
    metatest_avg_return_mamltrpo = metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(float)
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(metatest_avg_successrate_mamltrpo
                                                                                    != '')].astype(float) * 100.0)
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_mamltrpo = (metatest_avg_stdreturn_mamltrpo[np.where(metatest_avg_stdreturn_mamltrpo != '')]
                                       .astype(float))
    
    # Max success and return bar chart Meta 7 train/test tasks
    # Meta 7 test tasks MAML:
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
    
    # Meta 7 train tasks:
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
    
    with open('Experiment_Data/Robosuite_IIWA14_Meta7/ml_rl2_ppo/progress.csv', 'r') as file:
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
    
    
    # Plot everything
    fig4, axis4 = plt.subplots(2, 1)
    fig4_2, axis4_2 = plt.subplots(2, 1)

    axis4[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis4[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis4[0].set_ylim([0, 500])
    legend = axis4[0].legend()
    legend.get_frame().set_facecolor('white')

    axis4[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis4[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis4[1].set_ylim([0, 100])
    legend = axis4[1].legend()
    legend.get_frame().set_facecolor('white')

    axis4_2[0].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis4_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis4_2[0].set_ylim([0, 500])
    legend = axis4_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis4_2[1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis4_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis4_2[1].set_ylim([0, 100])
    legend = axis4_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis4[0], ylabel='Average Return')
    plt.setp(axis4[1], ylabel='Success Rate (%)')
    plt.setp(axis4_2[0], ylabel='Average Return')
    plt.setp(axis4_2[1], ylabel='Success Rate (%)')
    plt.setp(axis4[1], xlabel='Total Environment Steps')
    plt.setp(axis4_2[1], xlabel='Total Environment Steps')
    fig4.suptitle('Meta 7 with IIWA14 (Meta Training)', fontsize=14)
    fig4_2.suptitle('Meta 7 with IIWA14 (Meta Test)', fontsize=14)
    fig4.savefig('Meta7_IIWA14_Train.pgf')
    fig4_2.savefig('Meta7_IIWA14_Test.pgf')
    
    # Max success and return bar chart Meta 7 train/test tasks
    # Meta 7 test tasks RL2:
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
    metatest_pickplacecan_return_RL2 = (metatest_pickplacecan_return_RL2[np.where(metatest_pickplacecan_return_RL2 != '')]
                                        .astype(float))
    metatest_stack_return_RL2 = data_rows[:, header.index('MetaTest/stack-blocks/MaxReturn')]
    metatest_stack_return_RL2 = metatest_stack_return_RL2[np.where(metatest_stack_return_RL2 != '')].astype(float)
    
    # Meta 7 train tasks:
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
    
    max_return_train_envs_rl2 = [np.amax(door_return_RL2), np.amax(lift_return_RL2), np.amax(nutassemblyround_return_RL2),
                                 np.amax(nutassembly_return_RL2), np.amax(pickplacemilk_return_RL2),
                                 np.amax(pickplacecereal_return_RL2), np.amax(pickplacebread_return_RL2)]
    
    max_return_test_envs_rl2 = [np.amax(metatest_pickplacecan_return_RL2), np.amax(metatest_nutassemblysquare_return_RL2),
                                np.amax(metatest_stack_return_RL2)]
    
    max_return_train_envs_maml = [np.amax(door_return_MAML), np.amax(lift_return_MAML),
                                  np.amax(nutassemblyround_return_MAML), np.amax(nutassembly_return_MAML),
                                  np.amax(pickplacemilk_return_MAML), np.amax(pickplacecereal_return_MAML),
                                  np.amax(pickplacebread_return_MAML)]
    
    max_return_test_envs_maml = [np.amax(metatest_pickplacecan_return_MAML),
                                 np.amax(metatest_nutassemblysquare_return_MAML),
                                 np.amax(metatest_stack_return_MAML)]
    
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
    fig5.suptitle('Meta 7 - Max Returns/Success Rates per Meta Train Task (IIWA14)', fontsize=13)
    fig5_2.suptitle('Meta 7 - Max Returns/Success Rates per Meta Test Task (IIWA14)', fontsize=13)
    fig5.tight_layout()
    fig5_2.tight_layout()
    fig5.savefig('Meta7_IIWA14_SuccessReturns_MetaTrain.pgf')
    fig5_2.savefig('Meta7_IIWA14_SuccessReturns_MetaTest.pgf')


    # Rethink Robotics Sawyer in Meta 7 with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------

    data_rows = []
    
    with open('Experiment_Data/Robosuite_Sawyer_Meta7/ml_maml_trpo/progress.csv', 'r') as file:
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
    metatest_avg_return_mamltrpo = metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')].astype(float)
    metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
    metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(metatest_avg_successrate_mamltrpo
                                                                                    != '')].astype(float) * 100.0)
    metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
    metatest_avg_stdreturn_mamltrpo = (metatest_avg_stdreturn_mamltrpo[np.where(metatest_avg_stdreturn_mamltrpo != '')]
                                       .astype(float))
    
    # Max success and return bar chart Meta 7 train/test tasks
    # Meta 7 test tasks MAML:
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
    
    # Meta 7 train tasks:
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
    
    with open('Experiment_Data/Robosuite_Sawyer_Meta7/ml_rl2_ppo/progress.csv', 'r') as file:
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
    
    
    # Plot everything
    fig6, axis6 = plt.subplots(2, 1)
    fig6_2, axis6_2 = plt.subplots(2, 1)

    axis6[0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
    axis6[0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
    axis6[0].set_ylim([0, 500])
    legend = axis6[0].legend()
    legend.get_frame().set_facecolor('white')

    axis6[1].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
    axis6[1].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
    axis6[1].set_ylim([0, 100])
    legend = axis6[1].legend()
    legend.get_frame().set_facecolor('white')

    axis6_2[0].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
    axis6_2[0].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
    axis6_2[0].set_ylim([0, 500])
    legend = axis6_2[0].legend()
    legend.get_frame().set_facecolor('white')

    axis6_2[1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red',
                    label='MAML-TRPO')
    axis6_2[1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green',
                    label='RL2-PPO')
    axis6_2[1].set_ylim([0, 100])
    legend = axis6_2[1].legend()
    legend.get_frame().set_facecolor('white')

    plt.setp(axis6[0], ylabel='Average Return')
    plt.setp(axis6[1], ylabel='Success Rate (%)')
    plt.setp(axis6_2[0], ylabel='Average Return')
    plt.setp(axis6_2[1], ylabel='Success Rate (%)')
    plt.setp(axis6[1], xlabel='Total Environment Steps')
    plt.setp(axis6_2[1], xlabel='Total Environment Steps')
    fig6.suptitle('Meta 7 with Sawyer (Meta Training)', fontsize=14)
    fig6_2.suptitle('Meta 7 with Sawyer (Meta Test)', fontsize=14)
    fig6.savefig('Meta7_Sawyer_Train.pgf')
    fig6_2.savefig('Meta7_Sawyer_Test.pgf')
    
    # Max success and return bar chart Meta 7 train/test tasks
    # Meta 7 test tasks RL2:
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
    metatest_pickplacecan_return_RL2 = (metatest_pickplacecan_return_RL2[np.where(metatest_pickplacecan_return_RL2 != '')]
                                        .astype(float))
    metatest_stack_return_RL2 = data_rows[:, header.index('MetaTest/stack-blocks/MaxReturn')]
    metatest_stack_return_RL2 = metatest_stack_return_RL2[np.where(metatest_stack_return_RL2 != '')].astype(float)
    
    # Meta 7 train tasks:
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
    
    train_envs = ["Door", "Lift", "NutAssemblyRound", "NutAssembly", "PickPlaceMilk",
                  "PickPlaceCereal", "PickPlaceBread"]
    test_envs = ["PickPlaceCan", "NutAssemblySquare", "Stack"]
    
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
    
    max_return_train_envs_rl2 = [np.amax(door_return_RL2), np.amax(lift_return_RL2), np.amax(nutassemblyround_return_RL2),
                                 np.amax(nutassembly_return_RL2), np.amax(pickplacemilk_return_RL2),
                                 np.amax(pickplacecereal_return_RL2), np.amax(pickplacebread_return_RL2)]
    
    max_return_test_envs_rl2 = [np.amax(metatest_pickplacecan_return_RL2), np.amax(metatest_nutassemblysquare_return_RL2),
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
    fig7_2, axis7_2 = plt.subplots(2, 1)
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
    axis7_2[0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                    label='RL2-PPO', align='edge')
    axis7_2[0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
                    alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7_2[0].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis7_2[0].legend()
    legend.get_frame().set_facecolor('white')
    axis7_2[0].set_xlim([0, 100])
    axis7_2[0].set_title('Max Success Rate (%)', fontsize=12)
    axis7_2[1].set_title('Max Return', fontsize=12)
    axis7_2[1].barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                    label='RL2-PPO', align='edge')
    axis7_2[1].barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
                    alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7_2[1].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis7_2[1].legend()
    legend.get_frame().set_facecolor('white')
    axis7_2[1].set_xlim([0, 500])
    plt.setp(axis7[0], xlabel='Max Success Rate (%)')
    plt.setp(axis7[1], xlabel='Max Return')
    plt.setp(axis7_2[0], xlabel='Max Success Rate (%)')
    plt.setp(axis7_2[1], xlabel='Max Return')
    fig7.suptitle('Meta 7 - Max Returns/Success Rates per Meta Train Task (Sawyer)', fontsize=13)
    fig7_2.suptitle('Meta 7 - Max Returns/Success Rates per Meta Test Task (Sawyer)', fontsize=13)
    fig7.tight_layout()
    fig7_2.tight_layout()
    fig7.savefig('Meta7_Sawyer_SuccessReturns_MetaTrain.pgf')
    fig7_2.savefig('Meta7_Sawyer_SuccessReturns_MetaTest.pgf')

    # plt.show()


if __name__ == "__main__":
    with plt.style.context('ggplot'):
        plot_all()
