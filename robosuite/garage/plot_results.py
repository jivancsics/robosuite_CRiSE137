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


def plot_all():

    # Sawyer in Robosuite's blocklifting task (ML1) with MAML_TRPO and RL2_PPO
    # ------------------------------------------------------------------------

    data_rows = []

    with open('Experiment_Data/Robosuite_Sawyer_ML1_Blocklifting/singleml_maml_trpo/progress.csv', 'r') as file:
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

    with open('Experiment_Data/Robosuite_Sawyer_ML1_Blocklifting/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    axis[1].set_ylim([0, 100])
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
    axis_2[1].set_ylim([0, 100])
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
    

    # Kuka IIWA14 with no linear axis in Robosuite's blocklifting task (ML1) with MAML_TRPO and RL2_PPO
    # -------------------------------------------------------------------------------------------------
    
    data_rows = []
    
    with open('Experiment_Data/Robosuite_IIWA14_ML1_Blocklifting/singleml_maml_trpo/progress.csv', 'r') as file:
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
    
    with open('Experiment_Data/Robosuite_IIWA14_ML1_Blocklifting/singleml_rl2_ppo/progress.csv', 'r') as file:
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
    axis3[1].set_ylim([0, 100])
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
    axis3_2[1].set_ylim([0, 100])
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
    """
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
    """
    
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
    # axis7[0].barh(bar_y_pos1 + width_single_bar, max_success_train_envs_maml, height=-width_single_bar,
    #               alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7[0].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis7[0].legend()
    legend.get_frame().set_facecolor('white')
    axis7[0].set_xlim([0, 100])
    axis7[0].set_title('Max Success Rate (%)', fontsize=12)
    axis7[1].set_title('Max Return', fontsize=12)
    axis7[1].barh(bar_y_pos1, max_return_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                  label='RL2-PPO', align='edge')
    # axis7[1].barh(bar_y_pos1 + width_single_bar, max_return_train_envs_maml, height=-width_single_bar,
    #               alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7[1].set_yticks(bar_y_pos1, train_envs, fontsize=5)
    legend = axis7[1].legend()
    legend.get_frame().set_facecolor('white')
    axis7[1].set_xlim([0, 500])
    axis7_2[0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                    label='RL2-PPO', align='edge')
    # axis7_2[0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
    #                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7_2[0].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis7_2[0].legend()
    legend.get_frame().set_facecolor('white')
    axis7_2[0].set_xlim([0, 100])
    axis7_2[0].set_title('Max Success Rate (%)', fontsize=12)
    axis7_2[1].set_title('Max Return', fontsize=12)
    axis7_2[1].barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                    label='RL2-PPO', align='edge')
    # axis7_2[1].barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
    #                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
    axis7_2[1].set_yticks(bar_y_pos2, test_envs, fontsize=5)
    legend = axis7_2[1].legend()
    legend.get_frame().set_facecolor('white')
    axis7_2[1].set_xlim([0, 500])
    # plt.setp(axis7[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis7[1], xlabel='Max Return')
    # plt.setp(axis7_2[0], xlabel='Max Success Rate (%)')
    # plt.setp(axis7_2[1], xlabel='Max Return')
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
