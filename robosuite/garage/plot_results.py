"""File for plotting all the experimental results"""

import csv
import numpy as np
import matplotlib.pyplot as plt

data_rows = []

# Sawyer in Robosuite's blocklifting task (ML1) with MAML_TRPO and RL2_PPO
# ------------------------------------------------------------------------

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
fig, axis = plt.subplots(2, 2)
axis[0, 0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
axis[0, 0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
axis[0, 0].set_title('Average Return Training')
axis[0, 0].legend()

axis[0, 1].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
axis[0, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis[0, 1].set_title('Return Meta Test')
axis[0, 1].legend()

axis[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis[1, 0].set_title('Average Successrate Training')
axis[1, 0].legend()

axis[1, 1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', marker='o', linestyle='',
                label='MAML-TRPO')
axis[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', marker='o', linestyle='',
                label='RL2-PPO')
axis[1, 1].set_title('Successrate Meta Test')
axis[1, 1].legend()

plt.setp(axis[0, 0], ylabel='Average Return')
plt.setp(axis[1, 0], ylabel='Success Rate (%)')
plt.setp(axis[1, 0], xlabel='Total Environment Steps')
plt.setp(axis[1, 1], xlabel='Total Environment Steps')
fig.suptitle('Lift Block Single RML with Sawyer', fontsize=20)


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
metatest_avg_return_mamltrpo = metatest_avg_return_mamltrpo[np.where(metatest_avg_return_mamltrpo != '')]
metatest_avg_successrate_mamltrpo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(metatest_avg_successrate_mamltrpo != '')]
                                     .astype(float)) * 100.0
metatest_avg_stdreturn_mamltrpo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
metatest_avg_stdreturn_mamltrpo = metatest_avg_stdreturn_mamltrpo[np.where(metatest_avg_stdreturn_mamltrpo != '')]


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
# metatest_avg_return_rl2ppo = data_rows[:, header.index('MetaTest/Average/AverageReturn')]
total_env_steps_rl2ppo = data_rows[:, header.index('TotalEnvSteps')].astype(int)
# total_testenv_steps_rl2ppo = total_env_steps_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(int)

# Get meta test average return, success rate and standard return RL2_PPO
# metatest_avg_return_rl2ppo = metatest_avg_return_rl2ppo[np.where(metatest_avg_return_rl2ppo != '')].astype(float)
# metatest_avg_successrate_rl2ppo = data_rows[:, header.index('MetaTest/Average/SuccessRate')]
# metatest_avg_successrate_rl2ppo = (metatest_avg_successrate_rl2ppo[np.where(metatest_avg_successrate_rl2ppo != '')]
#                                    .astype(float) * 100.0)
# metatest_avg_stdreturn_rl2ppo = data_rows[:, header.index('MetaTest/Average/StdReturn')]
# metatest_avg_stdreturn_rl2ppo = (metatest_avg_stdreturn_rl2ppo[np.where(metatest_avg_stdreturn_rl2ppo != '')]
#                                  .astype(float))

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
fig2, axis2 = plt.subplots(2, 2)
axis2[0, 0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
axis2[0, 0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
axis2[0, 0].set_title('Average Return Training')
axis2[0, 0].legend()

axis2[0, 1].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
# axis2[0,1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis2[0, 1].plot(total_env_steps_pearl, metatest_avg_return_pearl, color='blue', label='PEARL')
axis2[0, 1].set_title('Return Meta Test')
axis2[0, 1].legend()

axis2[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis2[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis2[1, 0].set_title('Average Successrate Training')
axis2[1, 0].legend()

axis2[1, 1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', label='MAML-TRPO')
# axis2[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', label='RL2-PPO')
axis2[1, 1].plot(total_env_steps_pearl, metatest_avg_successrate_pearl, color='blue', label='PEARL')
axis2[1, 1].set_title('Successrate Meta Test')
axis2[1, 1].legend()

plt.setp(axis2[0, 0], ylabel='Average Return')
plt.setp(axis2[1, 0], ylabel='Success Rate (%)')
plt.setp(axis2[1, 0], xlabel='Total Environment Steps')
plt.setp(axis2[1, 1], xlabel='Total Environment Steps')
fig2.suptitle('Meta-World ML10 with Sawyer', fontsize=20)


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
fig3, axis3 = plt.subplots(2, 2)
axis3[0, 0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
axis3[0, 0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
axis3[0, 0].set_title('Average Return Training')
axis3[0, 0].legend()

axis3[0, 1].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
axis3[0, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis3[0, 1].set_title('Return Meta Test')
axis3[0, 1].legend()

axis3[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis3[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis3[1, 0].set_title('Average Successrate Training')
axis3[1, 0].legend()

axis3[1, 1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', marker='o', linestyle='',
                 label='MAML-TRPO')
axis3[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', marker='o', linestyle='',
                 label='RL2-PPO')
axis3[1, 1].set_title('Successrate Meta Test')
axis3[1, 1].legend()

plt.setp(axis3[0, 0], ylabel='Average Return')
plt.setp(axis3[1, 0], ylabel='Success Rate (%)')
plt.setp(axis3[1, 0], xlabel='Total Environment Steps')
plt.setp(axis3[1, 1], xlabel='Total Environment Steps')
fig3.suptitle('Lift Block Single RML with IIWA14', fontsize=20)
plt.show()
