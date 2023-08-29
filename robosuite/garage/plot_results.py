"""File for plotting all the experimental results"""

import csv
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
fig, axis = plt.subplots(2, 2)
axis[0, 0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
axis[0, 0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
axis[0, 0].set_ylim([0, 500])
axis[0, 0].set_title('Average Return Training')
axis[0, 0].legend()

axis[0, 1].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
axis[0, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis[0, 1].set_ylim([0, 500])
axis[0, 1].set_title('Return Meta Test')
axis[0, 1].legend()

axis[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis[1, 0].set_ylim([0, 100])
axis[1, 0].set_title('Average Successrate Training')
axis[1, 0].legend()

axis[1, 1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', marker='o', linestyle='',
                label='MAML-TRPO')
axis[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', marker='o', linestyle='',
                label='RL2-PPO')
axis[1, 1].set_ylim([0, 100])
axis[1, 1].set_title('Successrate Meta Test')
axis[1, 1].legend()

plt.setp(axis[0, 0], ylabel='Average Return')
plt.setp(axis[1, 0], ylabel='Success Rate (%)')
plt.setp(axis[1, 0], xlabel='Total Environment Steps')
plt.setp(axis[1, 1], xlabel='Total Environment Steps')
fig.suptitle('Lift Block Single RML with Sawyer', fontsize=20)
fig.savefig('Meta1_Sawyer_LiftBlock.pgf')


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
metatest_avg_successrate_mamltrpo = (metatest_avg_successrate_mamltrpo[np.where(
    metatest_avg_successrate_mamltrpo != '')].astype(float)) * 100.0
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
axis3[0, 0].set_ylim([0, 500])
axis3[0, 0].set_title('Average Return Training')
axis3[0, 0].legend()

axis3[0, 1].plot(total_env_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
axis3[0, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis3[0, 1].set_ylim([0, 500])
axis3[0, 1].set_title('Return Meta Test')
axis3[0, 1].legend()

axis3[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis3[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis3[1, 0].set_ylim([0, 100])
axis3[1, 0].set_title('Average Successrate Training')
axis3[1, 0].legend()

axis3[1, 1].plot(total_env_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', marker='o', linestyle='',
                 label='MAML-TRPO')
axis3[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', marker='o', linestyle='',
                 label='RL2-PPO')
axis3[1, 1].set_ylim([0, 100])
axis3[1, 1].set_title('Successrate Meta Test')
axis3[1, 1].legend()

plt.setp(axis3[0, 0], ylabel='Average Return')
plt.setp(axis3[1, 0], ylabel='Success Rate (%)')
plt.setp(axis3[1, 0], xlabel='Total Environment Steps')
plt.setp(axis3[1, 1], xlabel='Total Environment Steps')
fig3.suptitle('Lift Block Single RML with IIWA14', fontsize=20)

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
fig4, axis4 = plt.subplots(2, 2)
axis4[0, 0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
axis4[0, 0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
axis4[0, 0].set_ylim([0, 500])
axis4[0, 0].set_title('Average Return Training')
axis4[0, 0].legend()

axis4[0, 1].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
axis4[0, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis4[0, 1].set_ylim([0, 500])
axis4[0, 1].set_title('Average Return Meta Test')
axis4[0, 1].legend()

axis4[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis4[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis4[1, 0].set_ylim([0, 100])
axis4[1, 0].set_title('Average Success Rate Training')
axis4[1, 0].legend()

axis4[1, 1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', marker='o', linestyle='',
                 label='MAML-TRPO')
axis4[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', marker='o', linestyle='',
                 label='RL2-PPO')
axis4[1, 1].set_ylim([0, 100])
axis4[1, 1].set_title('Average Success Rate Meta Test')
axis4[1, 1].legend()

plt.setp(axis4[0, 0], ylabel='Average Return')
plt.setp(axis4[1, 0], ylabel='Average Success Rate (%)')
plt.setp(axis4[1, 0], xlabel='Total Environment Steps')
plt.setp(axis4[1, 1], xlabel='Total Environment Steps')
fig4.suptitle('Meta 7 - RML across tasks with IIWA14', fontsize=20)

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

width_single_bar = 0.2
fig5, axis5 = plt.subplots(2, 2)
bar_y_pos1 = np.arange(7)
bar_y_pos2 = np.arange(3)
axis5[0, 0].barh(bar_y_pos1, max_success_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis5[0, 0].barh(bar_y_pos1 + width_single_bar, max_success_train_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis5[0, 0].set_yticks(bar_y_pos1, train_envs)
axis5[0, 0].legend()
axis5[0, 0].set_xlim([0, 100])
axis5[0, 0].set_title('Meta 7 Train Tasks Max Success Rates')
axis5[0, 1].barh(bar_y_pos1, max_return_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis5[0, 1].barh(bar_y_pos1 + width_single_bar, max_return_train_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis5[0, 1].set_yticks(bar_y_pos1, train_envs)
axis5[0, 1].legend()
axis5[0, 1].set_xlim([0, 500])
axis5[0, 1].set_title('Meta 7 Train Tasks Max Returns')
axis5[1, 0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis5[1, 0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis5[1, 0].set_yticks(bar_y_pos2, test_envs)
axis5[1, 0].legend()
axis5[1, 0].set_xlim([0, 100])
axis5[1, 0].set_title('Meta 7 Test Tasks Max Success Rates')
axis5[1, 1].barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis5[1, 1].barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis5[1, 1].set_yticks(bar_y_pos2, test_envs)
axis5[1, 1].legend()
axis5[1, 1].set_xlim([0, 500])
axis5[1, 1].set_title('Meta 7 Test Tasks Max Returns')
plt.setp(axis5[0, 0], ylabel='Train Tasks')
plt.setp(axis5[1, 0], ylabel='Test Tasks')
plt.setp(axis5[1, 0], xlabel='Max Success Rate (%)')
plt.setp(axis5[1, 1], xlabel='Max Return')
fig5.suptitle('Meta 7 - Max Returns and Success Rates per Task with IIWA14', fontsize=20)


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
fig6, axis6 = plt.subplots(2, 2)
axis6[0, 0].plot(total_env_steps_mamltrpo, average_return_mamltrpo, color='red', label='MAML-TRPO')
axis6[0, 0].plot(total_env_steps_rl2ppo, average_return_rl2ppo, color='green', label='RL2-PPO')
axis6[0, 0].set_ylim([0, 500])
axis6[0, 0].set_title('Average Return Training')
axis6[0, 0].legend()

axis6[0, 1].plot(total_testenv_steps_mamltrpo, metatest_avg_return_mamltrpo, color='red', label='MAML-TRPO')
axis6[0, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_return_rl2ppo, color='green', label='RL2-PPO')
axis6[0, 1].set_ylim([0, 500])
axis6[0, 1].set_title('Average Return Meta Test')
axis6[0, 1].legend()

axis6[1, 0].plot(total_env_steps_mamltrpo, average_successrate_mamltrpo, color='red', label='MAML-TRPO')
axis6[1, 0].plot(total_env_steps_rl2ppo, average_successrate_rl2ppo, color='green', label='RL2-PPO')
axis6[1, 0].set_ylim([0, 100])
axis6[1, 0].set_title('Average Success Rate Training')
axis6[1, 0].legend()

axis6[1, 1].plot(total_testenv_steps_mamltrpo, metatest_avg_successrate_mamltrpo, color='red', marker='o', linestyle='',
                 label='MAML-TRPO')
axis6[1, 1].plot(total_testenv_steps_rl2ppo, metatest_avg_successrate_rl2ppo, color='green', marker='o', linestyle='',
                 label='RL2-PPO')
axis6[1, 1].set_ylim([0, 100])
axis6[1, 1].set_title('Average Success Rate Meta Test')
axis6[1, 1].legend()

plt.setp(axis6[0, 0], ylabel='Average Return')
plt.setp(axis6[1, 0], ylabel='Average Success Rate (%)')
plt.setp(axis6[1, 0], xlabel='Total Environment Steps')
plt.setp(axis6[1, 1], xlabel='Total Environment Steps')
fig6.suptitle('Meta 7 - RML across tasks with Sawyer', fontsize=20)

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

width_single_bar = 0.2
fig7, axis7 = plt.subplots(2, 2)
bar_y_pos1 = np.arange(7)
bar_y_pos2 = np.arange(3)
axis7[0, 0].barh(bar_y_pos1, max_success_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis7[0, 0].barh(bar_y_pos1 + width_single_bar, max_success_train_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis7[0, 0].set_yticks(bar_y_pos1, train_envs)
axis7[0, 0].legend()
axis7[0, 0].set_xlim([0, 100])
axis7[0, 0].set_title('Meta 7 Train Tasks Max Success Rates')
axis7[0, 1].barh(bar_y_pos1, max_return_train_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis7[0, 1].barh(bar_y_pos1 + width_single_bar, max_return_train_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis7[0, 1].set_yticks(bar_y_pos1, train_envs)
axis7[0, 1].legend()
axis7[0, 1].set_xlim([0, 500])
axis7[0, 1].set_title('Meta 7 Train Tasks Max Returns')
axis7[1, 0].barh(bar_y_pos2, max_success_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis7[1, 0].barh(bar_y_pos2 + width_single_bar, max_success_test_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis7[1, 0].set_yticks(bar_y_pos2, test_envs)
axis7[1, 0].legend()
axis7[1, 0].set_xlim([0, 100])
axis7[1, 0].set_title('Meta 7 Test Tasks Max Success Rates')
axis7[1, 1].barh(bar_y_pos2, max_return_test_envs_rl2, height=-width_single_bar, alpha=0.9, color='orange',
                 label='RL2-PPO', align='edge')
axis7[1, 1].barh(bar_y_pos2 + width_single_bar, max_return_test_envs_maml, height=-width_single_bar,
                 alpha=0.8, color='yellow', label='MAML-TRPO', align='edge')
axis7[1, 1].set_yticks(bar_y_pos2, test_envs)
axis7[1, 1].legend()
axis7[1, 1].set_xlim([0, 500])
axis7[1, 1].set_title('Meta 7 Test Tasks Max Returns')
plt.setp(axis7[0, 0], ylabel='Train Tasks')
plt.setp(axis7[1, 0], ylabel='Test Tasks')
plt.setp(axis7[1, 0], xlabel='Max Success Rate (%)')
plt.setp(axis7[1, 1], xlabel='Max Return')
fig7.suptitle('Meta 7 - Max Returns and Success Rates per Task with Sawyer', fontsize=20)
"""
plt.show()
