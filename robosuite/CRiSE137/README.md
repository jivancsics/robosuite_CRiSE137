# CRiSE 1-3-7 — Compilations of Real-World inspired Robotic Task Simulation Environments

This directory contains all the CRiSE 1-3-7 software components that constitute all tasks and integrate the compilations into the experiment structure of **Garage**. Furthermore, utility scripts are provided that help to analyse the experiments. E.g., important functionalities like the unique sampling and decoding of _parametric variations_ and the set up of experiments are done in ml_robosuite.py. See the source codes for a description.

## experiment_launchers

This directory contains all experiment launchers of CRiSE 1-3-7 and utility scripts for visualisation of learned MRL policies, for video recording and for measuring the mean runtime per single environment step in all CRiSE 7 meta-train and meta-test tasks. The csv_merger_meta3 script helps to merge experiment data if the meta-training is cancelled and again resumed in a new experiment afterwards.

The sub-directory **IIWA14_extended_nolinear** contains the experiment launchers for CRiSE 1-3-7 experiments with the Robotic Lab setup mimicking _KUKA LBR IIWA14_ robot. E.g., an RL^2-PPO-based CRiSE 1 experiment is started by entering:

```shell
python3 robosuite_rl2_ppo_crise1.py
```

For getting more information on how to set the experiment parameters over passed arguments, analyse the respective experiment launcher scripts.

In addition to this, the sub-directory **Sawyer** contains the experiment launchers for CRiSE 1-3-7 experiments with the Robosuite default _Rethink Robotics Sawyer_ robot.   

## Robosuite_IIWA14_extended_nolinear and Robosuite_Sawyer

These directories contain all CRiSE tasks that got designed in a Garage MRL-friendly structure. Amongst others, the _parametric variations_ are set here.

## plot_results.py
The directories in this script must be modified to the individual directory structure.
Proposed structure (no modifications are necessary with this structure):

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
