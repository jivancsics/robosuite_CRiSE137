# CRiSE 1-3-7 — Compilations of Real-World inspired Robotic Task Simulation Environments

This directory contains all the CRiSE 1-3-7 software components that constitute all tasks and integrate the compilations into the experiment structure of **Garage**. Furthermore, utility scripts are provided to help analyse the experiments. E.g., important functionalities like the unique sampling and decoding of _parametric variations_ and the set-up of experiments are done in ml_robosuite.py. Please take a look at the source codes for a description.

## experiment_launchers

This directory contains all experiment launchers of CRiSE 1-3-7 and utility scripts for visualisation of learned MRL policies, for video recording and for measuring the mean runtime per single environment step in all CRiSE 7 meta-train and meta-test tasks. The csv_merger_meta3 script helps to merge experiment data if the meta-training is cancelled and again resumed in a new experiment afterwards.

The sub-directory **IIWA14_extended_nolinear** contains the experiment launchers for CRiSE 1-3-7 experiments with the Robotic Lab setup mimicking _KUKA LBR IIWA14_ robot. E.g., an RL^2-PPO-based CRiSE 1 experiment is started by entering:

```shell
python3 robosuite_rl2_ppo_crise1.py
```

For more information on how to set the experiment parameters over passed arguments, analyse the respective experiment launcher scripts.

In addition to this, the sub-directory **Sawyer** contains the experiment launchers for CRiSE 1-3-7 experiments with the Robosuite default _Rethink Robotics Sawyer_ robot.   

## Robosuite_IIWA14_extended_nolinear and Robosuite_Sawyer

These directories contain all CRiSE tasks that were designed in a Garage MRL-friendly structure. Amongst others, the _parametric variations_ are set here.
