# CRiSE 1-3-7 — Compilations of Real-World inspired Robotic Task Simulation Environments

The Garage-related main part of the CRiSE 1-3-7 source code can be found in the directory **CRiSE137**.

**environments/manipulation** contains all new designed tasks.

The added Robotics Lab mimicking KUKA LBR IIWA14 with locked linear axes can be found in **models/robots/manipulators** (iiwa14_extended_nolinear). For the initialisation necessary data (meshes, .xml) is located in **models/assets/robots**.

Major adaptations were done to the Gym wrapper in **wrappers/wrapper.py**. 

