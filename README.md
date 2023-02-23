# Object detection using a Single Shot Decetor
In this project, my goal is to train a neural network to detect both robots and balls in the context of [RoboCup SPL](https://spl.robocup.org/). 
I have implemented a Single Shot Detector (SSD) following to the work of Liu et al.[^1]. The architecture of the initial network is based on the works of [BHuman](https://github.com/bhuman/BHumanCodeRelease). 
However, implementation of BHuman is limited to detect only Robots. This project is my attempt to extent the network such that it detect both robots and balls.

[^1]:[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
