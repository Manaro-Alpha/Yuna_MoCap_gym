# Yuna-MoCap-Gym

## About

This repo is an attempt to make a hexapod learn walking motion using Proximal Policy Optimisation and using Motion Capture data generated using a custom CPG controller. This project is inspired by ![legged_gym](https://github.com/leggedrobotics/legged_gym)

![]()

## Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && pip install -e .` 
5. Install legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`

## Usage

- To generate MoCap data, run `data_generator.py`
- To train yuna run `python3 train.py --task=yuna`
- To test the trained policy run `python3 test.py --task=yuna`
