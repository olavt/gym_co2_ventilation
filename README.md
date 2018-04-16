# How to create and install a custom Open AI Gym Environment

### Author: [Olav Tollefsen](https://www.linkedin.com/in/olavtollefsen/)

## Introduction

This repository contains a rather simple custom OpenAI Gym environment 'gym_co2_ventilation', which can be used by several frameworks and tools to experiment with Reinforcement Learning algorithms. The problem solved in this sample environment is to train the software to control a ventilation system. The goals are to keep an acceptable level of CO2 in the indoor air, while minimizing the energy used for ventilation / heating / cooling.

## System Requirements

- Python 3.6 or higher
- PIP

## Installation of the custom Gym environment

Clone the repository to a directory, change default directory to the top-level directory of the cloned repository and issue the following PIP command:

```
$ pip install -e .
```

You may need to restart Python in order for the new Gym environmnet to be available for use.

## Using the custom Gym environment

To use the new custom Gym environmnet, you need to import it into your code like this:

```python
import gym
# This will trigger the code to register the custom environment with Gym
import gym_co2_ventilation 

env = gym.make('CO2VentilationSimulator-v0')
env.reset()
for _ in range(360):
    env.render()
    action = env.action_space.sample()  # take a random action
    env.step(action) 
```

You should see output like this:
```
CO2VentilationSimulatorEnv - Version 0.0.1
Fan speed=1, CO2=776
Fan speed=1, CO2=791.0
Fan speed=4, CO2=806.0
Fan speed=2, CO2=776.0
Fan speed=1, CO2=786.0
Fan speed=4, CO2=801.0
Fan speed=4, CO2=771.0
Fan speed=4, CO2=741.0
```

## How does the custom environment work?

The main logic of the custom environment can be found in this file: [gym_co2_ventilation/gym_co2_ventilation/envs/co2_ventilation_simulator_env.py](https://github.com/olavt/gym_co2_ventilation/blob/master/gym_co2_ventilation/envs/co2_ventilation_simulator_env.py)

## Reinforcement Learining using the custom gym environment

An example on how to use the custom gym environment for Reinforcement Learning can be found here: [gym_co2_ventilation/examples/test_keras_rl.py](https://github.com/olavt/gym_co2_ventilation/blob/master/examples/test_keras_rl.py)