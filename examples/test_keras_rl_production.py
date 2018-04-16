import gym
import gym_co2_ventilation  # This will register the custom environment

import logging
import numpy as np
import requests
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

logger = logging.getLogger("Logger")
ch = logging.StreamHandler()
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Initialize logger for logging each step in the continious learning process
step_logger = logging.getLogger("StepLogger")
step_logger.setLevel(logging.INFO)
fh = logging.FileHandler(f'co2_ventilation_step_log_{time.strftime("%Y_%m_%d_%H%M")}.log', mode='w')
step_logger.addHandler(fh)
step_logger.info("Time,Iteration,Step,FanSpeed,Reward,CO2Level")
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)

# Initialize logger for logging summary for each episode in the continious learning process
episode_logger = logging.getLogger("EpisodeLogger")
episode_logger.setLevel(logging.INFO)
fh = logging.FileHandler(f'co2_ventilation_episode_log_{time.strftime("%Y_%m_%d_%H%M")}.log', mode='w')
episode_logger.addHandler(fh)
episode_logger.info("Time,Episode, Reward")
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)

ENV_NAME = 'CO2VentilationProduction-v0'

# Create the environment
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Build a neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

nb_episode_steps = 60
nb_episodes = 1
nb_episodes_memory = 1000

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=nb_episode_steps*nb_episodes, window_length=1)
#policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

try:
    dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
except (OSError):
    logger.warning ("File not found")

n = 0
while True:
    n += 1
    logger.info (f'Iteration #{n}')

    # Run some training
    history = dqn.fit(env, nb_max_episode_steps=nb_episode_steps, nb_steps=nb_episode_steps*nb_episodes, visualize=True, verbose=2)

    # Save neural network weights
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Write training results to log file
    rewards = history.history['episode_reward']
    for i in range(0, nb_episodes):
        episode_logger.info(f'{(n - 1)*nb_episodes + i + 1},{rewards[i]}')