import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class CO2VentilationSimulatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.0.1"
        print("CO2VentilationSimulatorEnv - Version {}".format(self.__version__))

        # Define the action_space
        # 0=VentilationFanSpeed1
        # 1=VentilationFanSpeed2
        # 2=VentilationFanSpeed3
        # 3=VentilationFanSpeed4
        self.action_space = spaces.Discrete(4)

        # Define the observation_space
        # First dimension is VentilationFanSpeed
        # Second dimension is CO2 level in the air
        # Third dimension is CO2 change from previous state
        low = np.array([0, 400, -100])
        high = np.array([3, 3000, 100])
        self.observation_space = spaces.Box(low, high)

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.curr_step += 1
        ventilation_speed, co2_level, co2_diff = self.state
        new_ventilation_speed = action

        # Compute reward
        if co2_level < 950:
            reward = 1.0
        elif co2_level < 1000:
            reward = 0.8
        elif co2_level < 1200:
            reward = 0.5
        elif co2_level < 1500:
            reward = -0.2
        else:
            reward = -0.6

        # Give penalty for energy consumption
        # Need to add outdoor / indoor temperature based calculation
        ventilation_cost = self._get_ventilation_cost(ventilation_speed)
        reward = reward - ventilation_cost

        # Give a small penalty for changing ventilation fan speed
        if new_ventilation_speed != ventilation_speed:
            reward = reward - 0.1

        self.total_reward += reward

        #print("Reward={}, Total reward={}".format(reward , self.total_reward))
        
        # Compute next state
        ventilation_volume = self._get_ventilation_volume(ventilation_speed)
        new_co2_diff = 20 - (ventilation_volume * 50)
        new_co2_level = co2_level + new_co2_diff
        if new_co2_level < 400:
            new_co2_level = 400
        elif new_co2_level > 3000:
            new_co2_level = 3000

        # Update to next state
        self.state = (new_ventilation_speed, new_co2_level, new_co2_diff)

        done = False

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.curr_step = 0
        self.total_reward = 0.0
        ventilation_speed = 0
        co2_level = np.random.randint(400, 900)
        co2_diff = 0
        self.state = (ventilation_speed, co2_level, co2_diff)
        return np.array(self.state)

    def render(self, mode='human'):
        ventilation_speed, co2_level, co2_diff = self.state
        print("Fan speed={}, CO2={}".format(ventilation_speed + 1, co2_level))

    def _get_ventilation_volume(self, ventilation_speed):
        ventilation_speed_volume = [0.1, 0.2, 0.5, 1.0]
        ventilation_volume = ventilation_speed_volume[ventilation_speed]
        return ventilation_volume

    def _get_ventilation_cost(self, ventilation_speed):
        ventilation_speed_cost = [0.0, 0.2, 0.5, 1.0]
        ventilation_cost = ventilation_speed_cost[ventilation_speed]
        return ventilation_cost