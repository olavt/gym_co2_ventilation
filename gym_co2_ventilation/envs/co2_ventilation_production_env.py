import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import json
import numpy as np
import os
import requests
from azure.servicebus import ServiceBusService, Message, Topic, Rule

CO2_SENSOR_ID = "1401011"

class CO2VentilationProductionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.logger = logging.getLogger("Logger")
        self.step_logger = logging.getLogger("StepLogger")
        self.__version__ = "0.0.1"
        self.logger.info(f"CO2VentilationProductionEnv - Version {self.__version__}")

        # Get config from environment variables
        self.service_bus_namespace = os.environ["SERVICE_BUS_NAMESPACE"]
        self.service_bus_sas_key_name = os.environ["SERVICE_BUS_SAS_KEY_NAME"]
        self.service_bus_sas_key_value = os.environ["SERVICE_BUS_SAS_KEY_VALUE"]
        self.ventilation_rest_url = os.environ["VENTILATION_REST_URL"]
        self.ventilation_rest_api_key = os.environ["VENTILATION_REST_API_KEY"]

        # Define the action_space
        # 0=VentilationFanSpeed1
        # 1=VentilationFanSpeed2
        # 2=VentilationFanSpeed3
        # 3=VentilationFanSpeed4
        self.action_space = spaces.Discrete(4)

        # Define the observation_space
        # First dimension is VentilationFanSpeed (0..3)
        # Second dimension is CO2 level in the air (400...3000)
        # Third dimension is CO2 change from previous state (-100..100)
        low = np.array([0, 400, -100])
        high = np.array([3, 3000, 100])
        self.observation_space = spaces.Box(low, high)

        self.curr_iteration = 0
        self.current_co2_level = 400
        self.previous_co2_level = 400

        self._initialize_event_subscriber()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.curr_step += 1
        self.logger.info("")
        self.logger.info(f"Iteration #{self.curr_iteration} Step #{self.curr_step}")
        t0_ventilation_speed, t0_co2_level, t0_co2_diff = self.state

        # Execute action on environment (change ventilation fan speed)
        self._execute_action(action)

        # Wait for environment to transition to next state
        self._transition_to_next_state()

        # Get reward for new state
        reward = self._get_reward(self.current_co2_level, self.current_ventilation_speed, t0_ventilation_speed)
        self.total_reward += reward

        self.logger.info(f"Reward={reward}, Total reward={self.total_reward}")
        self.step_logger.info(f"{self.curr_iteration},{self.curr_step},{self.current_ventilation_speed + 1},{reward},{self.current_co2_level}")

        done = False

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.curr_iteration += 1
        self.curr_step = 0
        self.total_reward = 0.0
        co2_level = self.current_co2_level
        co2_diff = self.current_co2_level - self.previous_co2_level
        self.state = (ventilation_speed, co2_level, co2_diff)
        return np.array(self.state)

    def render(self, mode='human'):
        ventilation_speed, co2_level, co2_diff = self.state
        self.logger.info(f"Environment state: Fan speed={ventilation_speed + 1}, CO2={co2_level}, CO2Diff={co2_diff}")

    def _execute_action(self, action):
        self.logger.info(f"Executing action, setting fan speed to {action + 1}")
        self.current_ventilation_speed = action
        # Call REST service to change the fan speed of the ventilation system
        fanSpeedCommandId = f'FanSpeed{self.current_ventilation_speed + 1}'
        data = '{"deviceGroupId":"Ventilation", "deviceId":"302", "capabilityId":"VentilationFan", "commandId":"' + f'{fanSpeedCommandId}' + '", "parameters":""}'
        try:
            r = requests.post(f'{self.ventilation_rest_url}?code={self.ventilation_rest_api_key}', data = data)
            if r.status_code != 200:
                self.logger.error(f'REST call to change ventilation speed failed: {r.reason}')
        except:
            self.logger.exception("Exception from REST call to change ventilation fan speed")

    def _transition_to_next_state(self):
        self.logger.info ("Waiting for environment to respond to action...")
        # Note: The timeout should be 120 seconds, but that crashes due to a bug in the Python SDK for Service Bus
        # Wait for new CO2 sensor data to be received
        try:
            msg = self.bus_service.receive_subscription_message('sensordata', 'test', peek_lock=True, timeout=60)
            if msg.body is not None:
                self._process_sensor_data(msg.body)
                msg.delete()
        except requests.exceptions.ReadTimeout:
            self.logger.exception("ReadTimeout from ServiceBusService.receive_subscription_message")

        # Update environment state
        t0_ventilation_speed, t0_co2_level, t0_co2_diff = self.state
        co2_diff = self.current_co2_level - t0_co2_level
        self.state = (self.current_ventilation_speed, self.current_co2_level, co2_diff)

    def _get_reward(self, t1_co2_level, current_ventilation_speed, previous_ventilation_speed):
        if t1_co2_level < 900:
            reward = 1.0
        elif t1_co2_level < 950:
            reward = 0.9
        elif t1_co2_level < 1000:
            reward = 0.8
        elif t1_co2_level < 1200:
            reward = 0.4
        elif t1_co2_level < 1500:
            reward = -0.2
        else:
            reward = -0.6

        # Give penalty for energy consumption
        # Need to add outdoor / indoor temperature based calculation, but keep it simple for now
        ventilation_cost = self._get_ventilation_cost(current_ventilation_speed)
        reward = reward - ventilation_cost

        # Give a small penalty for changing ventilation fan speed (but, not for the 1st step in an Episode)
        if self.curr_step > 1:
            if current_ventilation_speed != previous_ventilation_speed:
                reward = reward - 0.1

        return reward

    def _update_co2_level(self, co2_level):
        self.previous_co2_level = self.current_co2_level
        self.current_co2_level = co2_level
        if self.previous_co2_level == 0:
            self.previous_co2_level = self.current_co2_level

    def _get_ventilation_cost(self, ventilation_speed):
        ventilation_speed_cost = [0.0, 0.2, 0.4, 0.8]
        ventilation_cost = ventilation_speed_cost[ventilation_speed]
        return ventilation_cost

    def _initialize_event_subscriber(self):
        self.bus_service = ServiceBusService(
            service_namespace = self.service_bus_namespace,
            shared_access_key_name = self.service_bus_sas_key_name,
            shared_access_key_value = self.service_bus_sas_key_value)

        self._remove_all_event_subscriptions()
        self._add_event_subscription(CO2_SENSOR_ID)
        self._remove_all_event_messages()

    def _remove_all_event_messages(self):
        # Clear queue of existing messages
        self.logger.info('Removing any pending sensor data messages from service bus subscription')
        while True:
            msg = self.bus_service.receive_subscription_message('sensordata', 'test', peek_lock=True, timeout=5)
            if msg.body is None:
                break
            self._process_sensor_data(msg.body)
            msg.delete()

    def _remove_all_event_subscriptions(self):
        rules = self.bus_service.list_rules('sensordata', 'test')
        for rule in rules:
            self.bus_service.delete_rule('sensordata', 'test', rule.name)

    def _add_event_subscription(self, event_id):
        rule = Rule()
        rule.filter_type = 'SqlFilter'
        rule.filter_expression = f"EventId='{event_id}'"
        self.bus_service.create_rule('sensordata', 'test', event_id, rule)

    def _process_sensor_data(self, message_body):
        self.logger.info(message_body)
        sensordata = json.loads(message_body)
        sensor_id = sensordata['Id']
        sensor_value = sensordata['Value']
        if (sensor_id == CO2_SENSOR_ID):
            self._update_co2_level(sensor_value)