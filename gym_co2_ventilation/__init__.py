from gym.envs.registration import register

register(
    id='CO2VentilationProduction-v0',
    entry_point='gym_co2_ventilation.envs:CO2VentilationProductionEnv',
    timestep_limit=60,
)

register(
    id='CO2VentilationSimulator-v0',
    entry_point='gym_co2_ventilation.envs:CO2VentilationSimulatorEnv',
    timestep_limit=60,
)