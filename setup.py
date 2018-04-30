from setuptools import setup

setup(name='gym_co2_ventilation',
      version='0.0.1',
      install_requires=[
            'gym',
            'keras',
            'requests',
            'tensorflow',
            'azure.servicebus'
      ]
)