from setuptools import setup, find_packages

setup(name='wavebase',
      version='0.1',
      description='Featurizures and bases for signal processing.', 
      license='Apache License 2.0',     
      url='https://jonrbates.ai',
      package_dir={"": "src"},
      packages=find_packages("src"),
)