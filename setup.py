#!/usr/bin/env/python

from distutils.core import setup

setup(name='pydpiper',
      version='1.10',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description = 'Python code for flexible pipeline control', 
      author='Miriam Friedel, Matthijs van Eede, Jason Lerch, Jon Pipitone, Fraser MacDonald, Ben Darwin',
      maintainer_email='mfriedel@mouseimaging.ca',
      url='https://github.com/mfriedel/pydpiper',
      platforms="any",
      packages=['pydpiper', 'applications', 'atoms_and_modules'], 
      scripts=['pydpiper/pipeline_executor.py', 'pydpiper/check_pipeline_status.py', 'applications/MAGeT.py', 'applications/MBM.py', 'applications/registration_chain.py',
               'applications/twolevel_model_building.py', 'applications/pairwise_nlin.py', 'atoms_and_modules/NLIN.py', 'atoms_and_modules/LSQ12.py', 'atoms_and_modules/LSQ6.py'])
