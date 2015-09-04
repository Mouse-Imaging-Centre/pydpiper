#!/usr/bin/env/python

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

setup(name='pydpiper',
      version='2.0a1',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description = 'Python code for flexible pipeline control', 
      author='Miriam Friedel, Matthijs van Eede, Jason Lerch, Jon Pipitone, Fraser MacDonald, Ben Darwin',
      maintainer_email='matthijs@mouseimaging.ca',
      url='https://github.com/Mouse-Imaging-Centre/pydpiper',
      install_requires=[
        'ConfigArgParse',
        'networkx',
        'pygraphviz',
        'Pyro4',
        'pytest'
      ],
      platforms="any",
      packages=['pydpiper', 'pydpiper.core', 'pydpiper.minc', 'pydpiper.execution', 'pydpiper.atoms_and_modules', 'pydpiper.pipelines'],
      data_files=[('config', ['config/MICe.cfg','config/MICe_dev.cfg','config/SciNet.cfg','config/SciNet_debug.cfg'])],
      scripts=['pydpiper/execution/pipeline_executor.py', 'pydpiper/execution/check_pipeline_status.py',
               'pydpiper/pipelines/LSQ12.py', 'pydpiper/pipelines/LSQ6.py'],
      #tests_require=['pytest']
      )

