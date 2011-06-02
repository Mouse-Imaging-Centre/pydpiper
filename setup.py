#!/usr/bin/env/python

from distutils.core import setup

setup(name='pydpiper',
      version='0.1',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description = 'Python code for flexible pipeline control', 
      author='Jason Lerch, Miriam Friedel, Fraser MacDonald, Matthijs Van Eede',
      maintainer_email='mfriedel@phenogenomics.ca',
      url='https://github.com/mfriedel/pydpiper',
      platforms="any",
      packages=['pydpiper'], 
      scripts=['pipeline_executor.py']
      )
