#!/usr/bin/env python3
import os
from setuptools import setup

setup(name='pydpiper',
      version='2.0b1',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description='Python code for flexible pipeline control',
      author='Miriam Friedel, Matthijs van Eede, Jason Lerch, Jon Pipitone, Fraser MacDonald, Ben Darwin',
      maintainer_email='matthijs@mouseimaging.ca',
      url='https://github.com/Mouse-Imaging-Centre/pydpiper',
      install_requires=[
        'ConfigArgParse>=0.11.0',
        'networkx',
        'ordered-set',
        'pandas',
        #'pydotplus',  # use nx.nx_pydot.write_dot to write graphviz files
        #'pygraphviz',
        # in principle one could require 'pydotplus' instead (and use nx.nx_pydot.write_dot), but that package isn't
        # very maintained (pydot_ng seems to be more popular, but networkx hasn't switched).  This is
        # annoying because pygraphviz needs the graphviz headers (e.g., from libgraphviz-dev .deb) to compile.
        'pyminc',
        'Pyro4',
        'pytest',
        'typing'
      ],
      #extras_require = { 'graphing' : ['pygraphviz']},  # could make pygraphviz optional, but then won't auto-install
      platforms="any",
      packages=['pydpiper', 'pydpiper.core', 'pydpiper.minc', 'pydpiper.execution', 'pydpiper.pipelines'],
      data_files=[('config',
                   [os.path.join("config", f)
                    for f in ['CCM_HPF.cfg', 'MICe.cfg', 'MICe_dev.cfg', 'SciNet.cfg', 'SciNet_debug.cfg']])],
      scripts=([os.path.join("pydpiper/execution", script) for script in
                ['pipeline_executor.py', 'check_pipeline_status.py']] +
               [os.path.join("pydpiper/pipelines", f) for f in
                ['asymmetry.py', 'LSQ12.py', 'LSQ6.py', 'MAGeT.py', 'MBM.py', 'NLIN.py',
                 'registration_chain.py', 'twolevel_model_building.py']]),
                 #'stats.py',
      tests_require=['pytest']
      )
