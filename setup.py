#!/usr/bin/env python3

import os
import sys

from setuptools import setup

if sys.version_info < (3, 5):
    raise ValueError("Minimum Python version supported is 3.5")

setup(name='pydpiper',
      version='2.0.11',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description='Python code for flexible pipeline control',
      author='Miriam Friedel, Matthijs van Eede, Jason Lerch, Jon Pipitone, Fraser MacDonald, Ben Darwin',
      maintainer_email='matthijs.vaneede@sickkids.ca',
      url='https://github.com/Mouse-Imaging-Centre/pydpiper',
      install_requires=[
        'ConfigArgParse>=0.11.0',
        'networkx>=2.0b1',
        'ordered-set',
        'pandas',
        #'pydot',  # use nx.nx_pydot.write_dot to write graphviz files
        #'pygraphviz',
        # pygraphviz needs the graphviz headers (e.g., from libgraphviz-dev .deb) to compile.
        'pyminc',
        'Pyro4',
        'pytest',
        'typing',
        'qbatch'
      ],
      #extras_require = { 'graphing' : ['pygraphviz']},  # could make pygraphviz optional, but then won't auto-install
      platforms="any",
      packages=['pydpiper', 'pydpiper.core', 'pydpiper.execution', 'pydpiper.itk', 'pydpiper.minc', 'pydpiper.pipelines'],
      data_files=[('config',
                   [os.path.join("config", f)
                    for f in ['CCM_HPF.cfg', 'MICe.cfg', 'MICe_dev.cfg', 'SciNet.cfg', 'SciNet_debug.cfg']])],
      scripts=([os.path.join("pydpiper/execution", script) for script in
                ['pipeline_executor.py', 'check_pipeline_status.py']] +
               [os.path.join("pydpiper/pipelines", f) for f in
                ['asymmetry.py', 'LSQ12.py', 'LSQ6.py', 'MAGeT.py', 'MBM.py', 'NLIN.py',
                 'registration_chain.py', 'stage_embryos_in_4D_atlas.py', 'twolevel_model_building.py']]),
                 #'stats.py',
      tests_require=['pytest'],
      zip_safe=False  # since we want the data files to be installed on disk for the moment ...
      )
