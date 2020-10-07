#!/usr/bin/env python3

import os
import sys

from setuptools import setup

min_python_version = (3, 7)

if sys.version_info < min_python_version:
    raise ValueError("Minimum Python version supported is %d.%d" %
                       (min_python_version[0], min_python_version[1]))

setup(name='pydpiper',
      version='2.0.15',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description='Python code for flexible pipeline control',
      author='Miriam Friedel, Matthijs van Eede, Jason Lerch, Jon Pipitone, Fraser MacDonald, Ben Darwin, Nick Wang',
      maintainer_email='benjamin.darwin@sickkids.ca',
      url='https://github.com/Mouse-Imaging-Centre/pydpiper',
      python_requires=">=3.6",
      install_requires=[
        'ConfigArgParse>=0.11.0',
        'networkx>=2.3',
        'ordered-set',
        'pandas',
        #'pydot',  # use nx.nx_pydot.write_dot to write graphviz files
        #'pygraphviz',
        # pygraphviz needs the graphviz headers (e.g., from libgraphviz-dev .deb) to compile.
        'pyminc',
        'Pyro4',
       # 'pytest',
       #TODO fix get_model_building_procedure in pydpiper/minc/registration_strategies.py. ref issue #387
        'qbatch',
        'simplejson'
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
                 'registration_chain.py',
                 'registration_tamarack.py',
                 'stage_embryos_in_4D_atlas.py',
                 'twolevel_model_building.py']]),
                 #'stats.py',
      tests_require=['pytest'],
      zip_safe=False  # since we want the data files to be installed on disk for the moment ...
      )
