#!/usr/bin/env python3

from setuptools import setup


test_deps = ['pytest', 'pytest-console-scripts'],  # also rawtominc (minc_tools), param2xfm (mni_autoreg)

extras = {
  'test' : test_deps
}


setup(name='pydpiper',
      version='2.0.19',
      license='Modified BSD',
      description='Python code for flexible pipeline control',
      long_description='Python code for flexible pipeline control',
      author='Miriam Friedel, Matthijs van Eede, Jason Lerch, Jon Pipitone, Fraser MacDonald, Ben Darwin, Nick Wang',
      maintainer_email='benjamin.darwin@sickkids.ca',
      url='https://github.com/Mouse-Imaging-Centre/pydpiper',
      python_requires=">=3.7",
      install_requires=[
        'ConfigArgParse>=0.11.0',
        'Jinja2',
        'networkx',
        'ordered-set',
        'pandas',
        #'pydot',  # use nx.nx_pydot.write_dot to write graphviz files
        #'pygraphviz',
        # pygraphviz needs the graphviz headers (e.g., from libgraphviz-dev .deb) to compile.
        'pyminc',
        'Pyro5',
       #TODO fix get_model_building_procedure in pydpiper/minc/registration_strategies.py. ref issue #387
        'qbatch',
        'simplejson'
      ],
      extras_require=extras,
      platforms="any",
      packages=['pydpiper', 'pydpiper.core', 'pydpiper.execution', 'pydpiper.itk', 'pydpiper.minc', 'pydpiper.pipelines'],
      package_data={ 'pydpiper' : ['templates/*.sh'] },
      data_files=[('config',
                   [f"config/{f}" for f in ['CCM_HPF.cfg', 'MICe.cfg', 'MICe_dev.cfg', 'SciNet.cfg']])],
      entry_points={
          'console_scripts':
            [f'{pipe}{ext}=pydpiper.pipelines.{pipe}:application' for pipe in
                ['asymmetry', 'LSQ12', 'LSQ6', 'MAGeT', 'MBM', 'NLIN',
                 'registration_chain',
                 'registration_tamarack',
                 'stage_embryos_in_4D_atlas',
                 'twolevel_model_building'] for ext in ["", ".py"]] +
            ['pipeline_executor.py=pydpiper.execution.pipeline_executor:main',
             'check_pipeline_status.py=pydpiper.execution.check_pipeline_status:main'],
      },
      tests_require=test_deps,
      zip_safe=False  # since we want the data files to be installed on disk for the moment ...
      )
