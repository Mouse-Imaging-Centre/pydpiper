#!/usr/bin/env/python

from setuptools import setup
import sys

""" Testing required packages & Python version """
prepackages_displaylist = {'networkx': 'networkx (http://networkx.lanl.gov)', 'Pyro': 'Pyro (http://irmen.home.xs4all.nl/pyro3/)',
                          'pytest': 'pytest (http://doc.pytest.org/en/latest/) [OPTIONAL]'}
prepackages_stringlist = {'networkx': 'installed', 'Pyro': 'installed', 'pytest': 'installed'}

print 'Testing required packages ...'

printstr = "Python >= v. 2.6"
print ' ' + printstr.ljust(60, ' '),
if sys.version_info<=(2,5,0):
    print 'not installed (current v. = ' + sys.version[:5] + ')'
    sys.exit("ERROR: Cancelling install. Please update Python to at least version 2.6")
else:
    print 'installed'

for key in prepackages_stringlist.keys():
    try:             
        __import__(key)
    except ImportError:
        prepackages_stringlist[key] = 'not installed'
    print ' ' + prepackages_displaylist[key].ljust(61, ' ') + prepackages_stringlist[key]

print 'Done testing required packages.\n'

    
try:
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
          scripts=['pipeline_executor.py'],
          install_requires=['networkx', 'Pyro']
          )
except ImportError, e:
    print e
    
