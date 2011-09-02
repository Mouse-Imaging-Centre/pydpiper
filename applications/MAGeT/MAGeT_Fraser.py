#!/usr/bin/env python

from pydpiper.pipeline import *
from minctracc import *
from optparse import OptionParser
from os.path import dirname,isdir,abspath
from os import mkdir
import networkx as nx

Pyro.config.PYRO_MOBILE_CODE=1 

if __name__ == "__main__":
    usage = "%prog [options] input1.mnc ... inputn.mnc"
    description = "description needed"


    p = Pipeline()
    p.restart()
    
    #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
    # if use_ns is specified, Pyro NameServer must be started. 
    
    pipelineDaemon(p, options)
