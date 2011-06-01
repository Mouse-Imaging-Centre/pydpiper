#!/usr/bin/env python


from pipeline import *
from minctracc import *
from optparse import OptionParser
from os.path import basename,dirname,isdir,abspath
from os import mkdir
import time
import networkx as nx

Pyro.config.PYRO_MOBILE_CODE=1 

if __name__ == '__main__':	

    p = Pipeline()
    print '\nI) Pipeline created, now recovering reusable stages ... '
    try:
	p.recycle()
    except IOError:
	sys.exit("  IOError: backup files are not recoverable.  Pipeline restart required.\n")
    print '\nII) Previously completed stages (of ' + str(len(p.stages)) + ' total):'
    done = []        
    for i in p.G.nodes_iter():
        if p.stages[i].isFinished() == True:
	    done.append(i)
    print '  ' + str(done)    
    print '\nIII) Now computing starting nodes ... '
    p.computeGraphHeads()
    starters = []
    for i in p.G.nodes_iter():
        if p.stages[i].isFinished() == False:
            if len(p.G.predecessors(i)) == 0:
                starters.append(i)
            if len(p.G.predecessors(i)) != 0:
                predfinished = True
                for j in p.G.predecessors(i):
                    if p.stages[j].isFinished() == False:
                        predfinished = False
                if predfinished == True:
                    starters.append(i) 
    print '  ' + str(starters)
    print '\nIV) Now passing off to NoNSDaemon ...\n'
    
    pipelineNoNSDaemon(p)


# Lots To Do:
# If Pyro NameServer option was specified, use it
# Auto-check for pickled files
# Make sure re-queueing works
# Bundle all files together
# Make backup on a timer
    
