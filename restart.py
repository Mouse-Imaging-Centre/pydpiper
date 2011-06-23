#!/usr/bin/env python

from pydpiper.pipeline import *
from optparse import OptionParser
from os.path import isdir,abspath
from os import mkdir
import networkx as nx

Pyro.config.PYRO_MOBILE_CODE=1 

if __name__ == '__main__':	
    usage = "%prog [options]"
    description = "restart a crashed or terminated pipeline from backups"

    parser = OptionParser(usage=usage, description=description)
    
    parser.add_option("--output-dir", dest="output_directory",
                      type="string", default=".",
                      help="Directory for existing output and backups subdirectory.")
                      
    (options,args) = parser.parse_args()
    outputDir = abspath(options.output_directory)
    if not isdir(outputDir):
        sys.exit('Specified output/backups directory does not exist. Cannot reimport data from backups. Exiting...')
    p = Pipeline()
    p.setBackupFileLocation(outputDir)
    print 'Pipeline created, trying to recover reusable stages ... \n'
    try:
	    p.recycle()
    except IOError:
        sys.exit("Backup files are not recoverable.  Pipeline restart required.\n")
    print 'Previously completed stages (of ' + str(len(p.stages)) + ' total): \n'
    done = []        
    for i in p.G.nodes_iter():
        if p.stages[i].isFinished() == True:
	    done.append(i)
    print '  ' + str(done)    
    print 'Now computing starting nodes ... \n'
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
    print 'Reimport complete. Now passing off to NoNSDaemon ...\n'
    
    pipelineNoNSDaemon(p)


# Lots To Do:
# Make this code non-specific to type of pipeline (see issue #24)
# If Pyro NameServer option was specified, use it
# Auto-check for pickled files
# Make sure re-queueing works
# Bundle all files together
# Make backup on a timer
    
