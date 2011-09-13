#!/usr/bin/env python

import Pyro.core
import Pyro.naming
import time
import sys
import os
from optparse import OptionParser
from datetime import date
from multiprocessing import Process, Pool
from subprocess import call

Pyro.config.PYRO_MOBILE_CODE=1

#use Pyro.core.CallbackObjBase?? - need further review of documentation
class clientExecutor(Pyro.core.SynchronizedObjBase):
    def __init__(self):
        Pyro.core.SynchronizedObjBase.__init__(self)
        self.continueRunning =  True
    def continueLoop(self):
        return self.continueRunning
    def serverShutdownCall(self, serverShutdown):
        # receive call from server when all stages are processed
        if serverShutdown:
            self.continueRunning = False

def canRun(stageMem, stageProcs, maxMem, maxProcs, runningMem, runningProcs):
    if ( (int(stageMem) <= (maxMem-runningMem) ) and (int(stageProcs)<=(maxProcs-runningProcs)) ):
        return True
    else:
        return False
         
def runStage(serverURI, i):
    # Proc needs its own proxy as it's independent of executor
    p = Pyro.core.getProxyForURI(serverURI)
    s = p.getStage(i)
    
    # Run stage, set finished or failed accordingly  
    try:
        print("Running stage " + str(i) + ": " + str(s) + "\n")
        r = s.execStage()
        print("Stage " + str(i) + " finished, return was: " + str(r) + "\n")
        if r == 0:
            p.setStageFinished(i)
        else:
            p.setStageFailed(i)
        return[s.getMem(),s.getProcs()] # If completed, return mem & processes back for re-use    
    except:
        print "Failed in executor thread"
        print "Unexpected error: ", sys.exc_info()
        task_done()
    	sys.exit()        
         
class pipelineExecutor():
    #def __init__(self):
    # initialization here later, if needed. 
    def submitToQueue(self, options=None):
    # Need to put in memory mgmt here as well. 
        if options.queue=="sge":
            strprocs = str(options.proc) 
            # NOTE: May want to change definition of options.mem, as sge_batch multiplies this by # procs. 
            # Currently designed as total amount of memory needed. May want this to be mem/process or just divide. 
            # Update to allow for multiple executors
            strmem = "vf=" + str(options.mem) + "G"  
            jobname = "pipeline-" + str(date.today())
            # Add options for sge_batch command
            cmd = ["sge_batch", "-J", jobname, "-m", strprocs, "-l", strmem] 
            # Next line is for development and testing only -- will be removed in final checked in version
            cmd += ["-q", "defdev.q"]
            cmd += ["pipeline_executor.py", "--uri-file", options.urifile, "--proc", strprocs]
            print cmd
            call(cmd)   
        elif options.queue=="scinet":
            print "Specified queueing system == scinet"
    def launchPipeline(self, options=None):  
        # initialize pipeline_executor as both client and server       
        Pyro.core.initClient()
        Pyro.core.initServer()
        daemon = Pyro.core.Daemon()

        # set up communication with server from the URI string
        if options.use_ns:
            ns = Pyro.naming.NameServerLocator().getNS()
            serverURI = ns.resolve("pipeline")
            daemon.useNameServer(ns)
        else:
            if options.urifile==None:
                urifile = os.curdir + "/" + "uri"
            else:
                urifile = options.urifile
    	    try:
    	        uf = open(urifile)
    	        serverURI = Pyro.core.processStringURI(uf.readline())
    	        uf.close()
    	    except:
    	        print "Problem opening the specified uri file:", sys.exc_info()
    	        sys.exit()

        # instantiate the executor class and register the executor with the pipeline    
        executor = clientExecutor()
        clientURI=daemon.connect(executor,"executor")
        p = Pyro.core.getProxyForURI(serverURI)
        p.register(clientURI)
      
        maxMem = options.mem
        maxProcs = options.proc
        runningMem = 0
        runningProcs = 0        
        
        runningChildren = [] # no scissors
              
        # loop until the pipeline sets continueRunning to false
        pool = Pool(processes = options.proc)
        try:
            while executor.continueLoop():
                daemon.handleRequests(0)               
                # Check for new stages
                i = p.getRunnableStageIndex()                 
                if i == None:
                    print("No runnable stages. Sleeping...")
                    time.sleep(5)
                else:
                    # Before running stage, check usable mem & procs
                    completedTasks = []
                    for j,val in enumerate(runningChildren):
                        if val.ready():                         
                            completedTasks.insert(0,j)
                    for j in completedTasks:
                        runningMem -= runningChildren[j].get()[0]
                        runningProcs -= runningChildren[j].get()[1]
                        del runningChildren[j]    
                    s = p.getStage(i)
                    stageMem = s.getMem()
                    stageProcs = s.getProcs()
                    if canRun(stageMem, stageProcs, maxMem, maxProcs, runningMem, runningProcs):
                        runningMem += stageMem
                        runningProcs += stageProcs                   
                        runningChildren.append(pool.apply_async(runStage,(serverURI, i)))
                    else:
                        p.requeue(i)
        except:
            print "Failed in pipeline executor."
            print "Unexpected error: ", sys.exc_info()
            daemon.shutdown(True)
            pool.close()
            pool.join()
            sys.exit()
        else:
            print "Server has shutdown. Killing executor thread."
            pool.close()
            pool.join()        
            daemon.shutdown(True)


##########     ---     Start of program     ---     ##########   

if __name__ == "__main__":
    usage = "%prog [options]"
    description = "pipeline executor"

    # command line option handling    
    parser = OptionParser(usage=usage, description=description)
    
    parser.add_option("--uri-file", dest="urifile",
                      type="string", default=None,
                      help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    parser.add_option("--use-ns", dest="use_ns",
                      action="store_true",
                      help="Use the Pyro NameServer to store object locations")
    parser.add_option("--num-executors", dest="num_exec", 
                      type="int", default=1, 
                      help="Number of independent executors to launch.")
    parser.add_option("--proc", dest="proc", 
                      type="int", default=4,
                      help="Number of processes per executor. If not specified, default is 4. Also sets max value for processor use per executor.")
    parser.add_option("--mem", dest="mem", 
                      type="int", default=8,
                      help="Total amount of requested memory for all processes the executor runs. If not specified, default is 8 GB.")              
                      
    (options,args) = parser.parse_args()

    pe = pipelineExecutor()
    processes = [Process(target=pe.launchPipeline, args=(options,)) for i in range(options.num_exec)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
