#!/usr/bin/env python

import Pyro.core
import Pyro.naming
import time
import sys
import os
from optparse import OptionParser
from multiprocessing import Process
from multiprocessing import Pool

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

def canRun(s, maxMem, maxProcs, runningMem, runningProcs):
    if ( (int(s.getMem()) <= (maxMem-runningMem) ) and (int(s.getNumProcesses())<=(maxProcs-runningProcs)) ):
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
        return[s.getMem(),s.getNumProcesses()] # If completed, return mem & processes back for re-use    
    except:
        print "Failed in executor thread"
        print "Unexpected error: ", sys.exc_info()
        task_done()
    	sys.exit()        
         
class pipelineExecutor():
    #def __init__(self):
    # initialization here later, if needed. 
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
      
        maxMem = 8      # default
        maxProcs = 4    # default
        runningMem = 0
        runningProcs = 0        
        
        runningChildren = [] # no scissors
              
        # loop until the pipeline sets continueRunning to false
        pool = Pool(processes = options.proc)
        try:
            while executor.continueLoop():
                daemon.handleRequests(0)  #functions in place of daemon.requestLoop() to allow for custom event loop
                
                # Refresh usable mem & procs
                completedTasks = []
                for i,val in enumerate(runningChildren):
                    if val.ready():                         
                        completedTasks.insert(0,i)
                for i in completedTasks:
                    runningMem -= runningChildren[i].get()[0]
                    runningProcs -= runningChildren[i].get()[1]
                    del runningChildren[i]                    
                
                # Check for new stages
                try:
                    i = p.getRunnableStageIndex()
                    
                    if i == None:
                        print("No runnable stages. Sleeping...")
                        time.sleep(5)
                    else:
                        s = p.getStage(i)
                        if canRun(s, maxMem, maxProcs, runningMem, runningProcs):
                            runningMem += s.getMem()
                            runningProcs += s.getNumProcesses()                         
                            runningChildren.append(pool.apply_async(runStage,(serverURI, i)))
                        else:
                            p.requeue(i)
                except:
                    print "Executor failed"
                    print "Unexpected error: ", sys.exc_info()
                    sys.exit()              

        except:
            print "Failed in pipeline_executor"
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
                      help="Number of processes per executor. If not specified, default is 4.")
                      
    (options,args) = parser.parse_args()

    pe = pipelineExecutor()
    processes = [Process(target=pe.launchPipeline, args=(options,)) for i in range(options.num_exec)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
