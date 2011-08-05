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
    except:
        print "Failed in executor thread"
        print "Unexpected error: ", sys.exc_info()
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
      
        # loop until the pipeline sets continueRunning to false
        pool = Pool(processes=3)    # Limit the number of processes in the multiprocessing pool here
        try:
            while executor.continueLoop():
                daemon.handleRequests(0)  #functions in place of daemon.requestLoop() to allow for custom event loop
                try:
                    i = p.getRunnableStageIndex()
                    if i == None:
                        print("No runnable stages. Sleeping...")
                        time.sleep(5)
                        # This is still buggy - shorter sleep times cause the executor to miss
                        # the ShutdownCall and crash when calling p.getRunnableStageIndex()
                    else:
                        process = pool.apply_async(runStage,(serverURI, i))
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
                      
    (options,args) = parser.parse_args()

    pipelineExecutor = pipelineExecutor()
    pipelineExecutor.launchPipeline(options)
    
