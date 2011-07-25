#!/usr/bin/env python

import Pyro.core
import Pyro.naming
import time
import sys
import os
from optparse import OptionParser
from multiprocessing import Process 

Pyro.config.PYRO_MOBILE_CODE=1

#use Pyro.core.CallbackObjBase?? - need further review of documentation
class Executor(Pyro.core.SynchronizedObjBase):
    def __init__(self):
        Pyro.core.SynchronizedObjBase.__init__(self)
        self.continueRunning =  True
    def continueLoop(self):
        return self.continueRunning
    def serverShutdownCall(self, serverShutdown):
        # receive call from server when all stages are processed
        if serverShutdown:
            self.continueRunning = False
         

def executePipeline(serverURI, executor):
    # Need to get own proxy here, since separate thread
    p = Pyro.core.getProxyForURI(serverURI)
    #while True:
    while executor.continueLoop:
        try:
            i = p.getRunnableStageIndex()
            if i == None:
            	print("No runnable stages. Going to sleep")
            	time.sleep(5)
            else:
            	print("\n")
            	s = p.getStage(i)
            	print("Running stage " + str(i) + ":")
            	print(s)
            	r = s.execStage()
            	print("return was: " + str(r))
            	if r == 0:
                    p.setStageFinished(i)
            	else:
                    p.setStageFailed(i)
                ps = p.getProcessedStageCount()
                print("Number of processed stages: " + str(ps))
        except:
            print "Failed in executor thread"
    	    print "Unexpected error: ", sys.exc_info()
    	    sys.exit()

if __name__ == "__main__":
    usage = "%prog [options]"
    description = "pipeline executor"
    
    parser = OptionParser(usage=usage, description=description)
    
    parser.add_option("--uri-file", dest="urifile",
                      type="string", default=None,
                      help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    parser.add_option("--use-ns", dest="use_ns",
                      action="store_true",
                      help="Use the Pyro NameServer to store object locations")
                      
    (options,args) = parser.parse_args()
       
    Pyro.core.initClient()
    Pyro.core.initServer()
    daemon = Pyro.core.Daemon()
    
    #initialize executor as both client and server
    if options.use_ns:
        ns = Pyro.naming.NameServerLocator().getNS()
        serverURI = ns.resolve("pipeline")
        daemon.useNameServer(ns)
    else:
        if options.urifile==None:
            urifile = os.curdir + "/" + "uri"
        else:
            urifile = options.urifile
    	uf = open(urifile)
    	serverURI = Pyro.core.processStringURI(uf.readline())
    	uf.close()
    
    #TEST
    
    executor = Executor()
    clientURI=daemon.connect(executor,"executor")
    p = Pyro.core.getProxyForURI(serverURI)
    p.register(clientURI)
    
    process = Process(target=executePipeline, args=(serverURI, executor))
    process.start()
    try:
        daemon.requestLoop(executor.continueLoop)
    except:
        print "Failed in pipeline_executor"
    	print "Unexpected error: ", sys.exc_info()
    	process.join()
    	daemon.shutdown(True)
    	sys.exit()
    else:
        print "Server has shutdown. Killing executor thread."
        process.terminate()
        daemon.shutdown(True)
    
    
    

    
        
