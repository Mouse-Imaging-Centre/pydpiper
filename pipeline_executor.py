#!/usr/bin/env python

import Pyro.core
import Pyro.naming
import time
import sys

Pyro.config.PYRO_MOBILE_CODE=1


if __name__ == "__main__":
    
    Pyro.core.initClient()
    if len(sys.argv) == 1:
	# no command line arguments - assume using NS
        ns = Pyro.naming.NameServerLocator().getNS()
        uri = ns.resolve("pipeline")
    elif len(sys.argv) == 2:
   	# one argument --> assume not using NS, reading in uri file
    	uf = open(sys.argv[1])
    	uri = Pyro.core.processStringURI(uf.readline())
    	uf.close()
    else:
	sys.exit("pipeline_executor should have only one command line argument.")
    
    p = Pyro.core.getProxyForURI(uri)

    while True:
    	try:
            i = p.getRunnableStageIndex()
            if i == None:
            	print("No runnable stages. Going to sleep")
            	time.sleep(5)
            else:
            	print("in: ")
            	print(i)
            	s = p.getStage(i)
            	print("Running:")
            	print(s)
            	r = s.execStage()
            	print("return was: " + str(r))
            	if r == 0:
                    p.setStageFinished(i)
            	else:
                    p.setStageFailed(i)
                ps = p.getProcessedStageCount()
                print("Number of processed stages: " + str(ps))
        except Pyro.errors.ConnectionClosedError:
            sys.exit("Connection with server closed. Server shutdown and system exit.")
        except:
            print "Failed in pipeline_executor"
    	    print "Unexpected error: ", sys.exc_info()
    	    sys.exit()
        
