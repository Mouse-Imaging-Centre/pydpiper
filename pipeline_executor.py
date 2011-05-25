#!/usr/bin/env python

#import matplotlib
#matplotlib.use("PS") #to avoid X errors - see
# http://matplotlib.sourceforge.net/faq/installing_faq.html#backends

import Pyro.core
import time
from pipeline import *
from MAGeT import *
from minctracc import *
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
            	print("Going to sleep")
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
        except Pyro.errors.ConnectionClosedError:
            sys.exit("Connection with server closed. Server shutdown and system exit.")
        except:
            sys.exit("An error has occurred. Pipeline may not have completed properly. Check logs and restart if needed.")
        
