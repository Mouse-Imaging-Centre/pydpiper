#!/usr/bin/env python

#import matplotlib
#matplotlib.use("PS") #to avoid X errors - see
# http://matplotlib.sourceforge.net/faq/installing_faq.html#backends

import Pyro.core
import time
from pipeline import *
from MAGeT import *
import sys

Pyro.config.PYRO_MOBILE_CODE=1


if __name__ == "__main__":
    
    #locator = Pyro.naming.NameServerLocator()
    #ns = locator.getNS()
    #p = Pyro.core.getProxyForURI("PYRONAME://ptest")
    uf = open(sys.argv[1])
    uri = Pyro.core.processStringURI(uf.readline())
    uf.close()
    p = Pyro.core.getProxyForURI(uri)
    #p = Pyro.core.getProxyForURI("PYROLOC://172.20.103.84:7766/ptest")

    while True:
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
