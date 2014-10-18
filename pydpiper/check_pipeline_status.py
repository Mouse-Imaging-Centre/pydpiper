#!/usr/bin/env python

import os
from optparse import OptionParser

# setup the log file name before importing the Pyro4 library

import Pyro4

""" check the status of a pydpiper pipeline by querying the server using its uri"""

if __name__ == '__main__':
    usage = "Usage: " + __file__ + " uri\n" + \
            "   or: " + __file__ + "--help"

    parser = OptionParser(usage)

    options, args = parser.parse_args()

    if len(args) != 1:
        parser.error("please specify the uri file")
        
    uri_file = args[0]

    # find the server
    try:
        uf = open(uri_file)
        serverURI = Pyro4.URI(uf.readline())
        uf.close()
    except:
        print "There is a problem opening the specified uri file: %s" % uri_file
        raise

    proxyServer = Pyro4.Proxy(serverURI)

    # total number of stages in the pipeline:
    numStages = proxyServer.getTotalNumberOfStages()
    processedStages = proxyServer.getNumberProcessedStages()
    print "Total number of stages in the pipeline: ", numStages
    print "Number of stages already processed:     ", processedStages, "\n"

    # some info about executors
    runningClients = proxyServer.getNumberOfRunningClients()
    waitingClients = proxyServer.getNumberOfQueuedClients()
    print "Number of active clients:               ", runningClients
    print "Number of clients waiting in the queue: ", waitingClients, "\n"

    # stages currently running:
    runningStagesList = proxyServer.getCurrentlyRunningStages()
    print "Currently running stages: "
    for stage in runningStagesList:
        print stage, " ", proxyServer.getStageCommand(stage)

    # currently runnable jobs:
    numberRunnableStages = proxyServer.getNumberRunnableStages()
    print "\n\nNumber of runnable stages:               ", numberRunnableStages, "\n"



    
