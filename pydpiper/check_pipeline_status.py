#!/usr/bin/env python

from __future__ import print_function
import argparse
import signal

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import Pyro4

""" check the status of a pydpiper pipeline by querying the server using its uri"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("uri_file", type=str, help="file containing server's URI")

    options = parser.parse_args()

    uri_file = options.uri_file

    # find the server
    try:
        uf = open(uri_file)
        serverURI = Pyro4.URI(uf.readline())
        uf.close()
    except:
        print("There is a problem opening the specified uri file: %s" % uri_file)
        raise

    proxyServer = Pyro4.Proxy(serverURI)

    # total number of stages in the pipeline:
    numStages = proxyServer.getTotalNumberOfStages()
    processedStages = proxyServer.getNumberProcessedStages()
    print("Total number of stages in the pipeline: ", numStages)
    print("Number of stages already processed:     ", processedStages, "\n")

    # some info about executors
    runningClients = proxyServer.getNumberOfRunningClients()
    waitingClients = proxyServer.getNumberOfQueuedClients()
    print("Number of active clients:               ", runningClients)
    print("Number of clients waiting in the queue: ", waitingClients, "\n")

    # stages currently running:
    runningStagesList = proxyServer.getCurrentlyRunningStages()
    print("Currently running stages: ")
    for stage in runningStagesList:
        print(stage, " ", proxyServer.getStageCommand(stage))

    # currently runnable jobs:
    numberRunnableStages = proxyServer.getNumberRunnableStages()
    print("\nNumber of runnable stages:               ", numberRunnableStages, "\n")

    # number of failed stages:
    numberFailedStages = proxyServer.getNumberFailedStages()
    print("\nNumber of failed stages:                 ", numberFailedStages)
    # number of lost/died executors:
    numberFailedExecutors = proxyServer.getNumberFailedExecutors()
    print("Number of failed/lost/dead executors:    ", numberFailedExecutors, "\n")


    # memory requirements for runnable stages:
    memArray = proxyServer.getMemoryRequirementsRunnable()
    print("\nMemory requirement of runnable stages:   ", memArray)
    # memory available in registered executors:
    memAvailable = proxyServer.getMemoryAvailableInClients()
    print("Memory available in registered clients:  ", memAvailable, "\n")
    
