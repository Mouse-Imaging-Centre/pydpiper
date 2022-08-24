#!/usr/bin/env python

import os
import argparse
import signal
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import Pyro5

""" check the status of a pydpiper pipeline by querying the server using its uri"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("uri_file", type=str, help="file containing server's URI. If not given, defaults to *_uri", nargs="?")

    options = parser.parse_args()

    uri_file = options.uri_file

    if uri_file == None:
        wd = Path(".")
        uri_files = [path.name for path in wd.ls() if "_uri" in path.name]
        if len(uri_files) == 1:
            uri_file = uri_files[0]
            print("Using uri file: %s" % uri_file)
        elif len(uri_files) == 0:
            raise ValueError("No uri_files found in your current working directory %s" % os.getcwd())
        else:
            raise ValueError("Found multiple uri_files: %s" % uri_files)

    # find the server
    try:
        uf = open(uri_file)
        serverURI = Pyro5.api.URI(uf.readline())
        uf.close()
    except:
        print("There is a problem opening the specified uri file: %s" % uri_file)
        raise

    proxyServer = Pyro5.api.Proxy(serverURI)

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
    runningStages = proxyServer.getCurrentlyRunningStages()
    print("Currently running stages (%d): " % len(runningStages))
    for stage in runningStages:
        print("%s\t%s\n" % (stage, proxyServer.getStageCommand(stage)))

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
    print("\nMemory requirement of runnable stages: %s" % memArray)
    # memory available in registered executors:
    memAvailable = proxyServer.getMemoryAvailableInClients()
    print("Memory available in registered clients: %s \n" %
          [round(x, 4) for x in memAvailable])
