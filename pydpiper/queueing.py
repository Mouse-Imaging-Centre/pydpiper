#!/usr/bin/env python

import Pyro.core
import Pyro.naming
from optparse import OptionParser
from datetime import datetime
from os.path import isdir, abspath, basename
from os import mkdir
import os
import time
import networkx as nx
import re

Pyro.config.PYRO_MOBILE_CODE=1

class runOnQueueingSystem():
    def __init__(self, options, ppn=8, time="2:00:00:00", sysArgs=None):
        #Note: options are the same as whatever is in calling program
        #Options MUST also include standard pydpiper options
        self.arguments = sysArgs #sys.argv in calling program
        self.numexec = options.num_exec 
        self.mem = options.mem
        self.proc = options.proc
        self.queue = options.queue       
        self.ns = options.use_ns
        self.uri = options.urifile
        if self.uri==None:
            self.uri = os.path.abspath(os.curdir + "/" + "uri")
        self.jobDir = os.environ["HOME"] + "/pbs-jobs"
        if not isdir(self.jobDir):
            mkdir(self.jobDir) 
        if self.arguments==None:
            self.jobName = "pipeline"
        else:
            executablePath = os.path.abspath(self.arguments[0])
            self.jobName = basename(executablePath)
        self.ppn = ppn
        self.time = time
    def buildMainCommand(self):
        """Re-construct main command to be called in pbs script, removing un-necessary arguments"""
        reconstruct = ""
        for i in range(len(self.arguments)):
            if not (re.search("--num-executors", self.arguments[i]) or re.search("--proc", self.arguments[i])
                or re.search("--queue", self.arguments[i]) or re.search("--mem", self.arguments[i])):
                reconstruct += self.arguments[i]
                reconstruct += " "
        return reconstruct
    def constructJobFile(self, identifier, isMainFile):
        """Construct the bulk of the pbs script to be submitted via qsub"""
        now = datetime.now()  
        jobName = self.jobName + identifier + now.strftime("%Y%m%d-%H%M%S") + ".job"
        self.jobFileName = self.jobDir + "/" + jobName
        self.jobFile = open(self.jobFileName, "w")
        self.addHeaderAndCommands(isMainFile)
        self.completeJobFile()
        self.submitJob()
    def createPbsScripts(self): 
        """Creates pbs script(s) for main program and separate executors, if needed"""       
        self.createMainJobFile()
        if self.numexec >=2:
            for i in range(1, self.numexec):
                self.createExecutorJobFile(i)
    def createMainJobFile(self): 
        self.constructJobFile("-pipeline-", True)
    def createExecutorJobFile(self, i):
        # This is called directly from pipeline_executor
        # For multiple executors, this will be called multiple-times.
        execId = "-executor-" + str(i) + "-"
        self.constructJobFile(execId, False)
    def addHeaderAndCommands(self, isMainFile):
        """Constructs header and commands for pbs script, based on options input from calling program"""
        self.jobFile.write("#!/bin/bash" + "\n")
        requestNodes = 1
        execProcs = self.ppn
        mainCommand = ""
        name = self.jobName + "-executor"
        launchExecs = True   
        if isMainFile:
            # Number of nodes used depends on:
            # 1) number of available processors per node
            # 2) number of processes per executor
            nodes = divmod(self.proc, self.ppn)
            mainCommand = self.buildMainCommand()
            if self.numexec == 0:
                launchExecs = False
            elif self.numexec == 1:
                name = self.jobName + "-pipeline-all"
                execProcs = self.proc
                halfPpn = self.ppn/2
                if nodes[1]<=halfPpn:
                    requestNodes = nodes[0]
                else:
                    requestNodes = nodes[0] + 1               
            else:  
                name = self.jobName + "-pipeline-plus-exec" 
                execProcs = self.proc 
            
        self.jobFile.write("#PBS -l nodes=%d:ppn=%d,walltime=%s\n" % (requestNodes, self.ppn, self.time))
        self.jobFile.write("#PBS -N %s\n\n" % name)
        self.jobFile.write("cd $PBS_O_WORKDIR\n\n")
        if mainCommand:
            self.jobFile.write(self.buildMainCommand())
            self.jobFile.write("&\n\n")
            self.jobFile.write("sleep 60") # sleep to ensure that PyroServer has time to start
            self.jobFile.write("\n\n")
        if launchExecs:
            self.jobFile.write("pipeline_executor.py --uri-file=%s --proc=%d --mem=%.2f" % (self.uri, execProcs, self.mem))
            if self.ns:
                self.jobFile.write(" --use-ns")
            self.jobFile.write(" &\n")
    def completeJobFile(self):
        """Complets pbs script--wait included as per scinet wiki"""
        self.jobFile.write("wait" + "\n")
        self.jobFile.close()
    def submitJob(self): 
        """Submit job to batch queueing system"""
        os.system("qsub " + self.jobFileName)  
        print "Submitted!"
