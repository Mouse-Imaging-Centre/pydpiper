#!/usr/bin/env python

from datetime import datetime
from os.path import isdir, basename
from os import mkdir
import os
import re
import subprocess

SLEEP_TIME = 5

class runOnQueueingSystem():
    def __init__(self, options, sysArgs=None):
        #Note: options are the same as whatever is in calling program
        #Options MUST also include standard pydpiper options
        # TODO we can't override --scinet mem/procs with --mem/--procs since
        # we don't know if the values in options.mem/proc are supplied by the 
        # user or are defaults ... to fix this, we could use None as the 
        # default and set 6G later or use optargs in a more sophisticated way.
        if options.scinet:
            self.mem = 14
            self.procs = 8
            self.ppn = 8
            self.queue_name = options.queue_name or options.queue or "batch"
            self.queue_type = "pbs"
        else:
            self.mem = options.mem
            self.procs = options.proc
            self.ppn = options.ppn
            self.queue_name = options.queue_name or options.queue
            self.queue_type = options.queue or options.queue_type
        self.arguments = sysArgs #sys.argv in calling program
        self.numexec = options.num_exec 
        self.time = options.time or "2:00:00:00"      
        self.ns = options.use_ns
        self.uri_file = options.urifile
        if self.uri_file is None:
            self.uri_file = os.path.abspath(os.curdir + "/" + "uri")
        self.jobDir = os.environ["HOME"] + "/pbs-jobs"
        if not isdir(self.jobDir):
            mkdir(self.jobDir) 
        if self.arguments is None:
            self.jobName = "pydpiper"
        else:
            executablePath = os.path.abspath(self.arguments[0])
            self.jobName = basename(executablePath)
    def relevant(arg):
        if re.search("--queue-name", arg):
            return True
        elif re.search("--(num-executors|proc|queue|mem|time|ppn)", arg):
            return False
        else:
            return True
    def buildMainCommand(self):
        """Re-construct main command to be called in pbs script, removing un-necessary arguments"""
        def relevant(arg):
            if re.search("--(num-executors|proc|queue|mem|time|ppn|scinet)", arg):
                return False
            else:
                return True
        reconstruct = ""
        if self.arguments:
            reconstruct += ' '.join(filter(relevant, self.arguments)) + " --num-executors=0 "
        return reconstruct
    def constructAndSubmitJobFile(self, identifier, isMainFile, depends):
        """Construct the bulk of the pbs script to be submitted via qsub"""
        now = datetime.now()  
        jobName = self.jobName + identifier + now.strftime("%Y%m%d-%H%M%S%f") + ".job"
        self.jobFileName = self.jobDir + "/" + jobName
        self.jobFile = open(self.jobFileName, "w")
        self.addHeaderAndCommands(isMainFile)
        self.completeJobFile()
        jobId = self.submitJob(depends = depends)
        return jobId
    def createAndSubmitPbsScripts(self): 
        """Creates pbs script(s) for main program and separate executors, if needed"""       
        serverJobId = self.createAndSubmitMainJobFile()
        if self.numexec >= 2:
            for i in range(1, self.numexec):
                self.createAndSubmitExecutorJobFile(i, serverJobId)
    def createAndSubmitMainJobFile(self): 
        return self.constructAndSubmitJobFile("-pipeline-", isMainFile=True, depends = None)
    def createAndSubmitExecutorJobFile(self, i, serverJobId):
        # This is called directly from pipeline_executor
        # For multiple executors, this will be called multiple-times.
        execId = "-executor-" + str(i) + "-"
        self.constructAndSubmitJobFile(execId, isMainFile=False, depends=serverJobId)
    def addHeaderAndCommands(self, isMainFile):
        """Constructs header and commands for pbs script, based on options input from calling program"""
        self.jobFile.write("#!/bin/bash" + "\n")
        requestNodes = 1
        execProcs = self.procs
        mainCommand = ""
        name = self.jobName
        launchExecs = True   
        if isMainFile:
            # Number of nodes used depends on:
            # 1) number of available processors per node
            # 2) number of processes per executor
            nodes = divmod(self.procs, self.ppn)
            mainCommand = self.buildMainCommand()
            if self.numexec == 0:
                launchExecs = False
                name += "-no-executors"
            elif self.numexec == 1:
                name += "-all"
                execProcs = self.procs
                halfPpn = self.ppn/2
                if nodes[1] <= halfPpn and nodes[0] != 0:
                    requestNodes = nodes[0]
                else:
                    requestNodes = nodes[0] + 1               
            else:  
                name += "-plus-exec" 
                execProcs = self.procs
        else:
            name += "-executor"
        self.jobFile.write("#PBS -l nodes=%d:ppn=%d,walltime=%s\n" % (requestNodes, self.ppn, self.time))
        self.jobFile.write("#PBS -N %s\n" % name)
        self.jobFile.write("#PBS -q %s\n" % self.queue_name)
        self.jobFile.write("module load gcc intel python\n\n")
        self.jobFile.write("cd $PBS_O_WORKDIR\n\n") # jobs start in $HOME; $PBS_O_WORKDIR is the submission directory
        if mainCommand:
            self.jobFile.write(self.buildMainCommand())
            self.jobFile.write(" &\n\n")
        if launchExecs:
            # sleep to ensure that pipeline server has time to start
            self.jobFile.write("sleep %s\n\n" % SLEEP_TIME)
            self.jobFile.write("pipeline_executor.py --num-executors=1 --uri-file=%s --proc=%d --mem=%.2f" % (self.uri_file, execProcs, self.mem))
            if self.ns:
                self.jobFile.write(" --use-ns ")
            self.jobFile.write(" &\n\n")
    def completeJobFile(self):
        """Completes pbs script--wait for background jobs to terminate as per scinet wiki"""
        self.jobFile.write("wait" + "\n")
        self.jobFile.close()
    def submitJob(self, depends=None): 
        """Submit job to batch queueing system"""
        if depends is not None:
            out = subprocess.check_output(['qsub', '-Wafter:' + depends, self.jobFileName])
        else:
            out = subprocess.check_output(['qsub', self.jobFileName])
        jobId = out.strip()
        print(jobId)
        print("Submitted!")
        return jobId
