#!/usr/bin/env python

from datetime import datetime
from os.path import isdir, basename
from os import mkdir
import os
import re
import subprocess

# FIXME
SERVER_START_TIME = 50
# TODO instead of hard-coding SciNet min/max times for debug/batch queues,
# add extra options/env. vars for these
SCINET_MIN_LIFETIME = 15 * 60
SCINET_MAX_LIFETIME = 2  * 24 * 3600

class runOnQueueingSystem():
    def __init__(self, options, sysArgs=None):
        #Note: options are the same as whatever is in calling program
        #Options MUST also include standard pydpiper options
        # FIXME when using --scinet, we can't override options values having numerical defaults
        # (mem/procs/time-to-accept-jobs/time-to-seppuku) with --mem/--procs/... since
        # we don't know if the values in options are supplied by the 
        # user or are defaults ... to fix this, we could use None as the 
        # default and set non-scinet defaults later or use optargs in a more sophisticated way(?)
        self.timestr = options.time or '48:00:00'
        try:
            h,m,s = self.timestr.split(':')
        except:
            raise Exception("invalid (H)HH:MM:SS timestring: %s" % self.timestr)
        self.job_lifetime = 3600 * int(h) + 60 * int(m) + int(s)
        # TODO use self.time to give better time_to_accept_jobs
        # and to compute number of generations of scripts to submit
        if options.scinet:
            self.mem = 14
            self.procs = 8
            self.ppn = 8
            self.queue_name = options.queue_name or options.queue or "batch"
            self.queue_type = "pbs"
            self.time_to_accept_jobs = 47 * 60 # TODO compute based on lifetime
        else:
            self.mem = options.mem
            self.procs = options.proc
            self.ppn = options.ppn
            self.queue_name = options.queue_name or options.queue
            self.queue_type = options.queue or options.queue_type
            self.time_to_accept_jobs = options.time_to_accept_jobs
        self.arguments = sysArgs #sys.argv in calling program
        self.numexec = options.num_exec 
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
    def buildMainCommand(self):
        """Re-construct main command to be called in pbs script, removing un-necessary arguments"""
        def relevant(arg):
            if re.search("--(num-executors|proc|queue|mem|time|ppn|scinet)", arg):
                return False
            else:
                return True
        reconstruct = ""
        if self.arguments:
            reconstruct += ' '.join(filter(relevant, self.arguments))
        #TODO pass time as an arg to buildMainCmd, decrement by max_scinet_time - 5:00 with each generation
        reconstruct += " --num-executors=0 " + " --lifetime=%d " % self.job_lifetime
        return reconstruct
    def constructAndSubmitJobFile(self, identifier, time, isMainFile, depends):
        """Construct the bulk of the pbs script to be submitted via qsub"""
        now = datetime.now()  
        jobName = self.jobName + identifier + now.strftime("%Y%m%d-%H%M%S%f") + ".job"
        self.jobFileName = self.jobDir + "/" + jobName
        self.jobFile = open(self.jobFileName, "w")
        self.addHeaderAndCommands(time, isMainFile)
        self.completeJobFile()
        jobId = self.submitJob(jobName, depends)
        return jobId
    def createAndSubmitPbsScripts(self): 
        """Creates pbs script(s) for main program and separate executors, if needed"""       
        time_remaining = self.job_lifetime
        serverJobId = None
        while time_remaining > 0:
            t = min(max(SCINET_MIN_LIFETIME, time_remaining), SCINET_MAX_LIFETIME)
            time_remaining -= t
            serverJobId = self.createAndSubmitMainJobFile(time=t, depends=serverJobId)
            if self.numexec >= 2:
                for i in range(1, self.numexec):
                    self.createAndSubmitExecutorJobFile(i, time=t,depends=serverJobId)
            # in principle a server could overlap the previous generation of clients,
            # but at present the clients just die within seconds
    def createAndSubmitMainJobFile(self,time,depends=None): 
        return self.constructAndSubmitJobFile("-pipeline-",time,isMainFile=True,depends=depends)
    def createAndSubmitExecutorJobFile(self, i, time, depends):
        # This is called directly from pipeline_executor
        # For multiple executors, this will be called multiple-times.
        execId = "-executor-" + str(i) + "-"
        self.constructAndSubmitJobFile(execId, time, isMainFile=False, depends=depends)
    def addHeaderAndCommands(self, time, isMainFile):
        """Constructs header and commands for pbs script, based on options input from calling program"""
        self.jobFile.write("#!/bin/bash\n")
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
        m,s = divmod(time,60)
        h,m = divmod(m,60)
        timestr = "%d:%02d:%02d" % (h,m,s)
        self.jobFile.write("#PBS -l nodes=%d:ppn=%d,walltime=%s\n" % (requestNodes, self.ppn, timestr))
        self.jobFile.write("#PBS -N %s\n" % name)
        self.jobFile.write("#PBS -q %s\n" % self.queue_name)
        # the `module` shell procedure likes to return 0 when it shouldn't (and fails when `|`ed for some reason), so redirect to a file and grep for an error message:
        # TODO modules (or even calls to module) shouldn't be hard-coded
        self.jobFile.write("export fh=$(mktemp)\n")
        self.jobFile.write("module load gcc intel/15.0 python/2.7.8 gotoblas hdf5 gnuplot Xlibraries octave 2> $fh \n")
        self.jobFile.write("cat $fh | tee /dev/stderr | grep 'ERROR' && exit 13\n")
        self.jobFile.write("rm $fh \n")
        self.jobFile.write("cd $PBS_O_WORKDIR\n\n") # jobs start in $HOME; $PBS_O_WORKDIR is the submission directory
        if mainCommand:
            self.jobFile.write(self.buildMainCommand())
            self.jobFile.write(" &\n\n")
        if launchExecs:
            self.jobFile.write("sleep %s\n" % SERVER_START_TIME)
            self.jobFile.write("pipeline_executor.py --num-executors=1 --uri-file=%s --proc=%d --mem=%.2f --time-to-accept-jobs=%d" % (self.uri_file, execProcs, self.mem, self.time_to_accept_jobs))
            if self.ns:
                self.jobFile.write(" --use-ns ")
            self.jobFile.write(" &\n\n")
    def completeJobFile(self):
        """Completes pbs script--wait for background jobs to terminate as per scinet wiki"""
        self.jobFile.write("wait" + "\n")
        self.jobFile.close()
    def submitJob(self, jobName, depends):
        """Submit job to batch queueing system"""
        os.environ['PYRO_LOGFILE'] = jobName + '.log'
        # use -V to get all (Pyro) variables, incl. PYRO_LOGFILE
        cmd = ['qsub', '-o', jobName + '-eo.log', '-V']
        if depends is not None:
            cmd += ['-Wafter:' + depends]
        cmd += [self.jobFileName]
        out = subprocess.check_output(cmd)
        jobId = out.strip()
        print(cmd)
        print(jobId)
        print("Submitted!")
        return jobId
