#!/usr/bin/env python

from datetime import datetime
from os.path import isdir, basename
from os import mkdir
import os
import re
import subprocess

# FIXME
SERVER_START_TIME = 50

# FIXME huge hack - fix: form the parser from an iterable data structure of args
# and consult this
def remove_num_exec(args):
    args = args[:] # copy for politeness
    for ix, arg in enumerate(args):
        if re.search('--num-exec', arg):
            # matches? (argparse uses prefix matching...try to catch -
            # ideally we'd actually consult a table of legal arguments)
            if re.search('=', arg):
                # sys.argv has form [..., '--num-executors=3', ...]
                args.pop(ix)
            else:
                # sys.argv has form [..., '--num-executors', '3', ...]
                args.pop(ix)
                args.pop(ix)
    return args

class runOnQueueingSystem():
    def __init__(self, options, sysArgs=None):
        #Note: options are the same as whatever is in calling program
        #Options MUST also include standard pydpiper options
        # self.options = options would be easier than this manual unpacking
        # for vars that don't have much 'logic' associated with them ...
        self.timestr = options.time or '48:00:00'
        try:
            h,m,s = self.timestr.split(':')
        except:
            raise Exception("invalid (H)HH:MM:SS timestring: %s" % self.timestr)
        self.job_lifetime = 3600 * int(h) + 60 * int(m) + int(s)
        self.mem = options.mem
        self.max_walltime = options.max_walltime
        self.min_walltime = options.min_walltime
        self.procs = options.proc
        self.ppn = options.ppn
        self.queue_name = options.queue_name or options.queue
        self.queue_type = options.queue or options.queue_type
        # TODO use self.time to compute better time_to_accept_jobs?
        self.time_to_accept_jobs = options.time_to_accept_jobs
        self.arguments = sysArgs #sys.argv in calling program
        self.numexec = options.num_exec 
        self.ns = options.use_ns
        self.uri_file = options.urifile
        if self.uri_file is None:
            self.uri_file = os.path.abspath(os.path.join(os.curdir, "uri"))
        self.jobDir = os.path.join(os.environ["HOME"], "pbs-jobs")
        if not isdir(self.jobDir):
            mkdir(self.jobDir) 
        if self.arguments is None:
            self.jobName = "pydpiper"
        else:
            executablePath = os.path.abspath(self.arguments[0])
            self.jobName = basename(executablePath)
        self.prologue_file = options.prologue_file
    def buildMainCommand(self, t):
        """Re-construct main command to be called in pbs script, adding --local flag"""
        reconstruct = ""
        if self.arguments:
            reconstruct += ' '.join(remove_num_exec(self.arguments))
        reconstruct += " --local --num-executors=0 " # + " --lifetime=%d " % t # TODO remove
        return reconstruct
    def constructAndSubmitJobFile(self, identifier, time, isMainFile, after=None, afterany=None):
        """Construct the bulk of the pbs script to be submitted via qsub"""
        now = datetime.now()  
        jobName = self.jobName + identifier + now.strftime("%Y%m%d-%H%M%S%f") + ".job"
        self.jobFileName = os.path.join(self.jobDir, jobName)
        self.jobFile = open(self.jobFileName, "w")
        self.addHeaderAndCommands(time, isMainFile)
        self.completeJobFile()
        jobId = self.submitJob(jobName, after, afterany)
        return jobId
    def createAndSubmitPbsScripts(self): 
        """Creates pbs script(s) for main program and separate executors, if needed"""       
        time_remaining = self.job_lifetime
        serverJobId = None
        while time_remaining > 0:
            t = max(self.min_walltime, time_remaining)
            if self.max_walltime is not None:
                t = min(t, self.max_walltime)
            time_remaining -= t
            serverJobId = self.createAndSubmitMainJobFile(time=t, afterany=serverJobId)
            if self.numexec >= 2:
                for i in range(1, self.numexec):
                    self.createAndSubmitExecutorJobFile(i, time=t, after=serverJobId)
            # in principle a server could overlap the previous generation of clients,
            # but at present the clients just die within seconds
    def createAndSubmitMainJobFile(self,time, afterany=None):
        return self.constructAndSubmitJobFile("-pipeline-",time, isMainFile=True, afterany=afterany)
    def createAndSubmitExecutorJobFile(self, i, time, after):
        # This is called directly from pipeline_executor
        # For multiple executors, this will be called multiple-times.
        execId = "-executor-" + str(i) + "-"
        self.constructAndSubmitJobFile(execId, time, isMainFile=False, after=after)
    def addHeaderAndCommands(self, time, isMainFile):
        """Constructs header and commands for pbs script, based on options input from calling program"""
        self.jobFile.write("#!/bin/bash\n")
        requestNodes = 1
        execProcs = self.procs
        name = self.jobName
        launchExecs = True   
        if isMainFile:
            # Number of nodes used depends on:
            # 1) number of available processors per node
            # 2) number of processes per executor
            nodes = divmod(self.procs, self.ppn)
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
        if self.prologue_file is not None:
            try:
                with open(self.prologue_file, 'r') as fh:
                    for l in fh:
                        self.jobFile.write(l)
                self.jobFile.write('\n')
            except:
                print("Failed copying prologue script into submit script")
                raise
        # cd from $HOME into the submission directory:
        self.jobFile.write("cd $PBS_O_WORKDIR\n\n")
        if isMainFile:
            self.jobFile.write(self.buildMainCommand(time))
            self.jobFile.write(" &\n\n")
        if launchExecs:
            self.jobFile.write("sleep %s\n" % SERVER_START_TIME)
            cmd = "pipeline_executor.py --local --num-executors=1 "
            cmd += ' '.join(remove_num_exec(self.arguments[1:])) + ' &\n\n'
            self.jobFile.write(cmd)
    def completeJobFile(self):
        """Completes pbs script--wait for background jobs to terminate as per scinet wiki"""
        self.jobFile.write("wait" + "\n")
        self.jobFile.close()
    def submitJob(self, jobName, after=None, afterany=None):
        """Submit job to batch queueing system"""
        os.environ['PYRO_LOGFILE'] = jobName + '.log'
        # use -V to get all (Pyro) variables, incl. PYRO_LOGFILE
        cmd = ['qsub', '-o', jobName + '-o.log', '-e', jobName + '-e.log', '-V']
        if after is not None:
            cmd += ['-Wdepend=after:' + after]
        if afterany is not None:
            cmd += ['-Wdepend=afterany:' + afterany]
        cmd += [self.jobFileName]
        out = subprocess.check_output(cmd)
        jobId = out.strip()
        print(' '.join(cmd))
        print(jobId)
        print("Submitted!")
        return jobId
