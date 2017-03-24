#!/usr/bin/env python

from datetime import datetime
from os.path import isdir, basename
from os import mkdir
import os
import subprocess

# FIXME huge hack around not being able to pretty-print
# flags back out of a parsed representation
# fix: form the parser from an iterable data structure of args
# and consult this
# TODO has terrible performance/complexity
def remove_flags(flags, args):
    new_args = args[:] # copy for politeness
    for ix, arg in reversed(list(enumerate(new_args))):
        for flag in flags:
            if flag in arg: #re.search(flag, arg):
                # matches? (argparse uses prefix matching...try to catch -
                # ideally we'd actually consult a table of legal arguments)
                if '=' in arg:
                    # sys.argv has form [..., '--num-executors=3', ...]
                    new_args.pop(ix)
                elif len(new_args) - 1 == ix or new_args[ix+1][0] == '-':  # FIXME TOTAL HACK
                    # sys.argv has form [..., '--boolean-flag'] ( + [optionally] ['--other-flag', ...])
                # ... hopefully another flag, not a negative number ...
                    new_args.pop(ix)
                else:
                    # sys.argv has form [..., '--num-executors', '3', ...]
                    new_args.pop(ix)
                    new_args.pop(ix)
    return new_args

def timestr_to_secs(ts):
    # TODO replace with a library function
    # TODO put into a util module
    try:
        h, m, s = ts.split(':')
        return 3600 * int(h) + 60 * int(m) + int(s)
    except:
        raise Exception("invalid (H...)HH:MM:SS timestring: %s" % ts)

class runOnQueueingSystem():
    def __init__(self, options, sysArgs=None):
        #Note: options are the same as whatever is in calling program
        #Options MUST also include standard pydpiper options
        # self.options = options would be easier than this manual unpacking
        # for vars that don't have much 'logic' associated with them ...
        self.job_lifetime = timestr_to_secs(options.execution.time or '48:00:00')
        self.mem = options.execution.mem
        self.max_walltime = options.execution.max_walltime
        self.min_walltime = options.execution.min_walltime
        self.procs = options.execution.proc
        self.ppn = options.execution.ppn
        self.queue_name = options.execution.queue_name or options.execution.queue
        self.queue_type = options.execution.queue_type
        self.executor_start_delay = options.execution.executor_start_delay
        # TODO use self.time to compute better time_to_accept_jobs?
        self.time_to_accept_jobs = options.execution.time_to_accept_jobs
        self.arguments = sysArgs #sys.argv in calling program
        self.numexec = options.execution.num_exec
        self.ns = options.execution.use_ns
        self.uri_file = options.execution.urifile
        if self.uri_file is None:
            self.uri_file = os.path.abspath(os.path.join(os.curdir, "uri"))
        self.jobDir = os.path.abspath(os.path.join(os.curdir, "pbs-jobs"))
        if not isdir(self.jobDir):
            mkdir(self.jobDir) 
        if self.arguments is None:
            self.jobName = "pydpiper"
        else:
            executablePath = os.path.abspath(self.arguments[0])
            self.jobName = basename(executablePath)
        self.prologue_file = options.execution.prologue_file
    def buildMainCommand(self):
        """Re-construct main command to be called in pbs script, adding --local flag"""
        reconstruct = ""
        if self.arguments:
            reconstruct += ' '.join(remove_flags(['--num-exec', '--mem',
                                                  '--time-to-seppuku'],
                                                 self.arguments))
        reconstruct += " --local --num-executors=1 --time-to-seppuku=%d " \
                         % self.max_walltime
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
        # For multiple executors, this will be called multiple times.
        execId = "-executor-" + str(i) + "-"
        self.constructAndSubmitJobFile(execId, time, isMainFile=False, after=after)
    def addHeaderAndCommands(self, time, isMainFile):
        """Constructs header and commands for pbs script, based on options input from calling program"""
        self.jobFile.write("#!/bin/bash\n")
        requestNodes = 1
        name = self.jobName
        launchExecs = not isMainFile
        if isMainFile:
            # Number of nodes used depends on:
            # 1) number of available processors per node
            # 2) number of processes per executor
            nodes = divmod(self.procs, self.ppn)
            if self.numexec == 0:
                name += "-no-executors"
            elif self.numexec == 1:
                name += "-all"
                halfPpn = self.ppn/2
                if nodes[1] <= halfPpn and nodes[0] != 0:
                    requestNodes = nodes[0]
                else:
                    requestNodes = nodes[0] + 1               
            else:  
                name += "-plus-exec" 
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
            self.jobFile.write(self.buildMainCommand())
            self.jobFile.write(" &\n\n")
        if launchExecs:
            self.jobFile.write("sleep %s\n" %
                               self.executor_start_delay)
            cmd = "pipeline_executor.py --local --num-executors=1 "
            cmd += ' '.join(remove_flags(['--num-exec'], self.arguments[1:]))
            cmd += ' &\n\n'
            self.jobFile.write(cmd)
    def completeJobFile(self):
        """Completes pbs script--wait for background jobs to terminate as per scinet wiki"""
        self.jobFile.write("wait\n")
        self.jobFile.write("rm -f /dev/shm/* 2>/dev/null\n")
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
        cmd += ['-Wumask=0137', self.jobFileName]
        out = subprocess.check_output(cmd)
        jobId = out.strip()
        print(' '.join(cmd))
        print(jobId)
        print("Submitted!")
        return jobId
