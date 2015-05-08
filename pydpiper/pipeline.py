#!/usr/bin/env python

from __future__ import print_function

import networkx as nx
import os
import sys
import signal
import socket
import time
import re
from datetime import datetime
from subprocess import call, check_output
from shlex import split
from multiprocessing import Process, Event
import logging

# TODO move this and Pyro4 imports down into launchServer where pipeline name is available?
os.environ["PYRO_LOGLEVEL"] = os.getenv("PYRO_LOGLEVEL", "INFO")
os.environ["PYRO_LOGFILE"]  = os.path.splitext(os.path.basename(__file__))[0] + ".log"
# TODO name the server logfile more descriptively

import Pyro4
import pipeline_executor as pe

Pyro4.config.SERVERTYPE = pe.Pyro4.config.SERVERTYPE

LOOP_INTERVAL = 5
STAGE_RETRY_INTERVAL = 1

logger = logging.getLogger(__name__)

sys.excepthook = Pyro4.util.excepthook


class PipelineFile():
    def __init__(self, filename):
        self.filename = filename
        self.setType()
    def setType(self):
        self.fileType = None
    def __repr__(self):
        return(self.filename)

class InputFile(PipelineFile):
    def setType(self):
        self.fileType = "input"

class OutputFile(PipelineFile):
    def setType(self):
        self.fileType = "output"

class LogFile(PipelineFile):
    def setType(self):
        self.fileType = "log"

    """
    The executor client class:
    client:    URI string to represent the executor
    maxmemory: the total amount of memory the executor has at its disposal

    will be used to keep track of the stages it's running and whether
    it's still alive (based on a periodic heartbeat)
    """
class ExecClient():
    def __init__(self, client, maxmemory):
        self.clientURI = client
        self.maxmemory = maxmemory
        self.running_stages = set([])
        self.timestamp = time.time()

class PipelineStage():
    def __init__(self):
        self.mem = 2.0 # default memory allotted per stage
        self.procs = 1 # default number of processors per stage
        self.inputFiles = [] # the input files for this stage
        self.outputFiles = [] # the output files for this stage
        self.logFile = None # each stage should have only one log file
        self.status = None
        self.name = ""
        self.colour = "black" # used when a graph is created of all stages to colour the nodes
        self.number_retries = 0 

    def isFinished(self):
        return self.status == "finished"
    def setRunning(self):
        self.status = "running"
    def setFinished(self):
        self.status = "finished"
    def setFailed(self):
        self.status = "failed"
    def setNone(self):
        self.status = None
    def setMem(self, mem):
        self.mem = mem
    def getMem(self):
        return self.mem
    def setProcs(self, num):
        self.procs = num
    def getProcs(self):
        return self.procs
    def getHash(self):
        return(hash("".join(self.outputFiles) + "".join(self.inputFiles)))
    def __eq__(self, other):
        return self.inputFiles == other.inputFiles and self.outputFiles == other.outputFiles
    def __ne__(self, other):
        return not(self.__eq__(other))
    def getNumberOfRetries(self):
        return self.number_retries
    def incrementNumberOfRetries(self):
        self.number_retries += 1

class CmdStage(PipelineStage):
    def __init__(self, argArray):
        PipelineStage.__init__(self)
        self.argArray = argArray # the raw input array
        self.cmd = [] # the input array converted to strings
        self.parseArgs()
        self.checkLogFile()
        # functions to be called when the stage becomes runnable
        # (these might be called multiple times, so should be benign
        # in some sense)
        self.runnable_hooks = []
        # functions to be called when a stage finishes
        self.finished_hooks = []
    def parseArgs(self):
        if self.argArray:
            for a in self.argArray:
                ft = getattr(a, "fileType", None)
                if ft == "input":
                    self.inputFiles.append(str(a))
                elif ft == "output":
                    self.outputFiles.append(str(a))
                self.cmd.append(str(a))
                self.name = self.cmd[0]
    def checkLogFile(self):
        if not self.logFile:
            self.logFile = self.name + "." + datetime.isoformat(datetime.now()) + ".log"
    def setLogFile(self, logFileName): 
        self.logFile = str(logFileName)
    def execStage(self):
        of = open(self.logFile, 'a')
        of.write("Running on: " + socket.gethostname() + " at " + datetime.isoformat(datetime.now(), " ") + "\n")
        of.write(repr(self) + "\n")
        of.flush()

        args = split(repr(self))
        returncode = call(args, stdout=of, stderr=of, shell=False)
        of.close()
        return(returncode)

    def getHash(self):
        return(hash(" ".join(self.cmd)))
    def __repr__(self):
        return(" ".join(self.cmd))

class Pipeline():
    # TODO the way we initialize a pipeline is currently a bit gross, e.g.,
    # setting a bunch of instance variables after __init__ - the presence of a method
    # called `initialize` should be a hint that all is perhaps not well, but perhaps
    # there is indeed some information legitimately unavailable when we first construct
    def __init__(self,options=None):
        # the core pipeline is stored in a directed graph. The graph is made
        # up of integer indices
        self.G = nx.DiGraph()
        # an array of the actual stages (PipelineStage objects)
        self.stages = []
        self.nameArray = []
        # indices of the stages ready to be run
        self.runnable = set()
        # an array to keep track of stage memory requirements
        self.mem_req_for_runnable = []
        self.currently_running_stages = set()
        # the current stage counter
        self.counter = 0
        # hash to keep the output to stage association
        self.outputhash = {}
        # a hash per stage - computed from inputs and outputs or whole command
        self.stagehash = {}
        # an array containing stages that have been (successfully) processed
        self.processedStages = []
        self.failedStages = []
        # location of backup files for restart if needed
        self.backupFileLocation = None
        # table of registered clients (using ExecClient class instances) indexed by URI
        self.clients = {}
        # number of clients (executors) that have been launched by the server
        # we need to keep track of this because even though no (or few) clients
        # are actually registered, a whole bunch of them could be waiting in the
        # queue
        self.number_launched_and_waiting_clients = 0
        # clients we've lost contact with due to crash, etc.
        self.failed_executors = 0
        # main option hash, needed for the pipeline (server) to launch additional
        # executors during run time
        self.main_options = options
        # time to shut down, due to walltime or having completed all stages?
        # (use an event rather than a simple flag for shutdown notification
        # so that we can shut down even if a process is currently sleeping)
        self.shutdown_ev = Event()
        self.programName = None
        self.skipped_stages = 0
        self.verbose = 0
        # Handle to write out processed stages to
        self.finished_stages_fh = None
        
        if self.main_options:
            self.outputDir = self.main_options.output_directory 
            if not self.outputDir:
                self.outputDir = os.getcwd()
            # redirect the standard output to a text file
            serverLogFile = os.path.join(self.outputDir,self.main_options.pipeline_name + '_server_log_in_text')
            if self.main_options.queue_type == "pbs":
                # the 0 at the end indicates a 0 buffer, so 
                # things are printed right away
                sys.stdout = open(serverLogFile, 'a', 0) 
        
    # expose methods to get/set shutdown_ev via Pyro (setter not needed):
    def set_shutdown_ev(self):
        self.shutdown_ev.set()

    def get_shutdown_ev(self):
        return self.shutdown_ev

    def getNumberFailedExecutors(self):
        return self.failed_executors

    def getNumberFailedStages(self):
        return len(self.failedStages)

    def getTotalNumberOfStages(self):
        return len(self.stages)

    def getNumberProcessedStages(self):
        return len(self.processedStages)

    def getNumberOfRunningClients(self):
        return len(self.clients)

    def getNumberOfQueuedClients(self):
        return self.number_launched_and_waiting_clients

    def setVerbosity(self, verbosity):
        self.verbose = verbosity

    def getCurrentlyRunningStages(self):
        return self.currently_running_stages

    def getNumberRunnableStages(self):
        return len(self.runnable)

    def getMemoryRequirementsRunnable(self):
        return self.mem_req_for_runnable

    def getMemoryAvailableInClients(self):
        return [c.maxmemory for _, c in self.clients.iteritems()]

    def addStage(self, stage):
        """adds a stage to the pipeline"""
        # check if stage already exists in pipeline - if so, don't bother

        # check if stage exists - stage uniqueness defined by in- and outputs
        # for base stages and entire command for CmdStages
        h = stage.getHash()
        if self.stagehash.has_key(h):
            self.skipped_stages += 1
            #stage already exists - nothing to be done
        else: #stage doesn't exist - add it
            self.stagehash[h] = self.counter
            #self.statusArray[self.counter] = 'notstarted'
            self.stages.append(stage)
            self.nameArray.append(stage.name)
            # add all outputs to the output dictionary
            for o in stage.outputFiles:
                self.outputhash[o] = self.counter
            # add the stage's index to the graph
            self.G.add_node(self.counter, label=stage.name,color=stage.colour)
            self.counter += 1

    def setBackupFileLocation(self, outputDir=None):
        """Sets location of backup files."""
        if outputDir is None:
            # set backups in current directory if directory doesn't currently exist
            outputDir = os.getcwd()
        # TODO don't need this dir since we have only one backup file?
        #self.backupFileLocation = fh.createBackupDir(outputDir, self.main_options.pipeline_name)
        self.backupFileLocation = os.path.join(outputDir,
                                    self.main_options.pipeline_name
                                     + '_finished_stages')
    def addPipeline(self, p):
        if p.skipped_stages > 0:
            self.skipped_stages += p.skipped_stages
        for s in p.stages:
            self.addStage(s)

    def printStages(self, name):
        """Prints stages to a file, stage info to stdout"""

        fileForPrinting = os.path.abspath(os.curdir + "/" + str(name) + "_pipeline_stages.txt")
        pf = open(fileForPrinting, "w")
        for i in range(len(self.stages)):
            pf.write(str(i) + "  " + str(self.stages[i]) + "\n")
        pf.close()
        print("Total number of stages in the pipeline: ", len(self.stages))
                 
    def printNumberProcessedStages(self):
        print("Number of stages already processed:     ", len(self.processedStages))
                  
    def createEdges(self):
        """computes stage dependencies by examining their inputs/outputs"""
        starttime = time.time()
        # iterate over all nodes
        for i in self.G.nodes_iter():
            for ip in self.stages[i].inputFiles:
                # if the input to the current stage was the output of another
                # stage, add a directional dependence to the DiGraph
                if self.outputhash.has_key(ip):
                    self.G.add_edge(self.outputhash[ip], i)
        endtime = time.time()
        logger.info("Create Edges time: " + str(endtime-starttime))

    def computeGraphHeads(self):
        """adds stages with no incomplete predecessors to the runnable queue"""
        graphHeads = []
        for i in self.G.nodes_iter():
            if self.stages[i].isFinished() == False:
                """ either it has 0 predecessors """
                if len(self.G.predecessors(i)) == 0:
                    self.enqueue(i)
                    graphHeads.append(i)
                """ or all of its predecessors are finished """
                if len(self.G.predecessors(i)) != 0:
                    predfinished = True
                    for j in self.G.predecessors(i):
                        if self.stages[j].isFinished() == False:
                            predfinished = False
                    if predfinished == True:
                        self.enqueue(i)
                        graphHeads.append(i)
        logger.info("Graph heads: " + str(graphHeads))
    def getStage(self, i):
        """given an index, return the actual pipelineStage object"""
        return(self.stages[i])
    # getStage<...> are currently used instead of getStage due to previous bug; could revert:
    def getStageMem(self, i):
        return(self.stages[i].mem)
    def getStageProcs(self,i):
        return(self.stages[i].procs)
    def getStageCommand(self,i):
        return(repr(self.stages[i]))
    def getStageLogfile(self,i):
        return(self.stages[i].logFile)

    def is_time_to_drain(self):
        return self.shutdown_ev.is_set()
        # FIXME this isn't quite right ... once this is set, clients connecting
        # normally over the next few seconds will crash (note they have no jobs, 
        # so this is sort of OK)

    """Given client information, issue commands to the client (along similar
    lines to getRunnableStageIndex) and update server's internal view of client.
    This is highly stateful, being a resource-tracking wrapper around
    getRunnableStageIndex and hence a glorified Set.pop."""
    def getCommand(self, clientURIstr, clientMemFree, clientProcsFree):
        if self.is_time_to_drain():
            return ("shutdown_abnormally", None)

        # TODO now that getRunnableStageIndex pops from a set,
        # intelligently look for something this client can run
        # (e.g., by passing available resources
        # into getRunnableStageIndex)?
        flag, i = self.getRunnableStageIndex()
        if flag == "run_stage":
            eps = 0.000001
            memOK   = self.getStageMem(i) <= clientMemFree + eps
            procsOK = self.getStageProcs(i) <= clientProcsFree
            if memOK and procsOK:
                return (flag, i)
            else:
                if not memOK:
                    logger.debug("The executor does not have enough free memory (free: %.2fG, required: %.2fG) to run stage %d. (Executor: %s)", clientMemFree, self.getStageMem(i), i, clientURIstr)
                if not procsOK:
                    logger.debug("The executor does not have enough free processors (free: %.1f, required: %.1f) to run stage %d. (Executor: %s)", clientProcsFree, self.getStageProcs(i), i, clientURIstr)
                self.enqueue(i)
                return ("wait", None)
        else:
            return (flag, i)

    """Return a tuple of a command ("shutdown_normally" if all stages are finished,
    "wait" if no stages are currently runnable, or "run_stage" if a stage is
    available) and the next runnable stage if the flag is "run_stage", otherwise
    None"""
    def getRunnableStageIndex(self):
        if self.allStagesCompleted():
            return ("shutdown_normally", None)
        elif len(self.runnable) == 0:
            return ("wait", None)
        else:
            index = self.runnable.pop()
            # remove an instance of currently required memory
            try:
                self.mem_req_for_runnable.remove(self.stages[index].mem)
            except:
                logger.debug("mem_req_for_runnable: %s; mem: %s", self.mem_req_for_runnable, self.stages[index].mem)
                logger.exception("It wasn't here!")
            return ("run_stage", index)

    def allStagesCompleted(self): 
        return len(self.processedStages) == len(self.stages) 

    def addRunningStageToClient(self, clientURI, index):
        try:
            self.clients[clientURI].running_stages.add(index)
        except:
            print("Error: could not find client %s while trying to add running stage" % clientURI)
            raise

    def removeRunningStageFromClient(self, clientURI, index):
        try:
            c = self.clients[clientURI]
        except:
            print("Error: could not find client %s while trying to remove running stage" % clientURI)
            raise
        else:
            try:
                c.running_stages.remove(index)
            except:
                print("Error: unable to remove stage index from registered client: %s" % clientURI)
                logger.exception("Could not remove stage index from running stages list")
                raise

    def setStageStarted(self, index, clientURI):
        URIstring = "(" + str(clientURI) + ")"
        logger.info("Starting Stage " + str(index) + ": " + str(self.stages[index]) + URIstring)
        # There may be a bug in which a stage is added to the runnable set multiple times.
        # It would be better to catch that earlier (by using a different/additional data structure)
        # but for now look for the case when a stage is run twice at the same time, which may
        # produce bizarre results as both processes write files
        if self.stages[index].status == 'running':
            raise Exception('stage %d is already running' % index)
        self.addRunningStageToClient(clientURI, index)
        self.currently_running_stages.add(index)
        self.stages[index].setRunning()

    def checkIfRunnable(self, index):
        """stage added to runnable set if all predecessors finished"""
        canRun = True
        logger.debug("Checking if stage " + str(index) + " is runnable ...")
        if self.stages[index].isFinished():
            canRun = False
        else:
            for i in self.G.predecessors(index):              
                s = self.getStage(i)
                if not s.isFinished():
                    canRun = False
                    break
        logger.debug("Stage " + str(index) + " Runnable: " + str(canRun))
        return canRun

    def setStageFinished(self, index, clientURI, save_state = True, checking_pipeline_status = False):
        """given an index, sets corresponding stage to finished and adds successors to the runnable set"""
        # this function can be called when a pipeline is restarted, and 
        # we go through all stages and set the finished ones to... finished... :-)
        # in that case, we can not remove the stage from the list of running
        # jobs, because there is none.
        if checking_pipeline_status:
            logger.debug("Already finished stage " + str(index))
            self.stages[index].status = "finished"
        else:
            logger.info("Finished Stage " + str(index) + ": " + str(self.stages[index]))
            self.removeFromRunning(index, clientURI, new_status = "finished")
            # run any potential hooks now that the stage has finished:
            for f in self.stages[index].finished_hooks:
                f()
        self.processedStages.append(index)
        # write out the (index, hash) pairs to disk.  We don't actually need the indices
        # for anything (in fact, the restart code in skip_completed_stages is resilient 
        # against an arbitrary renumbering of stages), but a human-readable log is somewhat useful.
        self.finished_stages_fh.write("%d,%s\n" % (index, self.stages[index].getHash()))
        self.finished_stages_fh.flush()
        for i in self.G.successors(index):
            if self.checkIfRunnable(i):
                self.enqueue(i)

    def removeFromRunning(self, index, clientURI, new_status):
        try:
            self.currently_running_stages.discard(index)
        except:
            logger.exception("Unable to remove stage %d from client %s's stages: %s", index, clientURI, self.clients[clientURI].running_stages)
        self.removeRunningStageFromClient(clientURI, index)
        self.stages[index].status = new_status

    def setStageLost(self, index, clientURI):
        """Clean up a stage lost due to unresponsive client"""
        logger.info("Lost Stage %d: %s: ", index, self.stages[index])
        self.removeFromRunning(index, clientURI, new_status = None)
        self.enqueue(index)

    def setStageFailed(self, index, clientURI):
        # given an index, sets stage to failed, adds to failed stages array
        # But... only if this stage has already been retried twice (<- for now static)
        # Once in while retrying a stage makes sense, because of some odd I/O
        # read write issue (NFS race condition?). At least that's what I think is 
        # happening, so trying this to see whether it solves the issue.
        num_retries = self.stages[index].getNumberOfRetries()
        if num_retries < 2:
            # without a sleep statement, the stage will be retried within 
            # a handful of milliseconds, that won't solve anything...
            # this sleep command will block the server for a small amount 
            # of time, but should happen only sporadically
            time.sleep(STAGE_RETRY_INTERVAL)
            self.removeFromRunning(index, clientURI, new_status = None)
            self.stages[index].incrementNumberOfRetries()
            logger.info("RETRYING: ERROR in Stage " + str(index) + ": " + str(self.stages[index]))
            logger.info("RETRYING: adding this stage back to the runnable set.")
            logger.info("RETRYING: Logfile for Stage " + str(self.stages[index].logFile))
            self.enqueue(index)
        else:
            self.removeFromRunning(index, clientURI, new_status = "failed")
            logger.info("ERROR in Stage " + str(index) + ": " + str(self.stages[index]))
            # This is something we should also directly report back to the user:
            print("\nERROR in Stage %s: %s" % (str(index), str(self.stages[index])))
            print("Logfile for (potentially) more information:\n%s\n" % self.stages[index].logFile)
            sys.stdout.flush()
            self.failedStages.append(index)
            for i in nx.dfs_successors(self.G, index).keys():
                self.failedStages.append(i)
                

    def enqueue(self, i):
        """If stage cannot be run due to insufficient mem/procs, executor returns it to the runnable set"""
        logger.debug("Queueing stage %d", i)
        self.runnable.add(i)
        for f in self.stages[i].runnable_hooks:
            f()
        # keep track of the memory requirements of the runnable jobs
        self.mem_req_for_runnable.append(self.stages[i].mem)

    def initialize(self):
        """called once all stages have been added - computes dependencies and adds graph heads to runnable set"""
        self.createEdges()
        self.computeGraphHeads()
        
    """
        Returns True unless all stages are finished, then False
        
        This function also checks to see whether executors can be launched. The
        server keeps track of how many executors are launched/registered. If the
        server set to launch executors itself, it will do so if there are runnable
        stages and the addition of launched/registered executors is smaller than
        the max number of executors it can launch
    """
    def continueLoop(self):
        # We may be have been called one last time just as the parent thread is exiting
        # (if it wakes us with a signal).  In this case, don't do anything:
        if self.shutdown_ev.is_set():
            logger.debug("Shutdown event is set ... quitting")
            return False

        # exit if there are still stages that need to be run, 
        # but when there are no runnable nor any running stages left
        if (len(self.runnable) == 0 and
            len(self.currently_running_stages) == 0
            and len(self.failedStages) > 0):
            logger.info("ERROR: no more runnable stages, however not all stages have finished. Going to shut down.")
            print("\nERROR: no more runnable stages, however not all stages have finished. Going to shut down.\n")
            sys.stdout.flush()
            return False

        # return False if all executors have died but not spawning new ones:
        # 1) there are stages that can be run, and
        # 2) there are no running nor waiting executors, and
        # 3) the number of lost executors has exceeded the number of allowed failed execturs
        if(len(self.runnable) > 0 and 
           (self.number_launched_and_waiting_clients + len(self.clients)) == 0 and 
           self.failed_executors > self.main_options.max_failed_executors):             
            msg = "Error: %d executors have died. This is more than the number of allowed failed executors as set by the flag: --max-failed-executors. Can not spawn new ones. Exiting..." % self.failed_executors
            print(msg)
            logger.warn(msg)
            return False

        # TODO combine with above clause?
        if ((len(self.runnable) > 0) and
          # require no running jobs rather than no clients
          # since in some configurations (e.g., currently SciNet config has
          # the server-local executor shut down only when the server does)
          # the latter choice might lead to the system
          # running indefinitely with no jobs
          (self.number_launched_and_waiting_clients + len(self.clients) == 0 and
          self.executor_memory_required(self.runnable) > self.main_options.mem)):
            msg = "Shutting down due to jobs which require more memory (%.2fG) than available anywhere." % self.executor_memory_required(self.runnable)
            print(msg)
            logger.warn(msg)
            return False

        if self.allStagesCompleted():
            logger.info("All stages complete ... done")
            return False
        else:
            return True

    def updateClientTimestamp(self, clientURI, tick):
        t = time.time() # use server clock for consistency
        try:
            self.clients[clientURI].timestamp = t
            logger.debug("Client %s updated timestamp (tick %d)",
                         clientURI, tick)
        except:
            print("Error: could not find client %s while updating the time stamp" % clientURI)
            logger.exception("clientURI not found in server client list:")
            raise

    # requires: stages != []
    # a better interface might be (self, [stage]) -> { MemAmount : (NumStages, [Stage]) }
    # or maybe the same in a heap (to facilitate getting N stages with most memory)
    def executor_memory_required(self, stages):
        s = max(stages, key=lambda i: self.stages[i].mem)
        return self.stages[s].mem

    # this can't be a loop since we call it via sockets and don't want to block the socket forever
    def manageExecutors(self):
        logger.debug("Looping ...")
        executors_to_launch = self.numberOfExecutorsToLaunch()
        if executors_to_launch > 0:
            memNeeded = self.executor_memory_required(self.runnable)
            if memNeeded <= self.main_options.mem:
                self.launchExecutorsFromServer(executors_to_launch, memNeeded)
            else:
                # we ought to set the stage to failed and run other available stages.
                # However, usually the largest-memory stages run last and in parallel,
                # so there's little benefit to doing so, and it makes sense to finish
                # the running jobs and exit.
                msg = "A stage requires %.2fG of memory to run, but max allowed is %.2fG" \
                        % (memNeeded, self.main_options.mem)
                logger.error(msg)
                print(msg)
        if self.main_options.monitor_heartbeats:
            # look for dead clients and requeue their jobs
            t = time.time()
            # copy() as unregisterClient mutates self.clients during iteration over the latter
            for uri,client in self.clients.copy().iteritems():
                dt = t - client.timestamp
                if dt > pe.HEARTBEAT_INTERVAL + pe.LATENCY_TOLERANCE:
                    logger.warn("Executor at %s has died (no contact for %.1f sec)!", client.clientURI, dt)
                    print("\nWarning: there has been no contact with %s, for %.1f seconds. Considering the executor as dead!\n" % (client.clientURI, dt))
                    if self.failed_executors > self.main_options.max_failed_executors:
                        logger.warn("Currently %d executors have died. This is more than the number of allowed failed executors as set by the flag: --max-failed-executors. Too many executors lost to spawn new ones" % self.failed_executors)

                    self.failed_executors += 1

                    # the unregisterClient function will automatically requeue the
                    # stages that were associated with the lost client
                    self.unregisterClient(client.clientURI)

    """
        Returns an integer indicating the number of executors to launch
        
        This function first verifies whether the server can launch executors
        on its own (self.main_options.nums_exec != 0). Then it checks to
        see whether the executors are able to kill themselves. If they are,
        it's possible that new executors need to be launched. This happens when
        there are runnable stages, but the number of active executors is smaller
        than the number of executors the server is able to launch
    """
    def numberOfExecutorsToLaunch(self):
        if self.failed_executors > self.main_options.max_failed_executors:
            return 0

        if (len(self.runnable) > 0 and
            self.executor_memory_required(self.runnable) > self.main_options.mem):
            # we might still want to launch executors for the stages with smaller
            # requirements
            return 0
        
        if self.main_options.num_exec != 0:
            # Server should launch executors itself
            # This should happen regardless of whether or not executors
            # can kill themselves, because the server is now responsible 
            # for the inital launches as well.
            active_executors = self.number_launched_and_waiting_clients + len(self.clients)
            max_num_executors = self.main_options.num_exec
            executor_launch_room = max_num_executors - active_executors
            # there are runnable stages, and there is room to launch 
            # additional executors
            return min(len(self.runnable), executor_launch_room)
        else:
            return 0
        
    def launchExecutorsFromServer(self, number_to_launch, memNeeded):
        try:
            logger.info("Launching %i executors", number_to_launch)
            for i in range(number_to_launch):
                p = Process(target=launchPipelineExecutor,
                            args=(self.main_options, memNeeded, self.programName))
                p.start()
                self.incrementLaunchedClients()
        except:
            logger.exception("Failed launching executors from the server.")
            raise
        
    def getProcessedStageCount(self):
        return len(self.processedStages)

    def registerClient(self, clientURI, maxmemory):
        # Adds new client (represented by a URI string)
        # to array of registered clients. If the server launched
        # its own clients, we should remove 1 from the number of launched and waiting
        # clients (It's possible though that users launch clients themselves. In that 
        # case we should not decrease this variable)
        self.clients[clientURI] = ExecClient(clientURI, maxmemory)
        if self.number_launched_and_waiting_clients > 0:
            self.number_launched_and_waiting_clients -= 1
        logger.debug("Client registered (banzai): %s", clientURI)
        if self.verbose:
            print("Client registered (banzai!): %s" % clientURI)
            
    def unregisterClient(self, clientURI):
        # removes a client URI string from the table of registered clients. An executor 
        # calls this method when it decides on its own to shut down,
        # and the server may call it when a client is unresponsive
        logger.debug("Client %s calling unregisterClient", clientURI)
        try:
            for s in self.clients[clientURI].running_stages.copy():
                self.setStageLost(s, clientURI)
            del self.clients[clientURI]
        except:
            if self.verbose:
                print("Unable to un-register client: " + clientURI)
            logger.exception("Unable to un-register client: " + clientURI)
        else:
            if self.verbose:
                print("Client un-registered (seppuku!): " + clientURI)
            logger.info("Client un-registered (seppuku!): " + clientURI)

    def incrementLaunchedClients(self):
        self.number_launched_and_waiting_clients += 1

    def skip_completed_stages(self):
        try:
            with open(self.backupFileLocation, 'r') as fh:
                # a stage's index is just an artifact of the graph construction,
                # so load only the hashes of finished stages
                previous_hashes = frozenset((int(e.split(',')[1]) for e in fh.read().split()))
        except:
            logger.info("Finished stages log doesn't exist or is corrupt.")
            return
        # processedStages was read from finished_stages_fh, so:
        self.finished_stages_fh = open(self.backupFileLocation, 'w')
        runnable  = []
        completed = 0
        while True:
            # self.runnable should be populated initially by the graph heads.
            # Traverse the graph by removing stages from the runnable set and,
            # if they don't need to be re-run, adding their dependencies to that set
            # (via setStageFinished); if they do, accumulate them.  Once this set is emptied,
            # we have computed a set of stages which must and can be run
            # (either because the input/output filenames or command has changed
            # or because an ancestor has re-run, i.e., the files themselves have changed)
            # but whose ancestors have already been run
            flag,i = self.getRunnableStageIndex()
            if i == None:
                break

            s = self.getStage(i)

            if not isinstance(s, CmdStage):
                runnable.append(i)
                continue

            # we've never run this command before
            if not s.getHash() in previous_hashes:
                runnable.append(i)
                continue

            self.setStageFinished(i, clientURI = "fake_client_URI", checking_pipeline_status = True)
            completed += 1

        logger.debug("Runnable: %s", runnable)
        for i in runnable:
            self.enqueue(i)

        self.finished_stages_fh.close()
        logger.info('Previously completed stages (of %d total): %d', len(self.stages), completed)

    def printShutdownMessage(self):
        # it is possible that pipeline.continueLoop returns false, even though the
        # pipeline is not completed (for instance, when stages failed, and no more stages
        # can be run) so check that in order to provide the correct feedback to the user
        print("\n\n######################################################")
        print("Shutting down (" + str(len(self.clients)) + " clients remaining) ...")
        if self.allStagesCompleted():
            print("All pipeline stages have been processed.")
            print("Pipeline finished successfully!")
        else:
            print("Not all pipeline stages have been processed,")
            print("however there are no more stages that can be run.")
            print("Pipeline failed...")
        print("######################################################\n")
        logger.debug("Clients still registered at shutdown: " + str(self.clients))
        sys.stdout.flush()

def launchPipelineExecutor(options, memNeeded, programName=None):
    """Launch pipeline executor directly from pipeline"""
    pipelineExecutor = pe.pipelineExecutor(options, memNeeded)
    if options.queue_type == "sge" or options.queue == "sge":
        pipelineExecutor.submitToQueue(programName)
    else:
        pe.launchExecutor(pipelineExecutor)
        
def launchServer(pipeline, options):
    # first follow up on the previously reported total number of 
    # stages in the pipeline with how many have already finished:
    pipeline.printNumberProcessedStages()
    
    # is the server going to be verbose or not?
    if options.verbose:
        def verboseprint(*args):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
                print(arg,)
            print()
    else:
        verboseprint = lambda *a: None
    
    # getIpAddress is similar to socket.gethostbyname(...)
    # but uses a hack to attempt to avoid returning localhost (127....)
    network_address = Pyro4.socketutil.getIpAddress(socket.gethostname(),
                                                    workaround127 = True, ipVersion = 4)
    daemon = Pyro4.core.Daemon(host=network_address)
    pipelineURI = daemon.register(pipeline)
    
    if options.use_ns:
        # in the future we might want to launch a nameserver here
        # instead of relying on a separate executable running
        ns = Pyro4.locateNS()
        ns.register("pipeline", pipelineURI)
    else:
        # If not using Pyro NameServer, must write uri to file for reading by client.
        uf = open(options.urifile, 'w')
        uf.write(pipelineURI.asString())
        uf.close()
    
    pipeline.setVerbosity(options.verbose)

    try:
        # start Pyro server
        t = Process(target=daemon.requestLoop)
        # t.daemon = True # this isn't allowed
        t.start()

        # at this point requests made to the Pyro daemon will touch process `t`'s copy
        # of the pipeline, so modifiying `pipeline` won't have any effect.  The exception is
        # communication through its multiprocessing.Event, which we use below to wait
        # for termination.

        verboseprint("Daemon is running at: %s" % daemon.locationStr)
        logger.info("Daemon is running at: %s", daemon.locationStr)
        verboseprint("The pipeline's uri is: %s" % str(pipelineURI))
        logger.info("The pipeline's uri is: %s", str(pipelineURI))

        # handle SIGTERM (sent by SciNet 15-30s before hard kill) by setting
        # the shutdown event (we shouldn't actually see a SIGTERM on PBS
        # since PBS submission logic gives us a lifetime related to our walltime
        # request ...)
        def handler(sig, _stack):
            pipeline.shutdown_ev.set()
        signal.signal(signal.SIGTERM, handler)

        # spawn a loop to manage executors in a separate process
        # (here we use a proxy to make calls to manageExecutors because (a) 
        # processes don't share memory, (b) so that its logic is
        # not interleaved with calls from executors.  We could instead use a `select`
        # for both Pyro and non-Pyro socket events; see the Pyro documentation)
        p = Pyro4.Proxy(pipelineURI)
        def loop():
            try:
                logger.debug("Auxiliary loop started")
                while p.continueLoop():
                    p.manageExecutors()
                    pipeline.shutdown_ev.wait(LOOP_INTERVAL)
            except:
                logger.exception("Server loop encountered a problem.  Shutting down.")
            finally:
                logger.info("Server loop going to shut down ...")
                p.set_shutdown_ev()

        h = Process(target=loop)
        h.daemon = True
        h.start()

        try:
            jid    = os.environ["PBS_JOBID"]
            output = check_output(['qstat', '-f', jid])

            time_left = int(re.search('Walltime.Remaining = (\d*)', output).group(1))
            logger.debug("Time remaining: %d s" % time_left)
            time_to_live = time_left - pe.SHUTDOWN_TIME
        except:
            logger.info("I couldn't determine your remaining walltime from qstat.")
            time_to_live = None
        flag = pipeline.shutdown_ev.wait(time_to_live)
        if not flag:
            logger.info("Time's up!")
        pipeline.shutdown_ev.set()

    # FIXME if we terminate abnormally, we should _actually_ kill child executors (if running locally)
    except KeyboardInterrupt:
        logger.exception("Caught keyboard interrupt, killing executors and shutting down server.")
        print("\nKeyboardInterrupt caught: cleaning up, shutting down executors.\n")
        sys.stdout.flush()
    except:
        logger.exception("Exception running server in daemon.requestLoop. Server shutting down.")
        print("%s" % sys.exc_info())
    else:
        # allow time for all clients to contact the server and be told to shut down
        # (we could instead add a way for the server to notify its registered clients):
        # otherwise they will crash when they try to contact the (shutdown) server.
        # It's not important that clients shut down properly (if they see a server crash, they
        # will cancel their running jobs, but they're done by the time the server exits)
        # TODO this only makes sense if we are actually shutting down nicely,
        # and not because we're out of walltime, in which case this doesn't help
        # (client jobs will die anyway)
        #print("Sleeping %d s to allow time for clients to shutdown..." % pe.SHUTDOWN_TIME)
        #time.sleep(pe.SHUTDOWN_TIME)
        # trying to access variables from `p` in the `finally` clause (in order
        # to print a shutdown message) hangs for some reason, so do it here instead
        p.printShutdownMessage()
    finally:
        # brutal, but awkward to do with our system of `Event`s
        # could send a signal to `t` instead:
        t.terminate()

def flatten_pipeline(p):
    """return a list of tuples for each stage.
       Each item in the list is (id, command, [dependencies]) 
       where dependencies is a list of stages depend on this stage to be complete before they run.
    """
    def post(x, y):
        if y[0] in x[2]: 
            return 1
        elif x[0] in y[2]:
            return -1
        else:
            return 0 
                
    return sorted([(i, str(p.stages[i]), p.G.predecessors(i)) for i in p.G.nodes_iter()],cmp=post)

def pipelineDaemon(pipeline, options, programName=None):
    """Launches Pyro server and (if specified by options) pipeline executors"""

    if options.urifile is None:
        options.urifile = os.path.abspath(os.path.join(os.curdir, options.pipeline_name + "_uri"))

    if options.restart:
        logger.debug("Examining filesystem to determine skippable stages...")
        pipeline.skip_completed_stages()

    #check for valid pipeline 
    if len(pipeline.runnable) == 0:
        print("Pipeline has no runnable stages. Exiting...")
        sys.exit()
   
    logger.debug("Prior to starting server, total stages %i. Number processed: %i.", 
                 len(pipeline.stages), len(pipeline.processedStages))
    logger.debug("Number of stages in runnable queue: %i",
                 len(pipeline.runnable))
    
    pipeline.programName = programName
    try:
        # we are now appending to the stages file since we've already written
        # previously completed stages to it in skip_completed_stages
        with open(pipeline.backupFileLocation, 'a') as fh:
            pipeline.finished_stages_fh = fh
            logger.debug("Starting server...")
            launchServer(pipeline, options)
    except:
        logger.exception("Exception (=> quitting): ")
    finally:
        sys.exit(0)
