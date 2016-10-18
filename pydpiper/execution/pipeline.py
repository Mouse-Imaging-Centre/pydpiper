#!/usr/bin/env python3

import hashlib
import networkx as nx  # type: ignore
import os
import sys
import signal
import socket
import time
import re
import resource
from collections import defaultdict
from datetime import datetime
import subprocess
from shlex import split
from multiprocessing import Process, Event  # type: ignore
from configargparse import Namespace
import logging
import functools
from typing import Any

try:
    from sys import intern
except:
    "older Python doesn't namespace `intern`, so do nothing"

# TODO move this and Pyro4 imports down into launchServer where pipeline name is available?
#os.environ["PYRO_LOGLEVEL"] = os.getenv("PYRO_LOGLEVEL", "INFO")
#os.environ["PYRO_LOGFILE"]  = os.path.splitext(os.path.basename(__file__))[0] + ".log"
# TODO name the server logfile more descriptively

logger = logging # type: Any
#logger = logging.getLogger(__name__)
logger.basicConfig(filename="pipeline.log", level=os.getenv("PYDPIPER_LOGLEVEL", "INFO"),
                   datefmt="%Y-%m-%d %H:%M:%S",
                   format="[%(asctime)s.%(msecs)03d,"
                          +__name__+",%(levelname)s] %(message)s")  # type: ignore
                
import Pyro4  # type: ignore
from . import pipeline_executor as pe

Pyro4.config.SERVERTYPE = pe.Pyro4.config.SERVERTYPE

LOOP_INTERVAL = 5
STAGE_RETRY_INTERVAL = 1
SUBDEBUG = 5

sys.excepthook = Pyro4.util.excepthook # type: ignore

class PipelineFile(object):
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
class ExecClient(object):
    def __init__(self, client, maxmemory):
        self.clientURI = client
        self.maxmemory = maxmemory
        self.running_stages = set([])
        self.timestamp = time.time()

def memoize_hook(hook):  # TODO replace with functools.lru_cache (?!) in python3
    data = Namespace(called=False, result=None)  # because of Python's bizarre assignment rules
    def g():
        if data.called:
            return data.result
        else:
            data.result = hook()
            data.called = True
            return data.result
    return g

class PipelineStage(object):
    def __init__(self):
        self.mem = None # if not set, use pipeline default
        self.procs = 1 # default number of processors per stage
        # the input files for this stage
        self.inputFiles  = [] # type: List[str]
        # the output files for this stage
        self.outputFiles = [] # type: List[str]
        self.logFile = None
        self.status = None
        self.name = ""
        self.colour = "black" # used when a graph is created of all stages to colour the nodes
        self.number_retries = 0
        # functions to be called when the stage becomes runnable
        # (these might be called multiple times, so should be benign
        # in some sense)
        self._runnable_hooks = []
        # functions to be called when a stage finishes
        self.finished_hooks = []

    def add_runnable_hook(self, h, memoize=True):
        self._runnable_hooks.append((memoize_hook if memoize else lambda x: x)(h))

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
    pipeline_start_time = datetime.isoformat(datetime.now())
    logfile_id = 0
    def __init__(self, argArray):
        PipelineStage.__init__(self)
        self.cmd = [] # the input array converted to strings
        self.parseArgs(argArray)
        #self.checkLogFile()
    def parseArgs(self, argArray):
        if argArray:
            for a in argArray:
                ft = getattr(a, "fileType", None)
                s = intern(str(a))
                if ft == "input":
                    self.inputFiles.append(s)
                elif ft == "output":
                    self.outputFiles.append(s)
                self.cmd.append(s)
            self.name = self.cmd[0]
    def checkLogFile(self):  # TODO silly, this is always called since called by __init__
        if not self.logFile:
            self.logFile = self.name + "." + CmdStage.pipeline_start_time + '-' + str(CmdStage.logfile_id) + ".log"
            CmdStage.logfile_id += 1
    def setLogFile(self, logFileName): 
        self.logFile = str(logFileName)
    def execStage(self):
        of = open(self.logFile, 'a')
        of.write("Running on: " + socket.gethostname() + " at " + datetime.isoformat(datetime.now(), " ") + "\n")
        of.write(repr(self) + "\n")
        of.flush()

        args = split(repr(self))
        returncode = subprocess.call(args, stdout=of, stderr=of, shell=False)
        of.close()
        return(returncode)

    def getHash(self):
        """Return a small value which can be used to compare objects.
        Use a deterministic hash to allow persistence across restarts (calls to `hash` and `__hash__`
        depend on the value of PYTHONHASHSEED as of Python 3.3)"""
        return hashlib.md5("".join(self.cmd).encode()).hexdigest()

    def __repr__(self):
        return(" ".join(self.cmd))
    def __eq__(self, other):
        return (self is other) or (self.__class__ == other.__class__ and self.cmd == other.cmd)
    def __hash__(self):
        return tuple(self.cmd).__hash__()

"""A graph with no edge information; see networkx/classes/digraph.py"""
class ThinGraph(nx.DiGraph):
    all_edge_dict = {'weight': 1}
    def single_edge_dict(self):
        return self.all_edge_dict
    edge_attr_dict_factory = single_edge_dict


"""A graph with no edge information; see networkx/classes/digraph.py"""
class ThinGraph(nx.DiGraph):
    all_edge_dict = {'weight': 1}
    def single_edge_dict(self):
        return self.all_edge_dict
    edge_attr_dict_factory = single_edge_dict
    
class Pipeline(object):
    # TODO the way we initialize a pipeline is currently a bit gross, e.g.,
    # setting a bunch of instance variables after __init__ - the presence of a method
    # called `initialize` should be a hint that all is perhaps not well, but perhaps
    # there is indeed some information legitimately unavailable when we first construct
    def __init__(self, stages, options):
        # the core pipeline is stored in a directed graph. The graph is made
        # up of integer indices
        # main set of options, needed since (a) we don't bother unpacking
        # every single option into `self`, (b) since (most of) the option set
        # is needed to launch new executors at run time
        self.pipeline_name = options.application.pipeline_name
        self.options = options
        self.exec_options = options.execution
        self.G = ThinGraph()
        # a map from indices to the number of unfulfilled prerequisites
        # of the corresponding graph node (will be populated later -- __init__ is a misnomer)
        self.unfinished_pred_counts = []
        # an array of the actual stages (PipelineStage objects)
        self.stages = []
        self.nameArray = []
        # indices of the stages ready to be run
        self.runnable = set()
        # an array to keep track of stage memory requirements
        self.mem_req_for_runnable = []
        # a hideous hack; the idea is that after constructing the underlying graph,
        # a pipeline running executors locally will measure its own maxRSS (once)
        # and subtract this from the amount of memory claimed available for use on the node.
        self.memAvail = None
        self.currently_running_stages = set()
        # the current stage counter
        self.counter = 0
        # hash to keep the output to stage association
        self.outputhash = {}
        # a hash per stage - computed from inputs and outputs or whole command
        self.stage_dict = {}
        self.num_finished_stages = 0
        self.failedStages = []
        # location of backup files for restart if needed
        self.backupFileLocation = self._backup_file_location()
        # table of registered clients (using ExecClient class instances) indexed by URI
        self.clients = {}
        # number of clients (executors) that have been launched by the server
        # we need to keep track of this because even though no (or few) clients
        # are actually registered, a whole bunch of them could be waiting in the
        # queue
        self.number_launched_and_waiting_clients = 0
        # clients we've lost contact with due to crash, etc.
        self.failed_executors = 0
        # time to shut down, due to walltime or having completed all stages?
        # (use an event rather than a simple flag for shutdown notification
        # so that we can shut down even if a process is currently sleeping)
        self.shutdown_ev = None # was Event(), but this is too slow
        self.programName = None
        self.skipped_stages = 0
        self.verbose = 0
        # Handle to write out processed stages to
        self.finished_stages_fh = None
        
        self.outputDir = self.options.application.output_directory or os.getcwd()

        if self.options.execution.submit_server and self.options.execution.local:
            # redirect the standard output to a text file
            serverLogFile = os.path.join(self.outputDir, self.pipeline_name + '_server_stdout.log')
            LINE_BUFFERING = 1
            sys.stdout = open(serverLogFile, 'a', LINE_BUFFERING)

        for s in stages:
            self._add_stage(s)

        self.createEdges()
        # could also set this on G itself ...
        self.unfinished_pred_counts = [ len([i for i in self.G.predecessors(n)
                                             if not self.stages[i].isFinished()])
                                        for n in range(self.G.order()) ]
        graph_heads = [n for n in self.G.nodes_iter()
                       if self.unfinished_pred_counts[n] == 0]
        logger.info("Graph heads: " + str(graph_heads))
        for n in graph_heads:
            self.enqueue(n)
       
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
        return self.num_finished_stages

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
        return [c.maxmemory for _, c in self.clients.items()]

    def _add_stage(self, stage):
        """adds a stage to the pipeline"""
        # check if stage already exists in pipeline - if so, don't bother

        # check if stage exists - stage uniqueness defined by in- and output
        # for base stages and entire command for CmdStages
        # FIXME this logic is rather redundant and can be simplified
        # (assuming that the set of stages the pipeline is given has
        # the same equality relation as is used here)
        # FIXME as getHash is called several times per stage, cache the hash in the stage.
        # To save memory, we could make use of the fact that strings
        # are actually interned, so it might be faster/cheaper
        # just to store/compare the pointers.
        h = stage.getHash()
        if h in self.stage_dict:
            self.skipped_stages += 1
            #stage already exists - nothing to be done
        else: #stage doesn't exist - add it
            self.stage_dict[h] = self.counter
            #self.statusArray[self.counter] = 'notstarted'
            self.stages.append(stage)
            self.nameArray.append(stage.name)
            # add all outputs to the output dictionary
            for o in stage.outputFiles:
                self.outputhash[o] = self.counter
            # add the stage's index to the graph
            self.G.add_node(self.counter, label=stage.name, color=stage.colour)
            self.counter += 1
        # huge hack since default isn't available in CmdStage() constructor
        # (may get overridden later by a hook, hence may really be wrong ... ugh):
        if stage.mem is None and self.exec_options is not None:
            stage.setMem(self.exec_options.default_job_mem)

    def _backup_file_location(self, outputDir=None):
        loc = os.path.join(outputDir or os.getcwd(),
                           self.pipeline_name + '_finished_stages')
        return loc

    def printStages(self, name):
        """Prints stages to a file, stage info to stdout"""

        fileForPrinting = os.path.abspath(os.curdir + "/" + str(name) + "_pipeline_stages.txt")
        pf = open(fileForPrinting, "w")
        for i in range(len(self.stages)):
            pf.write(str(i) + "  " + str(self.stages[i]) + "\n")
        pf.close()
        print("Total number of stages in the pipeline: ", len(self.stages))
                 
    def printNumberProcessedStages(self):
        print("Number of stages already processed:     ", self.num_finished_stages)
                  
    def createEdges(self):
        """computes stage dependencies by examining their inputs/outputs"""
        starttime = time.time()
        # iterate over all nodes
        for i in self.G.nodes_iter():
            for ip in self.stages[i].inputFiles:
                # if the input to the current stage was the output of another
                # stage, add a directional dependence to the DiGraph
                if ip in self.outputhash:
                    self.G.add_edge(self.outputhash[ip], i)
        endtime = time.time()
        logger.info("Create Edges time: " + str(endtime-starttime))

    def get_stage_info(self, i):
        s = self.stages[i]
        return pe.StageInfo(mem=s.mem, procs=s.procs, ix=i, cmd=s.cmd, log_file=s.logFile)

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
        if clientMemFree == 0:
            logger.debug("Executor has no free memory")
            return ("wait", None)
        if clientProcsFree == 0:
            logger.debug("Executor has no free processors")
            return ("wait", None)

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
        return self.num_finished_stages == len(self.stages) 

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

    #@Pyro4.oneway
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
        logger.log(SUBDEBUG, "Checking if stage %s is runnable ...", str(index))
        canRun = ((not self.stages[index].isFinished()) and (self.unfinished_pred_counts[index] == 0))
        logger.log(SUBDEBUG, "Stage %s Runnable: %s", str(index), str(canRun))
        return canRun

    def setStageFinished(self, index, clientURI, save_state = True,
                         checking_pipeline_status = False):
        """given an index, sets corresponding stage to finished and adds successors to the runnable set"""

        s = self.stages[index]
        
        # since we want to use refcounting (where a 'reference' is an
        # unsatisfied dependency of a stage and 'collection' is
        # adding to the set of runnable stages) to determine whether a stage's
        # prerequisites have run, we make it an error for the same stage
        # to finish more than once (alternately, we could merely avoid
        # decrementing counts of previously finished stages, but
        # this choice should expose bugs sooner)
        if s.isFinished():
            raise ValueError("Already finished stage %d" % index)
        
        # this function can be called when a pipeline is restarted, and 
        # we go through all stages and set the finished ones to... finished... :-)
        # in that case, we can not remove the stage from the list of running
        # jobs, because there is none.

        if checking_pipeline_status:
            logger.log(SUBDEBUG, "Already finished stage " + str(index))
            s.status = "finished"
        else:
            logger.info("Finished Stage %s: %s (on %s)", str(index), str(self.stages[index]), clientURI)
            self.removeFromRunning(index, clientURI, new_status = "finished")
            # run any potential hooks now that the stage has finished:
            for f in s.finished_hooks:
                f(s)
        self.num_finished_stages += 1
        # write out the (index, hash) pairs to disk.  We don't actually need the indices
        # for anything (in fact, the restart code in skip_completed_stages is resilient 
        # against an arbitrary renumbering of stages), but a human-readable log is somewhat useful.
        if not checking_pipeline_status:
            self.finished_stages_fh.write("%d,%s\n" % (index, self.stages[index].getHash()))
            self.finished_stages_fh.flush()
        # FIXME flush could turned off as an optimization (more sensibly, a small buffer size could be set)
        # ... though we might not record a stage's completion, this doesn't affect correctness.
        for i in self.G.successors(index):
            self.unfinished_pred_counts[i] -= 1
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
        logger.warning("Lost Stage %d: %s: ", index, self.stages[index])
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
        """Update pipeline data structures and run relevant hooks when a stage becomes runnable."""
        logger.log(SUBDEBUG, "Queueing stage %d", i)
        self.runnable.add(i)
        for f in self.stages[i]._runnable_hooks:
            f(self.stages[i])
        # the easiest place to ensure that all stages request at least
        # the default job mem is here. The hooks above might estimate
        # memory for the jobs, here we'll override that if they requested
        # less than the minimum
        if self.stages[i].mem < self.exec_options.default_job_mem:
            self.stages[i].setMem(self.exec_options.default_job_mem)
        # scale everything by the memory_factor
        self.stages[i].setMem(self.stages[i].mem * self.exec_options.memory_factor)
        # keep track of the memory requirements of the runnable jobs
        self.mem_req_for_runnable.append(self.stages[i].mem)

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

        elif self.allStagesCompleted():
            logger.info("All stages complete ... done")
            return False

        # exit if there are still stages that need to be run, 
        # but when there are no runnable nor any running stages left
        # (e.g., if some stages have repeatedly failed)
        # TODO this might indicate a bug, so better reporting would be useful
        elif (len(self.runnable) == 0
            and len(self.currently_running_stages) == 0):
            logger.info("ERROR: no more runnable stages, however not all stages have finished. Going to shut down.")
            print("\nERROR: no more runnable stages, however not all stages have finished. Going to shut down.\n")
            sys.stdout.flush()
            return False

        # return False if all executors have died but not spawning new ones:
        # 1) there are stages that can be run, and
        # 2) there are no running nor waiting executors, and
        # 3) the number of lost executors has exceeded the number of allowed failed execturs
        elif (len(self.runnable) > 0 and 
              (self.number_launched_and_waiting_clients + len(self.clients)) == 0 and 
              self.failed_executors > self.exec_options.max_failed_executors):
            msg = ("Error: %d executors have died. This is more than the number"
                   "(%d) allowed by --max-failed-executors.  No executors remain; exiting..."
                   % (self.failed_executors, self.exec_options.max_failed_executors))
            print(msg)
            logger.warn(msg)
            return False

        # TODO combine with above clause?
        elif ((len(self.runnable) > 0) and
          # require no running jobs rather than no clients
          # since in some configurations (e.g., currently SciNet config has
          # the server-local executor shut down only when the server does)
          # the latter choice might lead to the system
          # running indefinitely with no jobs
          (self.number_launched_and_waiting_clients + len(self.clients) == 0 and
          self.max_memory_required(self.runnable) > self.memAvail)):
            msg = "Shutting down due to jobs which require more memory (%.2fG) than available anywhere. Please use the --mem argument to increase the amount of memory that executors can request." % self.max_memory_required(self.runnable)
            print(msg)
            logger.warning(msg)
            return False
        else:
            return True

    #@Pyro4.oneway
    def updateClientTimestamp(self, clientURI, tick):
        logger.debug("Time... (%s)", clientURI)
        t = time.time()  # use server clock for consistency
        logger.debug("Got it... (%s)", clientURI)
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
    def max_memory_required(self, stages):
        s = max(stages, key=lambda i: self.stages[i].mem)
        return self.stages[s].mem

    # this can't be a loop since we call it via sockets and don't want to block the socket forever
    def manageExecutors(self):
        logger.debug("Checking if executors need to be launched ...")
        executors_to_launch = self.numberOfExecutorsToLaunch()
        if executors_to_launch > 0:
            # RAM needed to run a single job:
            memNeeded = self.max_memory_required(self.runnable)
            # RAM needed to run `proc` most expensive jobs (not the ideal choice):
            memWanted = sum(sorted([self.stages[i].mem for i in self.runnable],
                                   key = lambda x: -x)[0:self.exec_options.proc])
            logger.debug("wanted: %s", memWanted)
            logger.debug("needed: %s", memNeeded)
                
            if memNeeded > self.memAvail:
                msg = "A stage requires %.2fG of memory to run, but max allowed is %.2fG" \
                        % (memNeeded, self.memAvail)
                logger.error(msg)
                print(msg)
            else:
                if self.exec_options.greedy:
                    mem = self.memAvail
                elif memWanted <= self.memAvail:
                    mem = memWanted
                else:
                    mem = self.memAvail  #memNeeded?
                try:
                    self.launchExecutorsFromServer(executors_to_launch, mem)
                except pe.SubmitError:
                    logger.exception("Failed to submit executors; will retry")

        logger.debug("... checking for crashed executors ...")
        if self.exec_options.monitor_heartbeats:
            # look for dead clients and requeue their jobs
            t = time.time()
            # copy() as unregisterClient mutates self.clients during iteration over the latter
            for uri,client in self.clients.copy().items():
                dt = t - client.timestamp
                if dt > pe.HEARTBEAT_INTERVAL + self.exec_options.latency_tolerance:
                    logger.warning("Executor at %s has died (no contact for %.1f sec)!", client.clientURI, dt)
                    print("\nWarning: there has been no contact with %s, for %.1f seconds. Considering the executor as dead!\n" % (client.clientURI, dt))
                    if self.failed_executors > self.exec_options.max_failed_executors:
                        logger.warning("Currently %d executors have died. This is more than the number of allowed failed executors as set by the flag: --max-failed-executors. Too many executors lost to spawn new ones" % self.failed_executors)

                    self.failed_executors += 1

                    # the unregisterClient function will automatically requeue the
                    # stages that were associated with the lost client
                    self.unregisterClient(client.clientURI)
        logger.debug("...done.")

    """
        Returns an integer indicating the number of executors to launch
        
        This function first verifies whether the server can launch executors
        on its own (self.exec_options.nums_exec != 0). Then it checks to
        on its own (self.options.nums_exec != 0). Then it checks to
        see whether the executors are able to kill themselves. If they are,
        it's possible that new executors need to be launched. This happens when
        there are runnable stages, but the number of active executors is smaller
        than the number of executors the server is able to launch
    """
    def numberOfExecutorsToLaunch(self):
        if self.failed_executors > self.exec_options.max_failed_executors:
            return 0

        if (len(self.runnable) > 0 and
            self.max_memory_required(self.runnable) > self.memAvail):
            # we might still want to launch executors for the stages with smaller
            # requirements
            return 0
        
        if self.exec_options.num_exec != 0:
            # Server should launch executors itself
            # This should happen regardless of whether or not executors
            # can kill themselves, because the server is now responsible 
            # for the initial launches as well.
            active_executors = self.number_launched_and_waiting_clients + len(self.clients)
            desired_num_executors = min(len(self.runnable), self.exec_options.num_exec)
            executor_launch_room = desired_num_executors - active_executors
            # there are runnable stages, and there is room to launch 
            # additional executors
            return max(executor_launch_room, 0)
        else:
            return 0
        
    def launchExecutorsFromServer(self, number_to_launch, memNeeded):
        logger.info("Launching %i executors", number_to_launch)
        try:
            launchPipelineExecutors(options=self.options, number=number_to_launch,
                                    mem_needed=memNeeded, uri_file=self.exec_options.urifile)
            self.number_launched_and_waiting_clients += number_to_launch
        except:
            logger.exception("Failed launching executors from the server.")
            raise
        
    def getProcessedStageCount(self):
        return self.num_finished_stages

    def registerClient(self, clientURI, maxmemory):
        # Adds new client (represented by a URI string)
        # to array of registered clients. If the server launched
        # its own clients, we should remove 1 from the number of launched and waiting
        # clients (It's possible though that users launch clients themselves. In that 
        # case we should not decrease this variable)
        # FIXME this is a completely broken way to decide whether to decrement ...
        self.clients[clientURI] = ExecClient(clientURI, maxmemory)
        if self.number_launched_and_waiting_clients > 0:
            self.number_launched_and_waiting_clients -= 1
        logger.debug("Client registered (Eh!): %s", clientURI)
        if self.verbose:
            print("Client registered (Eh!): %s" % clientURI)
            
    #@Pyro4.oneway
    def unregisterClient(self, clientURI):
        # removes a client URI string from the table of registered clients. An executor 
        # calls this method when it decides on its own to shut down,
        # and the server may call it when a client is unresponsive
        logger.debug("unregisterClient: un-registering %s", clientURI)
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
                print("Client un-registered (Cheers!): " + clientURI)
            logger.info("Client un-registered (Cheers!): " + clientURI)

    def incrementLaunchedClients(self):
        self.number_launched_and_waiting_clients += 1

    def skip_completed_stages(self):
        logger.debug("Consulting logs to determine skippable stages...")
        try:
            with open(self.backupFileLocation, 'r') as fh:
                # a stage's index is just an artifact of the graph construction,
                # so load only the hashes of finished stages
                previous_hashes = frozenset((e.split(',')[1] for e in fh.read().split()))
        except:
            logger.info("Finished stages log doesn't exist or is corrupt.")
            return

        runnable  = []
        finished  = []
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

            h = s.getHash()

            # we've never run this command before
            if not h in previous_hashes:
                runnable.append(i)
                continue

            self.setStageFinished(i, clientURI = "fake_client_URI", checking_pipeline_status = True)

            finished.append((i, h))  # stupid ... duplicates logic in setStageFinished ...
            completed += 1

        logger.debug("Runnable: %s", runnable)
        for i in runnable:
            self.enqueue(i)
        with open(self.backupFileLocation, 'w') as fh:
            # TODO For further optimization, it might (?) be even faster to write to memory and then
            # make a single file write when finished.
            # FIXME should write to tmp file in same dir, then overwrite (?) since a otherwise an interruption
            # in writing this file will cause progress to be lost
            for l in finished:
                fh.write("%d,%s\n" % l)
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

def launchPipelineExecutors(options, mem_needed, number, uri_file):
    """Launch pipeline executors directly from pipeline"""
    if options.execution.local or not options.execution.queue_type:
        for _ in range(number):
            e = pe.pipelineExecutor(options=options.execution,
                                    uri_file=uri_file,
                                    #pipeline_name=options.application.pipeline_name,
                                    memNeeded=mem_needed)
            pe.launchExecutor(e)
    else:
        pipelineExecutor = pe.pipelineExecutor(options=options.execution,
                                               uri_file=uri_file,
                                               #pipeline_name=options.application.pipeline_name,
                                               memNeeded=mem_needed)
        pipelineExecutor.submitToQueue(number=number)


def launchServer(pipeline):
    options = pipeline.options
    pipeline.printStages(options.application.pipeline_name)
    pipeline.printNumberProcessedStages()

    # expensive, so only create for pipelines that will actually run
    pipeline.shutdown_ev = Event()

    # for ideological reasons this should live in a method, but pipeline init is
    # rather baroque anyway, and arguably launchServer/pipelineDaemon ought to be
    # a single method with cleaned-up initialization
    executors_local = pipeline.exec_options.local or (pipeline.exec_options.queue_type is None)

    if executors_local:
        # measured once -- we assume that server memory usage will be
        # roughly constant at this point
        pipeline.memAvail = pipeline.exec_options.mem - (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 10**6)  # 2^20?
    else:
        pipeline.memAvail = pipeline.exec_options.mem
    
    # is the server going to be verbose or not?
    if options.application.verbose:
        verboseprint = print
    else:
        verboseprint = lambda *args: None
    
    # getIpAddress is similar to socket.gethostbyname(...)
    # but uses a hack to attempt to avoid returning localhost (127....)
    network_address = Pyro4.socketutil.getIpAddress(socket.gethostname(),
                                                    workaround127 = True, ipVersion = 4)
    daemon = Pyro4.core.Daemon(host=network_address)
    pipelineURI = daemon.register(pipeline)
    
    if options.execution.use_ns:
        # in the future we might want to launch a nameserver here
        # instead of relying on a separate executable running
        ns = Pyro4.locateNS()
        ns.register("pipeline", pipelineURI)
    else:
        # If not using Pyro NameServer, must write uri to file for reading by client.
        uf = open(options.execution.urifile, 'w')
        uf.write(pipelineURI.asString())
        uf.close()
    
    pipeline.setVerbosity(options.application.verbose)

    shutdown_time = pe.EXECUTOR_MAIN_LOOP_INTERVAL + options.execution.latency_tolerance
    
    try:
        t = Process(target=daemon.requestLoop)
        # t.daemon = True
        t.start()

        # at this point requests made to the Pyro daemon will touch process `t`'s copy
        # of the pipeline, so modifiying `pipeline` won't have any effect.  The exception is
        # communication through its multiprocessing.Event, which we use below to wait
        # for termination.
        #FIXME does this leak the memory used by the old pipeline?
        #if so, avoid doing this or at least `del` the old graph ...

        verboseprint("Daemon is running at: %s" % daemon.locationStr)
        logger.info("Daemon is running at: %s", daemon.locationStr)
        verboseprint("The pipeline's uri is: %s" % str(pipelineURI))
        logger.info("The pipeline's uri is: %s", str(pipelineURI))

        e = pipeline.shutdown_ev

        # handle SIGTERM (sent by SciNet 15-30s before hard kill) by setting
        # the shutdown event (we shouldn't actually see a SIGTERM on PBS
        # since PBS submission logic gives us a lifetime related to our walltime
        # request ...)
        def handler(sig, _stack):
            e.set()
        signal.signal(signal.SIGTERM, handler)

        # spawn a loop to manage executors in a separate process
        # (here we use a proxy to make calls to manageExecutors because (a)
        # processes don't share memory, (b) so that its logic is
        # not interleaved with calls from executors.  We could instead use a `select`
        # for both Pyro and non-Pyro socket events; see the Pyro documentation)
        p = Pyro4.Proxy(pipelineURI)

        #mem, memAvail = pipeline.options.execution.mem, pipeline.memAvail
        def loop():
            try:
                logger.debug("Auxiliary loop started")
                logger.debug("memory limit: %.3G; available after server overhead: %.3fG" % (options.execution.mem, pipeline.memAvail))
                while p.continueLoop():
                    p.manageExecutors()
                    e.wait(LOOP_INTERVAL)
            except:
                logger.exception("Server loop encountered a problem.  Shutting down.")
            finally:
                logger.info("Server loop going to shut down ...")
                p.set_shutdown_ev()

        h = Process(target=loop)
        h.daemon = True
        h.start()
        #del pipeline   # `top` shows this has no effect on vmem

        try:
            jid    = os.environ["PBS_JOBID"]
            output = subprocess.check_output(['qstat', '-f', jid], stderr=subprocess.STDOUT)

            time_left = int(re.search('Walltime.Remaining = (\d*)', output).group(1))
            logger.debug("Time remaining: %d s" % time_left)
            time_to_live = time_left - shutdown_time
        except:
            logger.info("I couldn't determine your remaining walltime from qstat.")
            time_to_live = None
        flag = e.wait(time_to_live)
        if not flag:
            logger.info("Time's up!")
        e.set()

    # FIXME if we terminate abnormally, we should _actually_ kill child executors (if running locally)
    except KeyboardInterrupt:
        logger.exception("Caught keyboard interrupt, killing executors and shutting down server.")
        print("\nKeyboardInterrupt caught: cleaning up, shutting down executors.\n")
        sys.stdout.flush()
    except:
        logger.exception("Exception running server in daemon.requestLoop. Server shutting down.")
        raise
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
                
    return sorted([(i, str(p.stages[i]), p.G.predecessors(i)) for i in p.G.nodes_iter()], key=functools.cmp_to_key(post))

def pipelineDaemon(pipeline, options, programName=None):
    """Launches Pyro server and (if specified by options) pipeline executors"""

    if options.execution.urifile is None:
        options.execution.urifile = os.path.abspath(os.path.join(os.curdir, options.application.pipeline_name + "_uri"))

    if options.application.restart:
        pipeline.skip_completed_stages()

    if len(pipeline.runnable) == 0:
        print("Pipeline has no runnable stages. Exiting...")
        sys.exit()
   
    logger.debug("Prior to starting server, total stages %i. Number processed: %i.", 
                 len(pipeline.stages), pipeline.num_finished_stages)
    logger.debug("Number of runnable stages: %i",
                 len(pipeline.runnable))
    
    pipeline.programName = programName
    try:
        # we are now appending to the stages file since we've already written
        # previously completed stages to it in skip_completed_stages
        with open(pipeline.backupFileLocation, 'a') as fh:
            pipeline.finished_stages_fh = fh
            logger.debug("Starting server...")
            launchServer(pipeline)
    except:
        logger.exception("Exception (=> quitting): ")
        raise
    #finally:
    #    sys.exit(0)
