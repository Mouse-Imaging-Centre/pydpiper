#!/usr/bin/env python3

import time
import sys
import os
from configargparse import ArgParser, Namespace  # type: ignore
from datetime import datetime
from multiprocessing import Process, Pool, Lock # type: ignore
import subprocess
import shlex
import pydpiper.execution.queueing as q
import math as m
import logging
import socket
import signal
import threading
os.environ["PYRO_LOGLEVEL"] = os.getenv("PYRO_LOGLEVEL", "INFO")
import Pyro4       # type: ignore
from typing import Any

from pyminc.volumes.volumes import mincException


Pyro4.config.REQUIRE_EXPOSE = False
Pyro4.config.SERVERTYPE = "multiplex"

class SubmitError(ValueError): pass

for boring_exception, name in [(mincException, "mincException"), (SubmitError, "SubmitError"), (KeyError,"KeyError")]:
    Pyro4.util.SerializerBase.register_dict_to_class(name, lambda _classname, dict: boring_exception)
    Pyro4.util.SerializerBase.register_class_to_dict(boring_exception, lambda obj: { "__class__" : name })


#TODO add these to executorArgumentGroup as options, pass into pipelineExecutor
EXECUTOR_MAIN_LOOP_INTERVAL = 30.0
HEARTBEAT_INTERVAL = EXECUTOR_MAIN_LOOP_INTERVAL  # Logically necessary since the two threads have been merged
#SHUTDOWN_TIME = EXECUTOR_MAIN_LOOP_INTERVAL + LATENCY_TOLERANCE

logger = logging # type: Any
#logger = logging.getLogger(__name__)

sys.excepthook = Pyro4.util.excepthook  # type: ignore


def ensure_exec_specified(numExec):
    if numExec < 1:
        msg = "You need to specify some executors for this pipeline to run. Please use the --num-executors command line option. Exiting..."
        logger.info(msg)
        print(msg)
        sys.exit(1)

def launchExecutor(executor):
    # Start executor that will run pipeline stages

    # getIpAddress is similar to socket.gethostbyname(...) 
    # but uses a hack to attempt to avoid returning localhost (127....)
    network_address = Pyro4.socketutil.getIpAddress(socket.gethostname(),
                                                    workaround127 = True, ipVersion = 4)
    daemon = Pyro4.core.Daemon(host=network_address)
    clientURI = daemon.register(executor)

    # find the URI of the server:
    if executor.ns:
        ns = Pyro4.locateNS()
        #ns.register("executor", executor, safe=True)
        serverURI = ns.lookup("pipeline")
    else:
        try:
            uf = open(executor.uri_file)
            serverURI = Pyro4.URI(uf.readline())
            uf.close()
        except:
            logger.exception("Problem opening the specified uri file:")
            raise

    p = Pyro4.Proxy(serverURI)
    # Register the executor with the pipeline
    # the following command only works if the server is alive. Currently if that's
    # not the case, the executor will die which is okay, but this should be
    # more properly handled: a more elegant check to verify the server is running
    p.registerClient(clientURI.asString(), executor.mem)

    executor.registeredWithServer()
    executor.setClientURI(clientURI.asString())
    executor.setServerURI(serverURI.asString())
    executor.setProxyForServer(p)
    
    logger.info("Connected to %s",  serverURI)
    logger.info("Client URI is %s", clientURI)
    
    executor.connection_time_with_server = time.time()
    logger.info("Connected to the server at: %s", datetime.isoformat(datetime.now(), " "))

    # you'd think this could be done in __init__, but sometimes that's in the wrong process,
    # which causes an error when calling `join` ...
    executor.initializePool()

    logger.debug("Executor daemon running at: %s", daemon.locationStr)
    try:
        # run the daemon, not the executor mainLoop, in a new thread
        # so that mainLoop exceptions (e.g., if we lose contact with the server)
        # cause us to shutdown (as Python makes it tedious to re-throw to calling thread)
        def with_exception_logging(f, thread_description, crash_hook=None):
            def _f(*args, **kwargs):
                try:
                    return f(*args, **kwargs)
                except Exception:
                    logger.exception("Crash in '%s' thread!  Details: " % thread_description)
                    crash_hook() if crash_hook else ()
                    raise
            return _f
        t = threading.Thread(target=with_exception_logging(daemon.requestLoop, "Pyro daemon"))
        t.daemon = True
        t.start()
        executor.mainLoop()
    except KeyboardInterrupt:
        logger.exception("Caught keyboard interrupt. Shutting down executor...")
        executor.generalShutdownCall()
        #daemon.shutdown()
        sys.exit(1)
    except Exception:
        logger.exception("Error during executor loop. Shutting down executor...")
        executor.generalShutdownCall()
        #daemon.shutdown()
        sys.exit(1)
    else:
        executor.completeAndExitChildren()
        logger.info("Executor shutting down.")
        daemon.shutdown()
        t.join()


# like a stage but lighter weight (no methods wasting memory...)
class StageInfo(object):
    def __init__(self, *, mem, procs, ix, cmd, log_file, output_files):
        self.mem = mem
        self.procs = procs
        self.ix = ix
        self.cmd = cmd
        self.log_file = log_file
        self.output_files = output_files


def stageinfo_dict_to_class(classname, d):
    return StageInfo(mem=d['mem'], procs=d['procs'], ix=d['ix'], cmd=d['cmd'], log_file=d['log_file'],
                     output_files=d['output_files'])


Pyro4.util.SerializerBase.register_dict_to_class("pydpiper.execution.pipeline_executor.StageInfo",
                                                 stageinfo_dict_to_class)


class MissingOutputs(ValueError): pass

def runStage(*, clientURI, stage, cmd_wrapper):
        ix = stage.ix

        logger.info("Running stage %i (on %s). Memory requested: %.2f", ix, clientURI, stage.mem)
        try:
            command_to_run  = ((cmd_wrapper + ' ') if cmd_wrapper else '') + ' '.join(stage.cmd)

            logger.info(command_to_run)
            command_logfile = stage.log_file

            # log file for the stage
            with open(command_logfile, 'a') as of:
                of.write("Stage " + str(ix) + " running on " + socket.gethostname()
                         + " (" + clientURI + ") at " + datetime.isoformat(datetime.now(), " ") + ":\n")
                of.write(command_to_run + "\n")
                of.flush()

                #args = shlex.split(command_to_run)
                args = command_to_run
                start_time = time.time()
                process = subprocess.Popen(args, stdout=of, stderr=of, shell=True)
                #client.addPIDtoRunningList(process.pid)
                process.communicate()
                #client.removePIDfromRunningList(process.pid)
                ret = process.returncode
                if ret == 0:
                    # TODO: better logic here, e.g., allow some tolerance for NFS slowness, etc.
                    missing_outputs = [o for o in stage.output_files
                                       if (not os.path.exists(o)) or os.path.getmtime(o) < start_time]
                    if len(missing_outputs) > 0:
                        logger.warning("some outputs not produced by Stage %i: %s", ix, missing_outputs)
                        of.write("[executor] ERROR: outputs not produced, failing this stage: %s\n" % missing_outputs)
                        raise MissingOutputs(missing_outputs)
        except Exception as e:
            logger.exception("Exception whilst running stage: %i (on %s)", ix, clientURI)
            return ix, e
        else:
            # TODO: the big try-catch block above is quite ugly ...
            logger.info("Stage %i finished, return was: %i (on %s)", ix, ret, clientURI)

            return ix, ret


class ChildProcess(object):
    """Used by the executor to store runtime information about the child processes it initiates to run commands."""
    def __init__(self, stage, result, mem, procs):
        self.stage = stage
        self.result = result
        self.mem = mem
        self.procs = procs

class InsufficientResources(Exception):
    pass

class pipelineExecutor(object):
    def __init__(self, options, uri_file, memNeeded = None):
        # better: self.options = options ... ?
        # TODO the additional argument `mem` represents the
        # server's estimate of the amount of memory
        # an executor may need to run available jobs
        # -- perhaps options.mem should be renamed
        # options.max_mem since this represents a per-node
        # limit (or at least a per-executor limit)
        logger.info("memNeeded: %s", memNeeded)
        self.mem = memNeeded or options.mem
        logger.info("self.mem = %0.2fG", self.mem)
        if self.mem > options.mem:
            raise InsufficientResources("executor requesting %.2fG memory but maximum is %.2fG" % (options.mem, self.mem))
        self.procs = options.proc
        self.ppn = options.ppn
        self.pe  = options.pe
        self.mem_request_attribute = options.mem_request_attribute
        self.time = options.time
        self.queue_type = options.queue_type
        self.queue_name = options.queue_name
        self.queue_opts = options.queue_opts
        self.cmd_wrapper = options.cmd_wrapper
        self.executor_wrapper = options.executor_wrapper
        self.ns = options.use_ns
        self.uri_file = options.urifile
        if self.uri_file is None:
            self.uri_file = os.path.abspath(os.path.join(os.curdir, uri_file))
        # the next variable is used to keep track of how long the
        # executor has been continuously idle/sleeping for. Measured
        # in seconds
        self.idle_time = 0
        self.prev_time = None
        self.current_time = None
        # the maximum number of minutes an executor can be continuously
        # idle for, before it has to kill itself.
        self.time_to_seppuku = options.time_to_seppuku
        # the time in minutes after which an executor will not accept new jobs
        self.time_to_accept_jobs = options.time_to_accept_jobs
        # stores the time of connection with the server
        self.connection_time_with_server = None
        #initialize runningMem and Procs
        self.runningMem = 0.0
        self.runningProcs = 0   
        self.runningChildren = {}  # was: # no scissors (i.e. children should not run around with sharp objects...)
        self.lock = Lock()
        self.pool = None  # type: Pool
        self.pyro_proxy_for_server = None
        self.clientURI = None
        self.serverURI = None
        #self.current_running_job_pids = []
        self.registered_with_server = False
        # we associate an event with each executor which is set when jobs complete.
        # in the future it might also be set by the server, and we might have more
        # than one event (for reclaiming, server messages, ...)
        self.e = threading.Event()
        self.heartbeat_tick = 0

    def wrapPyroCall(self, func, *args, **kwargs):
        try:
            # a bit expensive perhaps, but it is safer to create
            # a new proxy for the server whenever it is contacted.
            # We found that connecting to the server via the same proxy using several
            # Process-es, can bring down either or both the server and the client
            # (the documentation also states that proxies can only be shared between
            # threads).
            # also, note Ben and his bag of tricks! When a function on the server
            # side needs to be called, and wrapPyroCall is invoked, we do this
            # using the lambda functionality. Below the lambda p: p.call_at_the_server
            # will pass the new proxy to p. At the same time, pycharm is still able
            # to type check things.
            logger.debug("wrapPyroCall: %s", func)
            with Pyro4.Proxy(self.serverURI) as one_time_proxy:
                return func(one_time_proxy)(*args, **kwargs)
        except:
            logger.exception("Exception while placing a Pyro call at the server: %s", func)
            raise Exception("Pyro call with the server failed. Shutting down...")

    def registeredWithServer(self):
        self.registered_with_server = True

    #@Pyro4.oneway
    #def addPIDtoRunningList(self, pid):
    #    self.current_running_job_pids.append(pid)

    #@Pyro4.oneway
    #def removePIDfromRunningList(self, pid):
    #    self.current_running_job_pids.remove(pid)

    def initializePool(self):
        self.pool = Pool(processes = self.procs)
        
    def setClientURI(self, cURI):
        self.clientURI = cURI 
            
    def setServerURI(self, sURI):
        self.serverURI = sURI
            
    def setProxyForServer(self, proxy):
        self.pyro_proxy_for_server = proxy
    
    # TODO rename completeAndExitChildren,generalShutdownCall to something like
    # normalShutdown, dirtyShutdown
    def generalShutdownCall(self):
        # FIXME this comment is outdated (but is the problem described still a problem?)
        # stop the worker processes (children) immediately without completing outstanding work
        # Initially I wanted to stop the running processes using pool.terminate() and pool.join()
        # but the keyboard interrupt handling proved tricky. Instead, the executor now keeps
        # track of the process IDs (pid) of the current running jobs. Those are targetted by
        # os.kill in order to stop the processes in the Pool
        logger.info("Executor shutting down.  Killing running jobs...")
        #for subprocID in self.current_running_job_pids:
        #    os.kill(subprocID, signal.SIGTERM)
        self.pool.terminate()
        self.pool.join()
        logger.debug("Finished joining process pool.")
        # FIXME the death of the child process causes runStage
        # to notify the server of the job's destruction
        # so the job is no longer in the client's set of stages
        # when unregisterClient is called
        self.unregister_with_server()

    def completeAndExitChildren(self):
        # This function is called under normal circumstances (i.e., not because
        # of a keyboard interrupt). So we can close the pool of processes 
        # in the normal way, prevent more jobs from starting, and exit
        self.unregister_with_server()
        if len(self.runningChildren) > 0:
            logger.warning("Exiting with some processes still running: %s" % self.runningChildren)
        # wait for the worker processes (children) to exit (must be called after terminate() or close())
        self.pool.close()
        self.pool.join()

    def unregister_with_server(self):
        if self.registered_with_server:
            # unset the registered flag before calling unregisterClient
            # to prevent an (unimportant) race condition wherein the
            # unregisterClient() call begins while, simultaneously, the heartbeat
            # thread finds the flag true and so sends a heartbeat
            # request to the server, which raises an exception as the client has
            # since unregistered, so is no longer present in some data structure
            # (it's OK if the heartbeat begins before the flag is unset
            # since the server runs single-threaded)
            logger.info("Unsetting the registered-with-the-server flag for executor: %s", self.clientURI)
            self.registered_with_server = False
            self.wrapPyroCall(lambda p: p.unregisterClient, self.clientURI)
            logger.info("Done calling unregisterClient")

    def submitToQueue(self, number):
        """Submits to queueing system using qbatch"""
        if self.queue_type not in ['sge', 'pbs', 'slurm', None]:
            msg = ("Specified queueing system is: %s" % (self.queue_type) +
                   "Only `queue_type`s 'sge', 'pbs', and None currently support launching executors." +
                   "Exiting...")
            logger.warning(msg)
            sys.exit(msg)
        else:
            now = datetime.now().strftime("%Y-%m-%d-at-%H-%M-%S")
            ident = "pipeline-executor-" + now
            #jobname = ((os.path.basename(program_name) + '-') if program_name is not None else "") + ident
            jobname = ident
            # do we really need the program name here?
            env = os.environ.copy()
            cmd = ((self.executor_wrapper.split() if self.executor_wrapper else [])
                  + (["pipeline_executor.py", "--local",
                       '--uri-file', self.uri_file,
                       # Only one exec is launched at a time in this manner, so:
                       "--num-executors", str(1), '--mem', str(self.mem)]
                     + q.remove_flags(['--num-exec', '--mem'], sys.argv[1:])))

            #     header = '\n'.join(["#!/usr/bin/env bash",
            #                         # why `csh`?  It seems that `qsub` doesn't allow one to
            #                         # pass args to the shell specified with `-S`, so we can't pass --noprofile
            #                         # (or --norc) to bash.  As a result, /etc, ~/.bashrc, etc.,
            #                         # are read again by the executors, which is unlikely to be intended.
            #                         # Since no-one uses csh, this is less likely to be a problem.
            #                         # (This was implicitly happening before the `#$ -S ...` line was added
            #                         # and none of our many users complained...)
            #                         "#$ -S /bin/csh",
            #                         "setenv PYRO_LOGFILE logs/%s-${JOB_ID}-${SGE_TASK_ID}.log" % ident])
            #     # FIXME huge hack -- shouldn't we just iterate over options,
            #     # possibly checking for membership in the executor option group?
            #     # The problem is that we can't easily check if an option is
            #     # available from a parser (but what about calling get_defaults and
            #     # looking at exceptions?).  However, one possibility is to
            #     # create a list of tuples consisting of the data with which to
            #     # call parser.add_arguments and use this to check.
            #     # NOTE there's a problem with argparse's prefix matching which
            #     # also affects removal of --num-executors
            #
            # TODO: procs! ppn! umask for log files? log file names? jobname? (see version 2.0.8)
            if self.ppn > 1:
                logger.warning("ppn of %d currently ignored in this configuration" % self.ppn)
            cmd_str = ' '.join(cmd)
            script = '\n'.join([cmd_str for _ in range(number)])
            submit_cmd = (["qbatch",
                           "--chunksize=1",
                           "--cores=1",      # qbatch should run each executor as a separate job
                           #"--ppj=%s", self.ppn,
                           "--mem=%sGB" % m.ceil(self.mem)]  # some schedulers don't like floats
                           # TODO expose the rest of qbatch's options here (e.g. --footer, etc.?)
                           # the following options aren't really needed if qbatch is configured separately:
                           + (["-b", self.queue_type] if self.queue_type else [])
                           + (["--queue=%s" % self.queue_name] if self.queue_name else [])
                           # TODO change to "memvars" to match qbatch?
                           + (["--walltime=%s" % self.time] if self.time else [])  # is time ever falsy??
                           + (["--memvars=%s" % self.mem_request_attribute] if self.mem_request_attribute else [])
                           + (["--pe=%s" % self.pe] if self.pe else [])  # TODO: only if 'sge' ?
                           # TODO: add queue opts via qbatch -S (or - more general - change to qbatch_opts??)
                           + ["-"])
            #print(submit_cmd)
            #p = subprocess.Popen(submit_cmd, stdin=subprocess.PIPE, shell=False, env=env)
            #_out_data, _err_data = p.communicate(script.encode('ascii'))
            #print(out_data)
            #print(err_data, file=sys.stderr)
            # TODO better error reporting here ??
            p = subprocess.run(submit_cmd, input=script.encode('ascii'), env=env)
            if p.returncode != 0:
                raise SubmitError({ 'return' : p.returncode, 'failed_command' : submit_cmd })

    def canRun(self, stageMem, stageProcs, runningMem, runningProcs):
        """Calculates if stage is runnable based on memory and processor availability"""
        return stageMem <= self.mem - runningMem and stageProcs <= self.procs - runningProcs
    def is_seppuku_time(self):
        # Is it time to perform seppuku: has the
        # idle_time exceeded the allowed time to be idle?
        # time_to_seppuku is given in minutes
        # idle_time       is given in seconds
        if self.time_to_seppuku != None:
            if (self.time_to_seppuku * 60) < self.idle_time:
                return True
        return False
                        
    def is_time_to_drain(self):
        # check whether there is a limit to how long the executor
        # is allowed to accept jobs for. 
        if (self.time_to_accept_jobs != None):
            current_time = time.time()
            time_take_so_far = current_time - self.connection_time_with_server
            minutes_so_far, seconds_so_far = divmod(time_take_so_far, 60)
            if self.time_to_accept_jobs < minutes_so_far:
                return True
        return False

    # TODO do this cleanup in the callback!
    #def free_resources(self):
    #    # Free up resources from any completed (successful or otherwise) stages
    #    for child in self.runningChildren:
    #        if child.result.ready():
    #            logger.debug("Freeing up resources for stage %i.", child.stage)
    #            self.runningMem -= child.mem
    #            self.runningProcs -= child.procs
    #            self.runningChildren.remove(child)

    #@Pyro4.oneway
    def notifyStageTerminated(self, i, returncode=None):
        #try:
            if returncode == 0:
                logger.debug("Setting stage %d finished on the server side", i)
                self.wrapPyroCall(lambda p: p.setStageFinished, i, self.clientURI)
                logger.debug("Done setting stage finished")
            else:
                # a None returncode is also considered a failure
                logger.debug("Setting stage %d failed on the server side. Return code: %s", i, returncode)
                self.wrapPyroCall(lambda p: p.setStageFailed, i, self.clientURI)
                logger.debug("Done setting stage failed")
            # the server may have shutdown or otherwise become unavailable
            # (currently this is expected when a long-running job completes;
            # we should add a more elegant check for this state of affairs),
            # but the executor may have running jobs that shouldn't be killed
            # TODO add similar error handling around certain other Pyro calls)
            self.e.set()  # some work finished and server notified, so wake up

    def idle(self):
        return self.runningMem == 0 and self.runningProcs == 0 and self.prev_time

    # use an event set/timeout system to run the executor mainLoop -
    # we might want to pass some extra information in addition to waking the system
    def mainLoop(self):
        while self.mainFn():
            self.e.wait(EXECUTOR_MAIN_LOOP_INTERVAL)
            self.e.clear()
        logger.info("Main loop finished")

    def mainFn(self):
        """Try to get a job from the server (if appropriate) and update
        internal state accordingly.  Return True if it should be called
        again (i.e., there is more to do before shutting down),
        otherwise False (contract: but never False if there are still running jobs)."""

        self.prev_time = self.current_time
        self.current_time = time.time()

        # a bit coarse but we can't call `free_resources` directly in a function
        # such as notifyStageTerminated which is called from _within_ `runStage`
        # since resources won't be freed soon enough, causing a false resource starvation.
        # note we don't do resource accounting after leaving mainLoop, though that
        # doesn't matter too much as there will never be new jobs
        # (unless, in the future, we allow clients to connect to switch allegiances
        # to other servers)
        #self.free_resources()

        logger.debug("Updating timestamp with heartbeat tick: %d", self.heartbeat_tick)
        self.wrapPyroCall(lambda p: p.updateClientTimestamp, self.clientURI, tick=self.heartbeat_tick)
        self.heartbeat_tick += 1
        logger.debug("Done updating timestamp")

        if self.idle():
            self.idle_time += self.current_time - self.prev_time
            logger.debug("Current idle time: %d, and total seconds allowed: %d",
                         self.idle_time, self.time_to_seppuku * 60)

        if self.is_seppuku_time():
            logger.warning("Exceeded allowed idle time... Seppuku!")
            return False

        # It is possible that the executor does not accept any new jobs
        # anymore. If that is the case, we can leave this main loop
        # and just wait until current running jobs (children) have finished
        if self.is_time_to_drain():
            logger.debug("Time expired for accepting new jobs")  #...leaving main loop.")
            # was logger.info; made this a debug since we currently don't bail out of the loop ...
            #return False
            return True

        # TODO we get only one stage per loop iteration, so we have to wait for
        # another event/timeout to get another.  In general we might want 
        # getCommand to order multiple stages to be run on the same server
        # (just setting the event immediately would be somewhat hackish)
        # FIXME send/get the whole stageinfo here (it contains an `ix` field, right?)?!
        logger.debug("Going to get command from server")
        cmd, i = self.wrapPyroCall(lambda p: p.getCommand, clientURIstr=self.clientURI,
                                                           clientMemFree=self.mem - self.runningMem,
                                                           clientProcsFree=self.procs - self.runningProcs)
        logger.debug("Done getting command from server")

        if cmd == "shutdown_normally":
            logger.info('Saw shutdown command from server')
            return False
        # TODO this won't work yet since we'll just go to shutdown normally
        # and wait for jobs to finish instead of killing them -
        # maybe throwing an exception is better?
        elif cmd == "wait":
            return True
        elif cmd == "run_stage":
            logger.debug("Going to get stage info for stage: %d", i)
            stage = self.wrapPyroCall(lambda p: p.get_stage_info,i)
            logger.debug("Done getting stage information for stage: %d", i)
            # we trust that the server has given us a stage
            # that we have enough memory and processors to run ...
            # reset the idle time, we are running a stage!
            self.idle_time = 0
            with self.lock:
                self.runningMem += stage.mem
                self.runningProcs += stage.procs
            # The multiprocessing library must pickle things in order to execute them.
            # I wanted the following function (runStage) to be a function of the pipelineExecutor
            # class. That way we can access self.serverURI and self.clientURI from
            # within the function. However, bound methods are not picklable (a bound method
            # is a method that has "self" as its first argument, because if I understand 
            # this correctly, that binds the function to a class instance). There is
            # a way to make a bound function picklable, but this seems cumbersome. So instead
            # runStage is now a standalone function.

            # callback for result of runStage, run by executor
            def process_result(result):
                ix, res = result
                if isinstance(res, int):
                    # it's a return code
                    # don't do this logging in the callback for politenessoliphant
                    self.notifyStageTerminated(ix, res)
                elif isinstance(res, Exception):
                    # runStage raised an exception.  We could use apply_async's error_callback to handle this case
                    # instead, but we need to know the index of the stage we were attempting to run, so we'd have
                    # to catch the exception anyway to stuff the index into it ... this seems cleaner (no re-raising).
                    self.notifyStageTerminated(ix)
                logger.debug("Freeing up resources for stage %i.", ix)
                stage = self.runningChildren[ix]
                with self.lock:
                    self.runningMem -= stage.mem
                    self.runningProcs -= stage.procs
                del self.runningChildren[ix]

            # why does this need a separate call? should be able to infer that this stage will start from getCommand...
            logger.debug("Telling the server that stage %d has started", i)
            self.wrapPyroCall(lambda p: p.setStageStarted, i, self.clientURI)
            logger.debug("Server knows that stage started")
            result = self.pool.apply_async(runStage, args=(),
                                           kwds={ "clientURI" : self.clientURI, "stage" : stage,
                                                  "cmd_wrapper" : self.cmd_wrapper},
                                           callback=process_result)
            self.runningChildren[i] = ChildProcess(i, result, stage.mem, stage.procs)

            logger.debug("Added stage %i to the running pool.", i)
            return True
        else:
            raise Exception("Got invalid cmd from server: %s" % cmd)
                

def main():
    # command line option handling
    # use an environment variable to look for a default config file
    # Alternately, we could use a default location for the file
    # (say `files = ['/etc/pydpiper.cfg', '~/pydpiper.cfg', './pydpiper.cfg']`)
    # TODO this logic is duplicated in application.py
    default_config_file = os.getenv("PYDPIPER_CONFIG_FILE")
    if default_config_file is not None:
        files = [default_config_file]
    else:
        files = []
    parser = ArgParser(default_config_files=files)    

    from pydpiper.core.arguments import _mk_execution_parser
    _mk_execution_parser(parser)

    # using parse_known_args instead of parse_args is a hack since we
    # currently send ALL arguments from the main program to the executor.
    # Alternately, we could keep a copy of the executor parser around
    # when constructing the executor shell command
    options, _ = parser.parse_known_args()

    ensure_exec_specified(options.num_exec)

    def local_launch(options):
        pe = pipelineExecutor(options=options, uri_file=options.urifile)  #, pipeline_name=options.pipeline_name)
        # FIXME - I doubt missing the other options even works, otherwise we could change the executor interface!!
        # executors don't use any shared-memory constructs, so OK to copy
        ps = [Process(target=launchExecutor, args=(pe,))
              for _ in range(options.num_exec)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()

    if options.local:
        local_launch(options)
    elif options.submit_server:
        roq = q.runOnQueueingSystem(options, sysArgs=sys.argv)
        for i in range(options.num_exec):
            roq.createAndSubmitExecutorJobFile(i, after=None,
                                               time=q.timestr_to_secs(options.time))
    elif options.queue_type is not None:
        for i in range(options.num_exec):
            pe = pipelineExecutor(options=options, uri_file=options.urifile)   #, pipeline_name=pipeline_name)
            pe.submitToQueue(1)  # TODO is there a reason why we have logic for submitting `i` executors again here?
    else:
        local_launch(options)

if __name__ == "__main__":
    main()
