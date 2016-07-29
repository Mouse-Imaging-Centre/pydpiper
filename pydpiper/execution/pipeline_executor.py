#!/usr/bin/env python3

import time
import sys
import os
from configargparse import ArgParser, Namespace  # type: ignore
from datetime import datetime
from multiprocessing import Process, Pool # type: ignore
import subprocess
import shlex
import pydpiper.execution.queueing as q
import math as m
import logging
import socket
import signal
import threading
import Pyro4       # type: ignore
from typing import Any

from pyminc.volumes.volumes import mincException


Pyro4.config.REQUIRE_EXPOSE = False
Pyro4.config.SERVERTYPE = "multiplex"


class SubmitError(ValueError): pass

for boring_exception, name in [(mincException, "mincException"), (SubmitError, "SubmitError")]:
    Pyro4.util.SerializerBase.register_dict_to_class(name, lambda _classname, dict: boring_exception)
    Pyro4.util.SerializerBase.register_class_to_dict(boring_exception, lambda obj: { "__class__" : name })


#TODO add these to executorArgumentGroup as options, pass into pipelineExecutor
EXECUTOR_MAIN_LOOP_INTERVAL = 10.0
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
        #h = threading.Thread(target=with_exception_logging(executor.heartbeat, "heartbeat",
        #                                                   crash_hook=lambda : setattr(executor,  # Python is 'funny'
        #                                                                               "heartbeat_thread_crashed",
        #                                                                               True)))
        #h.daemon = True
        #h.start()
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

def runStage(serverURI, clientURI, i, cmd_wrapper):
    ## Proc needs its own proxy as it's independent of executor
    p = Pyro4.core.Proxy(serverURI)
    client = Pyro4.core.Proxy(clientURI)
    
    # Retrieve stage information, run stage and set finished or failed accordingly  
    try:
        logger.info("Running stage %i (on %s)", i, clientURI)
        logger.debug("Memory requested: %.2f", p.getStageMem(i))
        p.setStageStarted(i, clientURI)
        try:
            # get stage information
            command_to_run  = ((cmd_wrapper + ' ') if cmd_wrapper else '') + str(p.getStageCommand(i))
            logger.info(command_to_run)
            command_logfile = p.getStageLogfile(i)
            
            # log file for the stage
            of = open(command_logfile, 'a')
            of.write("Stage " + str(i) + " running on " + socket.gethostname() + " (" + clientURI + ") at " + datetime.isoformat(datetime.now(), " ") + ":\n")
            of.write(command_to_run + "\n")
            of.flush()
            
            args = shlex.split(command_to_run) 
            process = subprocess.Popen(args, stdout=of, stderr=of, shell=False)
            client.addPIDtoRunningList(process.pid)
            process.communicate()
            client.removePIDfromRunningList(process.pid)
            ret = process.returncode 
            of.close()
        except:
            logger.exception("Exception whilst running stage: %i (on %s)", i, clientURI)   
            client.notifyStageTerminated(i)
        else:
            logger.info("Stage %i finished, return was: %i (on %s)", i, ret, clientURI)
            client.notifyStageTerminated(i, ret)

        # If completed, return mem & processes back for re-use
        return (p.getStageMem(i), p.getStageProcs(i))
    except:
        logger.exception("Error communicating to server in runStage. " 
                        "Error raised to calling thread in launchExecutor. ")
        raise     


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
        logger.debug("memNeeded: %s", memNeeded)
        self.mem = memNeeded or options.mem
        logger.debug("self.mem = %0.2fG", self.mem)
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
        self.runningChildren = [] # no scissors (i.e. children should not run around with sharp objects...)
        self.pool = None
        self.pyro_proxy_for_server = None
        self.clientURI = None
        self.serverURI = None
        self.current_running_job_pids = []
        self.registered_with_server = False
        # we associate an event with each executor which is set when jobs complete.
        # in the future it might also be set by the server, and we might have more
        # than one event (for reclaiming, server messages, ...)
        self.e = threading.Event()
        
    def registeredWithServer(self):
        self.registered_with_server = True

    #@Pyro4.oneway
    def addPIDtoRunningList(self, pid):
        self.current_running_job_pids.append(pid)

    #@Pyro4.oneway
    def removePIDfromRunningList(self, pid):
        self.current_running_job_pids.remove(pid)

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
        # stop the worker processes (children) immediately without completing outstanding work
        # Initially I wanted to stop the running processes using pool.terminate() and pool.join()
        # but the keyboard interrupt handling proved tricky. Instead, the executor now keeps
        # track of the process IDs (pid) of the current running jobs. Those are targetted by
        # os.kill in order to stop the processes in the Pool
        logger.debug("Executor shutting down.  Killing running jobs:")
        for subprocID in self.current_running_job_pids:
            os.kill(subprocID, signal.SIGTERM)
        # FIXME the death of the child process causes runStage
        # to notify the server of the job's destruction
        # so the job is no longer in the client's set of stages
        # when unregisterClient is called
        self.unregister_with_server()

    def completeAndExitChildren(self):
        # This function is called under normal circumstances (i.e., not because
        # of a keyboard interrupt). So we can close the pool of processes 
        # in the normal way (don't need to use the pids here)
        # prevent more jobs from starting, and exit
        if len(self.current_running_job_pids) > 0:
            self.pool.close()
            # wait for the worker processes (children) to exit (must be called after terminate() or close()
            self.pool.join()
        self.unregister_with_server()

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
            logger.info("Now going to call unregisterClient on the server (executor: %s)", self.clientURI)
            self.pyro_proxy_for_server.unregisterClient(self.clientURI)

    def submitToQueue(self, number):
        """Submits to queueing system using qsub"""
        if self.queue_type not in ['sge', 'pbs']:
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

            os.system("mkdir -p logs")    # FIXME: this really doesn't belong here
            if self.queue_type == "sge":
                strprocs = str(self.procs)
                strmem = "%s=%sG" % (self.mem_request_attribute, float(self.mem))
                queue_opts = (['-V', '-j', 'yes', '-cwd', '-t', '1-%d' % number,
                              '-N', jobname,
                              '-l', strmem,
                              '-o', "logs/%s-$JOB_ID-$TASK_ID-eo.log" % jobname]
                              + (['-q', self.queue_name]
                                if self.queue_name else [])
                              + (['-pe', self.pe, strprocs]
                                if self.pe else [])
                              + shlex.split(self.queue_opts))
                qsub_cmd = ['qsub'] + queue_opts

                header = '\n'.join(["#!/usr/bin/env bash",
                                    # why `csh`?  It seems that `qsub` doesn't allow one to
                                    # pass args to the shell specified with `-S`, so we can't pass --noprofile
                                    # (or --norc) to bash.  As a result, /etc, ~/.bashrc, etc.,
                                    # are read again by the executors, which is unlikely to be intended.
                                    # Since no-one uses csh, this is less likely to be a problem.
                                    # (This was implicitly happening before the `#$ -S ...` line was added
                                    # and none of our many users complained...)
                                    "#$ -S /bin/csh",
                                    "setenv PYRO_LOGFILE logs/%s-${JOB_ID}-${SGE_TASK_ID}.log" % ident])
                # FIXME huge hack -- shouldn't we just iterate over options,
                # possibly checking for membership in the executor option group?
                # The problem is that we can't easily check if an option is
                # available from a parser (but what about calling get_defaults and
                # looking at exceptions?).  However, one possibility is to
                # create a list of tuples consisting of the data with which to
                # call parser.add_arguments and use this to check.
                # NOTE there's a problem with argparse's prefix matching which
                # also affects removal of --num-executors

            elif self.queue_type == 'pbs':
                if self.ppn > 1:
                    logger.warning("ppn of %d currently ignored in this configuration" % self.ppn)
                if "PBS_O_WORKDIR" in env:
                    del env["PBS_O_WORKDIR"]  # because on CCM, this is set to /home/user on qlogin nodes ...
                    del env["PBS_JOBID"]
                header = '\n'.join(["#!/usr/bin/env bash",
                                    "#PBS -N %s" % jobname,
                                    "#PBS -l nodes=1:ppn=1",
                                    # CCM is strict, and doesn't like float values:
                                    "#PBS -l walltime:%s" % (self.time),
                                    "#PBS -l %s=%dg\n" % (self.mem_request_attribute, m.ceil(self.mem)),
                                    # FIXME add walltime stuff here if specified (and check <= max_walltime ??)
                                    "df /dev/shm >&2",  # FIXME: remove
                                    "cd $PBS_O_WORKDIR",
                                    "export PYRO_LOGFILE=logs/%s-${PBS_JOBID}.log" % ident])
                qsub_cmd = (['qsub', '-V', '-t', '1-%d' % number,
                             '-o', "logs/%s-$PBS_JOBID-o.log" % ident,
                             '-e', "logs/%s-$PBS_JOBID-e.log" % ident,
                             '-Wumask=0137']
                             + (['-q', self.queue_name] if self.queue_name else []))

            script = header + '\n' + ' '.join(cmd) + '\n'
            #print(script)
            # TODO change to use subprocess.run(qsub_cmd, input=...) (Python >= 3.5)
            p = subprocess.Popen(qsub_cmd, stdin=subprocess.PIPE, shell=False, env=env)
            _out_data, _err_data = p.communicate(script.encode('ascii'))
            #print(out_data)
            #print(err_data, file=sys.stderr)
            if p.returncode != 0:
                raise SubmitError({ 'return' : p.returncode, 'failed_command' : qsub_cmd })

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
    
    def free_resources(self):
        # Free up resources from any completed (successful or otherwise) stages
        for child in self.runningChildren:
            if child.result.ready():
                logger.debug("Freeing up resources for stage %i.", child.stage)
                self.runningMem -= child.mem
                self.runningProcs -= child.procs
                self.runningChildren.remove(child)

    #@Pyro4.oneway
    def notifyStageTerminated(self, i, returncode=None):
        #try:
            if returncode == 0:
                self.pyro_proxy_for_server.setStageFinished(i, self.clientURI)
            else:
                # a None returncode is also considered a failure
                self.pyro_proxy_for_server.setStageFailed(i, self.clientURI)
        #except Pyro4.errors.CommunicationError:
            # the server may have shutdown or otherwise become unavailable
            # (currently this is expected when a long-running job completes;
            # we should add a more elegant check for this state of affairs),
            # but the executor may have running jobs that shouldn't be killed
            # TODO add similar error handling around certain other Pyro calls)
        #    logger.info("Error communing with server; couldn't notify it of stage %d's termination", i)
            self.e.set()  # some work finished and server notified, so wake up

    def idle(self):
        return self.runningMem == 0 and self.runningProcs == 0 and self.prev_time

    def heartbeat(self):
        tick = 0
        while self.registered_with_server:
            logger.debug("Sending heartbeat %d...", tick)
            tick += 1
            self.pyro_proxy_for_server.updateClientTimestamp(self.clientURI, tick)
            logger.debug("...finished")
            time.sleep(HEARTBEAT_INTERVAL)
            # this will take down the executor to avoid the case
            # where an executor wastes time processing jobs which the server
            # considers lost; there might be a better way to do this
            # (re-register/restart heartbeat and notify server of existing
            # jobs? quite complicated ...), and it could be done
            # 'atomically' using an event for better guarantees ...

    # use an event set/timeout system to run the executor mainLoop -
    # we might want to pass some extra information in addition to waking the system
    def mainLoop(self):
        while self.mainFn():
            self.e.wait(EXECUTOR_MAIN_LOOP_INTERVAL)
            self.e.clear()
        logger.debug("Main loop finished")

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
        self.free_resources()

        logger.debug("Updating timestamp...")
        self.pyro_proxy_for_server.updateClientTimestamp(self.clientURI, tick=42)  # FIXME (42)
        logger.debug("Done")
        #if self.heartbeat_thread_crashed:
        #    logger.debug("Heartbeat thread crashed; quitting")
        #    return False

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
            logger.info("Time expired for accepting new jobs")  #...leaving main loop.")
            #return False
            return True

        # TODO we get only one stage per loop iteration, so we have to wait for
        # another event/timeout to get another.  In general we might want 
        # getCommand to order multiple stages to be run on the same server
        # (just setting the event immediately would be somewhat hackish)
        cmd, i = self.pyro_proxy_for_server.getCommand(clientURIstr = self.clientURI,
                                                       clientMemFree = self.mem - self.runningMem,
                                                       clientProcsFree = self.procs - self.runningProcs)
        if cmd == "shutdown_normally":
            logger.debug('Saw shutdown command from server')
            return False
        #elif cmd == "shutdown_immediately":
        #    logger.debug('Saw immediate shutdown command - killing jobs ...')
        #    return False
        # TODO this won't work yet since we'll just go to shutdown normally
        # and wait for jobs to finish instead of killing them -
        # maybe throwing an exception is better?
        elif cmd == "wait":
            return True
        elif cmd == "run_stage":
            stageMem, stageProcs = self.pyro_proxy_for_server.getStageMem(i), self.pyro_proxy_for_server.getStageProcs(i)
            # we trust that the server has given us a stage
            # that we have enough memory and processors to run ...
            # reset the idle time, we are running a stage!
            self.idle_time = 0
            self.runningMem += stageMem
            self.runningProcs += stageProcs
            # The multiprocessing library must pickle things in order to execute them.
            # I wanted the following function (runStage) to be a function of the pipelineExecutor
            # class. That way we can access self.serverURI and self.clientURI from
            # within the function. However, bound methods are not picklable (a bound method
            # is a method that has "self" as its first argument, because if I understand 
            # this correctly, that binds the function to a class instance). There is
            # a way to make a bound function picklable, but this seems cumbersome. So instead
            # runStage is now a standalone function.
            result = self.pool.apply_async(runStage, (self.serverURI, self.clientURI, i, options.cmd_wrapper))

            self.runningChildren.append(ChildProcess(i, result, stageMem, stageProcs))
            logger.debug("Added stage %i to the running pool.", i)
            return True
        else:
            raise Exception("Got invalid cmd from server: %s" % cmd)
                


##########     ---     Start of program     ---     ##########   

if __name__ == "__main__":

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

    #Check to make sure some executors have been specified. 
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
            pe = pipelineExecutor(options=options)   #, pipeline_name=pipeline_name)
            pe.submitToQueue()
    else:
        local_launch(options)
