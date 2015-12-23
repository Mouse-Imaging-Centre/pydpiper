#!/usr/bin/env python

from __future__ import print_function
import time
import sys
import os
from configargparse import ArgParser  # type: ignore
from datetime import datetime
from multiprocessing import Process, Pool # type: ignore
import subprocess
import shlex
import pydpiper.execution.queueing as q
#import atoms_and_modules.registration_functions as rf
import logging
import socket
import signal
import threading
import Pyro4       # type: ignore
from typing import Any

Pyro4.config.SERVERTYPE = "multiplex"


#TODO add these to executorArgumentGroup as options, pass into pipelineExecutor
WAIT_TIMEOUT = 5.0
HEARTBEAT_INTERVAL = 10.0
#SHUTDOWN_TIME = WAIT_TIMEOUT + LATENCY_TOLERANCE

logger = logging # type: Any
#logger = logging.getLogger(__name__)

sys.excepthook = Pyro4.util.excepthook  # type: ignore

def addExecutorArgumentGroup(parser):
    group = parser.add_argument_group("Executor options",
                        "Options controlling how and where the code is run.")
    group.add_argument("--uri-file", dest="urifile",
                       type=str, default=None,
                       help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    group.add_argument("--use-ns", dest="use_ns",
                       action="store_true", default=False,
                       help="Use the Pyro NameServer to store object locations. Currently a Pyro nameserver must be started separately for this to work.")
    group.add_argument("--latency-tolerance", dest="latency_tolerance",
                       type=float, default=15.0,
                       help="Allowed grace period by which an executor may miss a heartbeat tick before being considered failed [Default = %(default)s.")
    group.add_argument("--num-executors", dest="num_exec", 
                       type=int, default=None, 
                       help="Number of independent executors to launch. [Default = %(default)s. Code will not run without an explicit number specified.]")
    group.add_argument("--max-failed-executors", dest="max_failed_executors",
                      type=int, default=2,
                      help="Maximum number of failed executors before we stop relaunching. [Default = %(default)s]")
    # TODO add corresponding --monitor-heartbeats
    group.add_argument("--no-monitor-heartbeats", dest="monitor_heartbeats",
                      action="store_false", default=True,
                      help="Don't assume executors have died if they don't check in with the server (NOTE: this can hang your pipeline if an executor crashes).")
    group.add_argument("--time", dest="time", 
                       type=str, default=None,
                       help="Wall time to request for each server/executor in the format hh:mm:ss. Required only if --queue-type=pbs. Current default on PBS is 48:00:00.")
    group.add_argument("--proc", dest="proc", 
                       type=int, default=1,
                       help="Number of processes per executor. Also sets max value for processor use per executor. [Default = %(default)s]")
    group.add_argument("--mem", dest="mem", 
                       type=float, default=6,
                       help="Total amount of requested memory (in GB) for all processes the executor runs. [Default = %(default)s].")
    group.add_argument("--pe", dest="pe",
                       type=str, default=None,
                       help="Name of the SGE pe, if any. [Default = %(default)s]")
    group.add_argument("--greedy", dest="greedy",
                       action="store_true", default=False,
                       help="Request the full amount of RAM specified by --mem rather than the (lesser) amount needed by runnable jobs.  Always use this if your executor is assigned a full node.") #TODO mutually exclusive --non-greedy
    group.add_argument("--ppn", dest="ppn", 
                       type=int, default=8,
                       help="Number of processes per node. Used when --queue-type=pbs. [Default = %(default)s].")
    group.add_argument("--queue-name", dest="queue_name", type=str, default=None,
                       help="Name of the queue, e.g., all.q (MICe) or batch (SciNet)")
    group.add_argument("--queue-type", dest="queue_type", type=str, default=None,
                       help="""Queue type to submit jobs, i.e., "sge" or "pbs".  [Default = %(default)s]""")
    group.add_argument("--queue", dest="queue", 
                       type=str, default=None,
                       help="[DEPRECATED; use --queue-type instead.]  Use specified queueing system to submit jobs. Default is None.")              
    group.add_argument("--sge-queue-opts", dest="sge_queue_opts", 
                       type=str, default=None,
                       help="[DEPRECATED; use --queue-name instead.]  For --queue=sge, allows you to specify different queues. [Default = %(default)s]")
    group.add_argument("--queue-opts", dest="queue_opts",
                       type=str, default="",
                       help="A string of extra arguments/flags to pass to qsub. [Default = '%(default)s']")
    group.add_argument("--executor-start-delay", dest="executor_start_delay", type=int, default=180,
                       help="Seconds before starting remote executors when running the server on the grid")
    group.add_argument("--time-to-seppuku", dest="time_to_seppuku", 
                       type=int, default=1,
                       help="The number of minutes an executor is allowed to continuously sleep, i.e. wait for an available job, while active on a compute node/farm before it kills itself due to resource hogging. [Default = %(default)s]")
    group.add_argument("--time-to-accept-jobs", dest="time_to_accept_jobs", 
                       type=int,
                       help="The number of minutes after which an executor will not accept new jobs anymore. This can be useful when running executors on a batch system where other (competing) jobs run for a limited amount of time. The executors can behave in a similar way by giving them a rough end time. [Default = %(default)s]")
    group.add_argument('--local', dest="local", action='store_true', default=False, help="Don't submit anything to any specified queueing system but instead run as a server/executor")
    group.add_argument("--config-file", type=str, metavar='config_file', is_config_file=True,
                       required=False, help='Config file location')
    group.add_argument("--prologue-file", type=str, metavar='file',
                       help="Location of a shell script to inline into PBS submit script to set paths, load modules, etc.")
    group.add_argument("--min-walltime", dest="min_walltime", type=int, default = 0,
            help="Min walltime (s) allowed by the queuing system [Default = %(default)s]")
    group.add_argument("--max-walltime", dest="max_walltime", type=int, default = None,
            help="Max walltime (s) allowed for jobs on the queuing system, or infinite if None [Default = %(default)s]")
    group.add_argument("--default-job-mem", dest="default_job_mem",
                       type=float, default = 1.75,
                       help="Memory (in GB) to allocate to jobs which don't make a request. [Default=%(default)s]")

def ensure_exec_specified(numExec):
    if not numExec:
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
        t = threading.Thread(target=daemon.requestLoop)
        t.daemon = True
        t.start()
        h = threading.Thread(target=executor.heartbeat)
        h.daemon = True
        h.start()
        executor.mainLoop()
    except KeyboardInterrupt:
        logger.exception("Caught keyboard interrupt. Shutting down executor...")
        executor.generalShutdownCall()
        #daemon.shutdown()
        sys.exit(0)
    except Exception:
        logger.exception("Error during executor loop. Shutting down executor...")
        executor.generalShutdownCall()
        #daemon.shutdown()
        sys.exit(0)
    else:
        executor.completeAndExitChildren()
        logger.info("Executor shutting down.")
        daemon.shutdown()
        t.join()

def runStage(serverURI, clientURI, i):
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
            command_to_run  = str(p.getStageCommand(i))
            logger.info(command_to_run)
            command_logfile = p.getStageLogfile(i)
            
            # log file for the stage
            of = open(os.path.join(p.get_log_dir(), command_logfile), 'a')
            of.write("Stage " + str(i) + " running on " + socket.gethostname() + " at " + datetime.isoformat(datetime.now(), " ") + ":\n")
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
        

        """
        This class is used for the actual commands that are run by the 
        executor. A child process is defined as a process that was 
        initiated by the executor
        """
class ChildProcess(object):
    def __init__(self, stage, result, mem, procs):
        self.stage = stage
        self.result = result
        self.mem = mem
        self.procs = procs 

class InsufficientResources(Exception):
    pass

class pipelineExecutor(object):
    def __init__(self, options, memNeeded = None):
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
        self.queue_type = options.queue_type
        self.queue_name = options.queue_name
        self.queue_opts = options.queue_opts
        self.ns = options.use_ns
        self.uri_file = options.urifile
        if self.uri_file is None:
            self.uri_file = os.path.abspath(os.path.join(os.curdir, options.pipeline_name + "_uri"))
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
        self.heartbeat_thread_crashed = False
        # we associate an event with each executor which is set when jobs complete.
        # in the future it might also be set by the server, and we might have more
        # than one event (for reclaiming, server messages, ...)
        self.e = threading.Event()
        
    def registeredWithServer(self):
        self.registered_with_server = True
        
    def addPIDtoRunningList(self, pid):
        self.current_running_job_pids.append(pid)
    
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
        
    def submitToQueue(self, programName=None):
        """Submits to sge queueing system using qsub""" 
        if self.queue_type == "sge":
            strprocs = str(self.procs)
            strmem = "vf=%sG" % float(self.mem)
            jobname = ""
            if programName is not None:
                executablePath = os.path.abspath(programName)
                jobname = os.path.basename(executablePath) + "-" 
            now = datetime.now().strftime("%Y-%m-%d-at-%H-%M-%S-%f")
            ident = "pipeline-executor-" + now
            jobname += ident
            queue_opts = ['-V', '-cwd', '-j', 'yes',
                          '-N', jobname,
                          '-l', strmem,
                          '-o', os.path.join(os.getcwd(),
                                             ident + '-eo.log')] \
                          + (['-q', self.queue_name]
                             if self.queue_name else []) \
                          + (['-pe', self.pe, strprocs]
                             if self.pe else []) \
                          + shlex.split(self.queue_opts)
            qsub_cmd = ['qsub'] + queue_opts
            cmd  = ["pipeline_executor.py", "--local"]
            cmd += ['--uri-file', self.uri_file]
            # Only one exec is launched at a time in this manner, so:
            cmd += ["--num-executors", str(1)]
            # pass most args to the executor
            cmd += q.remove_flags(['--num-exec', '--mem'], sys.argv[1:])
            cmd += ['--mem', str(self.mem)]
            script = "#!/usr/bin/env bash\n%s\n" % ' '.join(cmd)
            # FIXME huge hack -- shouldn't we just iterate over options,
            # possibly checking for membership in the executor option group?
            # The problem is that we can't easily check if an option is
            # available from a parser (but what about calling get_defaults and
            # looking at exceptions?).  However, one possibility is to
            # create a list of tuples consisting of the data with which to 
            # call parser.add_arguments and use this to check.
            # NOTE there's a problem with argparse's prefix matching which
            # also affects removal of --num-executors
            env = os.environ.copy()
            env['PYRO_LOGFILE'] = os.path.join(os.getcwd(), ident + ".log")
            p = subprocess.Popen(qsub_cmd, stdin=subprocess.PIPE, shell=False, env=env)
            p.communicate(script.encode('ascii'))
        else:
            logger.info("Specified queueing system is: %s" % (self.queue_type))
            logger.info("Only queue_type=sge or queue_type=None currently supports pipeline launching own executors.")
            logger.info("Exiting...")
            sys.exit()

    def canRun(self, stageMem, stageProcs, runningMem, runningProcs):
        """Calculates if stage is runnable based on memory and processor availibility"""
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
        try:
            tick = 0
            while self.registered_with_server:
                logger.debug("Heartbeat %d...", tick)
                tick += 1
                self.pyro_proxy_for_server.updateClientTimestamp(self.clientURI, tick)
                time.sleep(HEARTBEAT_INTERVAL)
        except:
            logger.exception("Heartbeat thread crashed: ")
            # this will take down the executor to avoid the case
            # where an executor wastes time processing jobs which the server
            # considers lost; there might be a better way to do this
            # (re-register/restart heartbeat and notify server of existing
            # jobs? quite complicated ...), and it could be done
            # 'atomically' using an event for better guarantees ...
            self.heartbeat_thread_crashed = True

    # use an event set/timeout system to run the executor mainLoop -
    # we might want to pass some extra information in addition to waking the system
    def mainLoop(self):
        while self.mainFn():
            self.e.wait(WAIT_TIMEOUT)
            self.e.clear()
        logger.debug("Main loop finished")

    def mainFn(self):
        """Try to get a job from the server (if appropriate) and update
        internal state accordingly.  Return True if it should be called
        again (i.e., there is more to do before shutting down),
        otherwise False."""

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

        if self.heartbeat_thread_crashed:
            logger.debug("Heartbeat thread crashed; quitting")
            return False

        if self.idle():
            self.idle_time += self.current_time - self.prev_time
            logger.debug("Current idle time: %d, and total seconds allowed: %d", self.idle_time, self.time_to_seppuku * 60)

        if self.is_seppuku_time():
            logger.warn("Exceeded allowed idle time... Seppuku!")
            return False

        # It is possible that the executor does not accept any new jobs
        # anymore. If that is the case, we can leave this main loop
        # and just wait until current running jobs (children) have finished
        if self.is_time_to_drain():
            logger.info("Time expired for accepting new jobs...leaving main loop.")
            return False

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
            result = self.pool.apply_async(runStage, (self.serverURI, self.clientURI, i))

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

    #rf.addGenRegArgumentGroup(parser) # just to get --pipeline-name
    addExecutorArgumentGroup(parser)

    # using parse_known_args instead of parse_args is a hack since we
    # currently send ALL arguments from the main program to the executor
    # on PBS queues (FIXME not yet true on SGE queues, but this is
    # not the best solution anyway).
    # Alternately, we could keep a copy of the executor parser around
    # when constructing the executor shell command
    options = parser.parse_known_args()[0]

    #Check to make sure some executors have been specified. 
    ensure_exec_specified(options.num_exec)

    def local_launch(options):
        pe = pipelineExecutor(options)
        # executors don't use any shared-memory constructs, so OK to copy
        ps = [Process(target=launchExecutor, args=(pe,))
              for _ in range(options.num_exec)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()

    if options.local:
        local_launch(options)
    elif options.queue_type == "pbs":
        roq = q.runOnQueueingSystem(options, sysArgs=sys.argv)
        for i in range(options.num_exec):
            roq.createAndSubmitExecutorJobFile(i, after=None,
                            time=q.timestr_to_secs(options.time))
    elif options.queue_type == "sge":
        for i in range(options.num_exec):
            pe = pipelineExecutor(options)
            pe.submitToQueue()
    else:
        local_launch(options)
