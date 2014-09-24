#!/usr/bin/env python

import time
import sys
import os
from optparse import OptionGroup, OptionParser
from datetime import datetime
from multiprocessing import Process, Pool
import subprocess as subprocess
from shlex import split
import pydpiper.queueing as q
import logging
import socket
import signal
import threading
import Pyro4

POLLING_INTERVAL = 5

logger = logging.getLogger(__name__)

sys.excepthook = Pyro4.util.excepthook

def addExecutorOptionGroup(parser):
    group = OptionGroup(parser, "Executor options",
                        "Options controlling how and where the code is run.")
    group.add_option("--uri-file", dest="urifile",
                      type="string", default=None,
                      help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    group.add_option("--use-ns", dest="use_ns",
                      action="store_true",
                      help="Use the Pyro NameServer to store object locations")
    group.add_option("--num-executors", dest="num_exec", 
                      type="int", default=-1, 
                      help="Number of independent executors to launch. [Default = -1. Code will not run without an explicit number specified.]")
    group.add_option("--time", dest="time", 
                      type="string", default="2:00:00:00", 
                      help="Wall time to request for each executor in the format dd:hh:mm:ss. Required only if --queue=pbs.")
    group.add_option("--proc", dest="proc", 
                      type="int", default=1,
                      help="Number of processes per executor. If not specified, default is 8. Also sets max value for processor use per executor.")
    group.add_option("--mem", dest="mem", 
                      type="float", default=6,
                      help="Total amount of requested memory for all processes the executor runs. If not specified, default is 16G.")
    group.add_option("--ppn", dest="ppn", 
                      type="int", default=8,
                      help="Number of processes per node. Default is 8. Used when --queue=pbs")
    group.add_option("--queue", dest="queue", 
                      type="string", default=None,
                      help="Use specified queueing system to submit jobs. Default is None.")              
    group.add_option("--sge-queue-opts", dest="sge_queue_opts", 
                      type="string", default=None,
                      help="For --queue=sge, allows you to specify different queues. If not specified, default is used.")
    group.add_option("--time-to-seppuku", dest="time_to_seppuku", 
                      type="int", default=15,
                      help="The number of minutes an executor is allowed to continuously sleep, i.e. wait for an available job, while active on a compute node/farm before it kills itself due to resource hogging. [Default=15 minutes]")
    group.add_option("--time-to-accept-jobs", dest="time_to_accept_jobs", 
                      type="int", default=180,
                      help="The number of minutes after which an executor will not accept new jobs anymore. This can be useful when running executors on a batch system where other (competing) jobs run for a limited amount of time. The executors can behave in a similar way by given them a rough end time. [Default=3 hours]")
    parser.add_option_group(group)


def noExecSpecified(numExec):
    #Exit with helpful message if no executors are specified
    if numExec < 0:
        logger.info("You need to specify some executors for this pipeline to run. Please use the --num-executors command line option. Exiting...")
        sys.exit()


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
            uf = open(executor.uri)
            serverURI = uf.readline()
            uf.close()
        except:
            logger.exception("Problem opening the specified uri file:")
            raise

    p = Pyro4.Proxy(serverURI)
    # Register the executor with the pipeline
    # the following command only works if the server is alive. Currently if that's
    # not the case, the executor will die which is okay, but this should be
    # more properly handled: a more elegant check to verify the server is running
    p.registerClient(clientURI.asString())

    executor.registeredWithServer()
    executor.setClientURI(clientURI)
    executor.setServerURI(serverURI)
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
        executor.mainLoop()
    except KeyboardInterrupt:
        logger.exception("Caught keyboard interrupt. Shutting down executor...")
        executor.generalShutdownCall()
        daemon.shutdown()
        sys.exit(0)
    except Exception:
        logger.exception("Error during executor loop. Shutting down executor...")
        executor.generalShutdownCall()
        daemon.shutdown()
        sys.exit(0)
    else:
        daemon.shutdown()
        t.join()
        logger.info("Executor shutting down.")
        executor.completeAndExitChildren()

def runStage(serverURI, clientURI, i):
    ## Proc needs its own proxy as it's independent of executor
    p = Pyro4.core.Proxy(serverURI)
    client = Pyro4.core.Proxy(clientURI)
    
    # Retrieve stage information, run stage and set finished or failed accordingly  
    try:
        logger.info("Running stage %i: ", i)
        p.setStageStarted(i, clientURI)
        try:
            # get stage information
            command_to_run  = p.getStageCommand(i)
            logger.info(command_to_run)
            command_logfile = p.getStageLogfile(i)
            
            # log file for the stage
            of = open(command_logfile, 'w')
            of.write("Running on: " + socket.gethostname() + " at " + datetime.isoformat(datetime.now(), " ") + "\n")
            of.write(command_to_run + "\n")
            of.flush()
            
            # check whether the stage is completed already. If not, run stage command
            if p.getStage_is_effectively_complete(i):
                of.write("All output files exist. Skipping stage.\n")
                ret = 0
            else:
                args = split(command_to_run) 
                process = subprocess.Popen(args, stdout=of, stderr=of, shell=False)
                client.addPIDtoRunningList(process.pid)
                process.communicate()
                client.removePIDfromRunningList(process.pid)
                ret = process.returncode 
            of.close()
        except:
            logger.exception("Exception whilst running stage: %i ", i)   
            client.notifyStageTerminated(i)
        else:
            logger.info("Stage %i finished, return was: %i", i, ret)
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
class ChildProcess():
    def __init__(self, stage, result, mem, procs):
        self.stage = stage
        self.result = result
        self.mem = mem
        self.procs = procs 

class pipelineExecutor():
    def __init__(self, options):
        self.continueRunning =  True
        #options cannot be null when used to instantiate pipelineExecutor
        self.mem = options.mem
        self.proc = options.proc
        self.queue = options.queue   
        self.sge_queue_opts = options.sge_queue_opts    
        self.ns = options.use_ns
        self.uri = options.urifile
        if self.uri==None:
            self.uri = os.path.abspath(os.curdir + "/" + "uri")
        self.setLogger()
        # the next variable is used to keep track of how long the
        # executor has been continuously idle/sleeping for. Measured
        # in seconds
        self.idle_time = 0
        # the maximum number of minutes an executor can be continuously
        # idle for, before it has to kill itself.
        self.time_to_seppuku = options.time_to_seppuku
        # the time in minutes after which an executor will not accept new jobs
        self.time_to_accept_jobs = options.time_to_accept_jobs
        # stores the time of connection with the server
        self.connection_time_with_server = None
        self.accept_jobs = True
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
        
    def registeredWithServer(self):
        self.registered_with_server = True
        
    def addPIDtoRunningList(self, pid):
        self.current_running_job_pids.append(pid)
    
    def removePIDfromRunningList(self, pid):
        self.current_running_job_pids.remove(pid)

    def initializePool(self):
        self.pool = Pool(processes = self.proc)
        
    def setClientURI(self, cURI):
        self.clientURI = cURI 
            
    def setServerURI(self, sURI):
        self.serverURI = sURI
            
    def setProxyForServer(self, proxy):
        self.pyro_proxy_for_server = proxy
    
    def generalShutdownCall(self):
        # receive call from server when all stages are processed
        logger.debug("Executor shutting down ...")
        sys.stdout.flush()

        self.continueRunning = False
        # stop the worker processes (children) immediately without completing outstanding work
        # Initially I wanted to stop the running processes using pool.terminate() and pool.join()
        # but the keyboard interrupt handling proved tricky. Instead, the executor now keeps
        # track of the process IDs (pid) of the current running jobs. Those are targetted by
        # os.kill in order to stop the processes in the Pool
        for subprocID in self.current_running_job_pids:
            os.kill(subprocID, signal.SIGTERM)
        if self.registered_with_server:
            self.pyro_proxy_for_server.unregister(self.clientURI.asString())
            self.registered_with_server = False
        
    def completeAndExitChildren(self):
        # This function is called under normal circumstances (i.e., not because
        # of a keyboard interrupt). So we can close the pool of processes 
        # in the normal way (don't need to use the pids here)
        # prevent more jobs from starting, and exit
        if len(self.current_running_job_pids) > 0:
            self.pool.close()
            # wait for the worker processes (children) to exit (must be called after terminate() or close()
            self.pool.join()
        if self.registered_with_server:
            self.pyro_proxy_for_server.unregister(self.clientURI.asString())
            self.registered_with_server = False
    
    def setLogger(self):
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s %(process)d/%(threadName)s: %(message)s'
        now = datetime.now()  
        FILENAME = "pipeline_executor.py-" + now.strftime("%Y%m%d-%H%M%S%f") + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)
        
    def submitToQueue(self, programName=None):
        """Submits to sge queueing system using sge_batch script""" 
        if self.queue == "sge":
            strprocs = str(self.proc) 
            # NOTE: sge_batch multiplies vf value by # of processors. 
            # Since options.mem = total amount of memory needed, divide by self.proc to get value 
            memPerProc = float(self.mem)/float(self.proc)
            strmem = "vf=" + str(memPerProc) + "G" 
            jobname = ""
            if not programName==None: 
                executablePath = os.path.abspath(programName)
                jobname = os.path.basename(executablePath) + "-" 
            now = datetime.now()
            jobname += "pipeline-executor-" + now.strftime("%Y%m%d-%H%M%S%f")
            # Add options for sge_batch command
            cmd = ["sge_batch", "-J", jobname, "-m", strprocs, "-l", strmem] 
            if self.sge_queue_opts:
                cmd += ["-q", self.sge_queue_opts]
            cmd += ["pipeline_executor.py", "--proc", strprocs, "--mem", str(self.mem)]
            # TODO this is getting ugly ...
            # TODO also, what other opts aren't being passed on here?
            if self.ns:
                cmd += ["--use-ns"]
            # TODO this is/was breaking silently -
            # is something using this even in ns mode? (resolved?)
            #if self.uri:
            #    cmd += ["--uri-file", self.uri]
            cmd += ["--uri-file", self.uri]
            #Only one exec is launched at a time in this manner, so we assume --num-executors=1
            cmd += ["--num-executors", str(1)]  
            cmd += ["--time-to-seppuku", str(self.time_to_seppuku)]
            cmd += ["--time-to-accept-jobs", str(self.time_to_accept_jobs)]
            subprocess.call(cmd)   
        else:
            logger.info("Specified queueing system is: %s" % (self.queue))
            logger.info("Only queue=sge or queue=None currently supports pipeline launching own executors.")
            logger.info("Exiting...")
            sys.exit()

    def canRun(self, stageMem, stageProcs, runningMem, runningProcs):
        """Calculates if stage is runnable based on memory and processor availibility"""
        return stageMem <= self.mem - runningMem and stageProcs <= self.proc - runningProcs
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
        if (self.time_to_accept_jobs != None) and self.accept_jobs:
            current_time = time.time()
            time_take_so_far = current_time - self.connection_time_with_server
            minutes_so_far, seconds_so_far = divmod(time_take_so_far, 60)
            if (self.time_to_accept_jobs < minutes_so_far):
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

    def notifyStageTerminated(self, i, r=None):
        # hack - if we had the ChildProcess we could do better:
        #self.free_resources()
        if r == 0:
            self.pyro_proxy_for_server.setStageFinished(i)
        else:
            self.pyro_proxy_for_server.setStageFailed(i)


    def continueLoop(self):
        return self.continueRunning

    def mainLoop(self):
        #
        #
        # This is the executor's main function; decisions are made here about
        # what to do... (TODO flesh out info)
        #
        #
        while self.continueLoop():

            logger.debug("executor loop - memory used: %s, cores in use: %s", self.runningMem, self.runningProcs)
        
            # TODO this seems a bit coarse
            self.free_resources()

            # if we've been doing nothing since the last time through,
            # increase idle_time by that amount:
            if self.runningMem == 0 and self.runningProcs == 0:
                self.idle_time += POLLING_INTERVAL

            if self.is_seppuku_time():
                logger.debug("Exceeded allowed idle time... Seppuku!")
                self.continueRunning = False
                break

            if self.is_time_to_drain():
                logger.debug("Exceeded allowed time to accept jobs... not getting any new ones!")
                self.accept_jobs = False
        
            # It is possible that the executor does not accept any new jobs
            # anymore. If that is the case, the executor should shut down as
            # soon as all current running jobs (children) have finished
            if (self.accept_jobs == False) and (len(self.runningChildren) == 0):
                # that's it, we're done. Nothing is running anymore
                # and no jobs can be accepted anymore.
                logger.debug("Now shutting down because not accepting new jobs and finished running jobs.")
                self.continueRunning = False
                break

            # # check if we have any free processes, and even a little bit of memory
            # if not self.canRun(1, 1, self.runningMem, self.runningProcs): 
            #     # the executor's resources are all being used, can not accept any
            #     # new jobs at this moment. Sleep, but do not increase the idle time here
            #     time.sleep(POLLING_INTERVAL)
            #     logger.debug("No resources ... sleeping for %d s", POLLING_INTERVAL)
            #     return self.continueRunning
        
            # check for available stages
            # TODO we might want something more efficient, e.g., the client might
            # ask for a runnable stage within certain memory limits ...
            flag, i = self.pyro_proxy_for_server.getRunnableStageIndex()
            if flag == "done":
                self.continueRunning = False
                break
            elif flag == "wait":
                logger.debug("No runnable stages...waiting")
                logger.debug("Current idle time: %d, and total seconds allowed: %d", self.idle_time, self.time_to_seppuku * 60)
                time.sleep(POLLING_INTERVAL)
                continue
            elif flag == "index":
                # Before running stage, check usable mem & procs
                logger.debug("Considering stage %i", i)
                stageMem, stageProcs = self.pyro_proxy_for_server.getStageMem(i), self.pyro_proxy_for_server.getStageProcs(i)
                if self.canRun(stageMem, stageProcs, self.runningMem, self.runningProcs):
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
                else:
                    logger.debug("Not enough resources to run stage %i. ", i)
                    self.pyro_proxy_for_server.requeue(i)
                    # TODO does it make sense to sleep here? might change when smarter...
                    time.sleep(POLLING_INTERVAL)
            else:
                raise Exception("Got invalid flag from server: %s" % flag)
                
        logger.debug("main loop finished")


##########     ---     Start of program     ---     ##########   

if __name__ == "__main__":

    usage = "%prog [options]"
    description = "pipeline executor"

    # command line option handling    
    parser = OptionParser(usage=usage, description=description)
    
    addExecutorOptionGroup(parser)
                      
    (options,args) = parser.parse_args()

    #Check to make sure some executors have been specified. 
    noExecSpecified(options.num_exec)
    
    if options.queue=="pbs":
        roq = q.runOnQueueingSystem(options)
        for i in range(options.num_exec):
            roq.createExecutorJobFile(i)
    elif options.queue=="sge":
        for i in range(options.num_exec):
            pe = pipelineExecutor(options)
            pe.submitToQueue()        
    else:
        for i in range(options.num_exec):
            pe = pipelineExecutor(options)
            process = Process(target=launchExecutor, args=(pe,))
            process.start()
            process.join()
