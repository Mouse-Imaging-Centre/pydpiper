#!/usr/bin/env python

import Pyro.core, Pyro.naming
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

logger = logging.getLogger(__name__)

POLLING_INTERVAL = 5 # poll for new jobs

Pyro.config.PYRO_MOBILE_CODE=1

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
    # initialize pipeline_executor as both client and server      
    Pyro.core.initClient()
    Pyro.core.initServer()
    # Due to changes in how the network address is resolved, the Daemon on Linux will basically use:
    #
    # import socket
    # socket.gethostbyname(socket.gethostname())
    #
    # depending on how your machine is set up, this could return localhost ("127...")
    # to avoid this from happening, provide the correct network address from the start:
    network_address = [(s.connect(('8.8.8.8', 80)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    daemon = Pyro.core.Daemon(host=network_address)
    
    # set up communication with server from the URI string
    if executor.ns:
        ns = Pyro.naming.NameServerLocator().getNS()
        serverURI = ns.resolve("pipeline")
        daemon.useNameServer(ns)
    else:
        try:
            uf = open(executor.uri)
            serverURI = Pyro.core.processStringURI(uf.readline())
            uf.close()
        except:
            print "Problem opening the specified uri file:", sys.exc_info()
            raise

    # instantiate the executor class and register the executor with the pipeline    
    clientURI=daemon.connect(executor,"executor")
    p = Pyro.core.getProxyForURI(serverURI)
    # the following command only works if the server is alive. Currently if that's
    # not the case, the executor will die which is okay, but this should be
    # more properly handled: a more elegant check to verify the server is running
    p.register(clientURI)
    
    executor.registeredWithServer()
    executor.setClientURI(clientURI)
    executor.setServerURI(serverURI)
    executor.setProxyForServer(p)
    
    logger.info("Connected to %s" % serverURI)
    logger.info("Client URI is %s" % clientURI)
    
    # connection time with the server
    executor.connection_time_with_server = time.time()
    logger.info("Connected to the server at: %s" % datetime.isoformat(datetime.now(), " "))
    
    executor.initializePool()
    
    try:
        daemon.requestLoop(executor.continueLoop)
    except KeyboardInterrupt:
        logger.exception("Caught keyboard interrupt. Shutting down executor...")
        executor.generalShutdownCall()
        daemon.shutdown(True)
        sys.exit(0)
    except Exception:
        logger.exception("Error during executor polling loop. Shutting down executor...")
        executor.generalShutdownCall()
        sys.exit(0)
    else:
        logger.info("Killing executor thread.")
    finally:      
        executor.completeAndExitChildren()
        daemon.shutdown(True)


def runStage(serverURI, clientURI, i):
    ## Proc needs its own proxy as it's independent of executor
    p = Pyro.core.getProxyForURI(serverURI)
    client = Pyro.core.getProxyForURI(clientURI)
    
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
                returncode = 0
            else:
                args = split(command_to_run) 
                process = subprocess.Popen(args, stdout=of, stderr=of, shell=False)
                client.addPIDtoRunningList(process.pid)
                process.communicate()
                client.removePIDfromRunningList(process.pid)
                returncode = process.returncode 
            of.close()
            r = returncode
        except:
            logger.exception("Exception whilst running stage: %i ", i)   
            p.setStageFailed(i)
        else:
            logger.info("Stage %i finished, return was: %i", i, r)
            if r == 0:
                p.setStageFinished(i)
            else:
                p.setStageFailed(i)

        # If completed, return mem & processes back for re-use
        return [p.getStageMem(i),p.getStageProcs(i)]     
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

class pipelineExecutor(Pyro.core.SynchronizedObjBase):
    def __init__(self, options):
        Pyro.core.SynchronizedObjBase.__init__(self)
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
        self.accept_jobs = 1
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
        self.continueRunning = False
        # stop the worker processes (children) immediately without completing outstanding work
        # Initially I wanted to stop the running processes using pool.terminate() and pool.join()
        # but the keyboard interrupt handling proved tricky. Instead, the executor now keeps
        # track of the process IDs (pid) of the current running jobs. Those are targetted by
        # os.kill in order to stop the processes in the Pool
        for subprocID in self.current_running_job_pids:
            os.kill(subprocID, signal.SIGTERM)
        if(self.registered_with_server == True):
            self.pyro_proxy_for_server.unregister(self.clientURI)
            self.registered_with_server = False
        
    def completeAndExitChildren(self):
        # This function is called under normal circumstances (i.e., not because
        # of a keyboard interrupt). So we can close the pool of processes 
        # in the normal way (don't need to use the pids here)
        # prevent more jobs from starting, and exit
        if(len(self.current_running_job_pids) > 0):
            self.pool.close()
            # wait for the worker processes (children) to exit (must be called after terminate() or close()
            self.pool.join()
        if(self.registered_with_server == True):
            self.pyro_proxy_for_server.unregister(self.clientURI)
            self.registered_with_server = False
    
    def setLogger(self):
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
        now = datetime.now()  
        FILENAME = "pipeline_executor.py-" + now.strftime("%Y%m%d-%H%M%S%f") + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)
        
    def submitToQueue(self, programName=None):
        """Submits to sge queueing system using sge_batch script""" 
        if self.queue=="sge":
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
            cmd += ["pipeline_executor.py", "--uri-file", self.uri, "--proc", strprocs, "--mem", str(self.mem)]
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
        if ( (stageMem <= (self.mem-runningMem) ) and (stageProcs<=(self.proc-runningProcs)) ):
            return True
        else:
            return False
            

    def is_seppuku_time(self):
        # Is it time to perform seppuku: has the
        # idle_time exceeded the allowed time to be idle
        # time_to_seppuku is given in minutes
        # idle_time       is given in seconds
        returnvalue = 0
        if self.time_to_seppuku != None :
            if (self.time_to_seppuku * 60) < self.idle_time :
                logger.debug("Exceeded allowed idle time... Seppuku!")
                self.continueRunning = False
                returnvalue = 1
        return returnvalue
                        
    def is_time_to_drain(self):
        # check whether there is a limit to how long the executor
        # is allowed to accept jobs for. 
        returnvalue = 0
        if (self.time_to_accept_jobs != None) and (self.accept_jobs == 1):
            current_time = time.time()
            time_take_so_far = current_time - self.connection_time_with_server
            minutes_so_far, seconds_so_far = divmod(time_take_so_far, 60)
            if (self.time_to_accept_jobs < minutes_so_far):
                returnvalue = 1
        return returnvalue
    
    def free_resources(self):
        # Free up resources from any completed (successful or otherwise) stages
        for child in [x for x in self.runningChildren if x.result.ready()]:
            logger.debug("Freeing up resources for stage %i." % child.stage)
            self.runningMem -= child.mem
            self.runningProcs -= child.procs
            self.runningChildren.remove(child)
    
    def continueLoop(self):
        #
        #
        # This is the executors main function, decisions are made here about
        # what to do... (TODO flesh out info)
        #
        #
        
        self.free_resources()
        
        if( self.is_seppuku_time() == 1 ):
            return self.continueRunning
        
        if( self.is_time_to_drain() == 1 ):
            logger.debug("Exceeded allowed time to accept jobs... not getting any new ones!")
            self.accept_jobs = 0
        
        # It is possible that the executor does not accept any new jobs
        # anymore. If that is the case, the executor should shut down as
        # soon as all current running jobs (children) have finished
        if (self.accept_jobs == 0) and (len(self.runningChildren) == 0):
            # that's it, we're done. Nothing is running anymore
            # and no jobs can be accepted anymore.
            logger.debug("Now shutting down because not accepting new jobs and finished running jobs.")
            self.continueRunning = False
            return self.continueRunning
        
        # check if we have any free processes, and even a little bit of memory
        if not self.canRun(1, 1, self.runningMem, self.runningProcs): 
            # the executor's resources are all being used, can not accept any
            # new jobs at this moment. Sleep, but do not increase the idle time here
            time.sleep(POLLING_INTERVAL)
            return self.continueRunning
        
        # check for available stages
        i = self.pyro_proxy_for_server.getRunnableStageIndex()                 
        if i == None:
            logger.debug("No runnable stages. Sleeping...")
            time.sleep(POLLING_INTERVAL)
            # increase the idle time by POLLING_INTERVAL
            self.idle_time += POLLING_INTERVAL
            logger.debug("Current idle time: %f, and total #seconds allowed: %f" % (self.idle_time, self.time_to_seppuku * 60))
            return self.continueRunning
        
        # Before running stage, check usable mem & procs
        logger.debug("Considering stage %i" % i)
        stageMem, stageProcs = self.pyro_proxy_for_server.getStageMem(i), self.pyro_proxy_for_server.getStageProcs(i)
        if self.canRun(stageMem, stageProcs, self.runningMem, self.runningProcs):
            # reset the idle time, we are running a stage!
            self.idle_time = 0
            self.runningMem += stageMem
            self.runningProcs += stageProcs
            # The multiprocessing library must pickle things in order to execute them.
            # I wanted the following function (runStage) to be a function of the pipelineExecutor
            # class. That way we can access self.serverURI and self.clientURI from
            # within the function. However, bound methods are not pickable (a bound method
            # is a method that has "self" as its first argument, because if I understand 
            # this correctly, that binds the function to a class instance). There is
            # a way to make a bound function picklable, but this seems cumbersome. So instead
            # runStage is now a standalone function.
            result = self.pool.apply_async(runStage,(self.serverURI, self.clientURI, i))
            self.runningChildren.append(ChildProcess(i, result, stageMem, stageProcs))
            logger.debug("Added stage %i to the running pool." % i)
        else:
            logger.debug("Not enough resources to run stage %i. " % i) 
            self.pyro_proxy_for_server.requeue(i)
        
        return self.continueRunning



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
