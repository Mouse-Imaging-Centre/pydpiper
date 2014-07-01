#!/usr/bin/env python

import Pyro.core, Pyro.naming
import time
import sys
import os
from optparse import OptionGroup, OptionParser
from datetime import datetime
from multiprocessing import Process, Pool, Lock
from subprocess import call
from shlex import split
import pydpiper.queueing as q
import logging
import socket


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
                      type="int", default=0, 
                      help="Number of independent executors to launch. [Default = 0. Number must be explicitly stated]")
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

#use Pyro.core.CallbackObjBase?? - need further review of documentation
class clientExecutor(Pyro.core.SynchronizedObjBase):
    def __init__(self):
        Pyro.core.SynchronizedObjBase.__init__(self)
        self.continueRunning =  True
        self.mutex = Lock() 
    def continueLoop(self):
        self.mutex.acquire()
        return self.continueRunning
    def serverShutdownCall(self):
        # receive call from server when all stages are processed
        self.mutex.acquire()
        self.continueRunning = False
        self.mutex.release()
    def internalShutdownCall(self, clientURI,proxy_for_server):
        # This method is called when the executor shuts itself down. Currently
        # this happens either when time_to_seppuku is set and the executor has 
        # been idle for too long, or when time_to_accept_jobs is set and regardless
        # of whether there are jobs available the executor will shut down. 
        # When a shutdown like this happens, the executor should first unregister
        # with the server in order to keep it up to date with how many slaves it
        # has roaming around.
        proxy_for_server.unregister(clientURI)
        self.continueRunning = False
         
def runStage(serverURI, clientURI, i):
    # Proc needs its own proxy as it's independent of executor
    p = Pyro.core.getProxyForURI(serverURI)
    
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
                returncode = call(args, stdout=of, stderr=of, shell=False) 
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

        return [p.getStageMem(i),p.getStageProcs(i)] # If completed, return mem & processes back for re-use    
    except:
        logger.exception("Error communicating to server in runStage. " 
                         "Error raised to calling thread in launchExecutor. ")
        raise        
         
class pipelineExecutor():
    def __init__(self, options):
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
            # pass these along when submitting to the queue
            cmd += ["--time-to-seppuku", str(self.time_to_seppuku)]
            cmd += ["--time-to-accept-jobs", str(self.time_to_accept_jobs)]
            call(cmd)   
        else:
            print("Specified queueing system is: %s" % (self.queue))
            print("Only queue=sge or queue=None currently supports pipeline launching own executors.")
            print("Exiting...")
            sys.exit()
    def canRun(self, stageMem, stageProcs, runningMem, runningProcs):
        """Calculates if stage is runnable based on memory and processor availibility"""
        if ( (stageMem <= (self.mem-runningMem) ) and (stageProcs<=(self.proc-runningProcs)) ):
            return True
        else:
            return False
    def launchExecutor(self):  
        """Start executor that will run pipeline stages"""   
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
        if self.ns:
            ns = Pyro.naming.NameServerLocator().getNS()
            serverURI = ns.resolve("pipeline")
            daemon.useNameServer(ns)
        else:
            try:
                uf = open(self.uri)
                serverURI = Pyro.core.processStringURI(uf.readline())
                uf.close()
            except:
                print "Problem opening the specified uri file:", sys.exc_info()
                raise

        # instantiate the executor class and register the executor with the pipeline    
        executor = clientExecutor()
        clientURI=daemon.connect(executor,"executor")
        p = Pyro.core.getProxyForURI(serverURI)
        p.register(clientURI)
      
        #initialize runningMem and Procs
        runningMem = 0.0
        runningProcs = 0               

        class ChildProcess():
            def __init__(self, stage, result, mem, procs):
                self.stage = stage
                self.result = result
                self.mem = mem
                self.procs = procs                 
                                
        runningChildren = [] # no scissors
 
        print "Connected to ", serverURI
        print "Client URI is ", clientURI
        
        # connection time with the server
        self.connection_time_with_server = time.time()
        logger.info("Connected to the server at: %s" % datetime.isoformat(datetime.now(), " "))
        
        # loop until the pipeline sets executor.continueLoop() to false
        pool = Pool(processes = self.proc)
        try:
            while executor.continueLoop(): 
                executor.mutex.release()               
                daemon.handleRequests(0)   
                
                # first check whether it's time to perform seppuku: has the
                # idle_time exceeded the allowed time to be idle
                # time_to_seppuku is given in minutes
                # idle_time       is given in seconds
                if self.time_to_seppuku != None :
                    if (self.time_to_seppuku * 60) < self.idle_time :
                        logger.debug("Exceeded allowed idle time... Seppuku!")
                        executor.internalShutdownCall(clientURI,p)
                        continue
                
                # second check whether there is a limit to how long the executor
                # is allowed to accept jobs for. 
                if (self.time_to_accept_jobs != None) and (self.accept_jobs == 1):
                    current_time = time.time()
                    time_take_so_far = current_time - self.connection_time_with_server
                    minutes_so_far, seconds_so_far = divmod(time_take_so_far, 60)
                    if (self.time_to_accept_jobs < minutes_so_far):
                        logger.debug("Exceeded allowed time to accept jobs... not getting any new ones!")
                        self.accept_jobs = 0
                
                # Free up resources from any completed (successful or otherwise) stages
                for child in [x for x in runningChildren if x.result.ready()]:
                    logger.debug("Freeing up resources for stage %i." % child.stage)
                    runningMem -= child.mem
                    runningProcs -= child.procs
                    runningChildren.remove(child)
                
                # It is possible that the executor does not accept any new jobs
                # anymore. If that is the case, the executor should shut down as
                # soon as all current running jobs (children) have finished
                if (self.accept_jobs == 0) and (len(runningChildren) == 0):
                    # that's it, we're done. Nothing is running anymore
                    # and no jobs can be accepted anymore.
                    logger.debug("Now shutting down because not accepting new jobs and finished running jobs.")
                    executor.internalShutdownCall(clientURI,p)
                    continue
                
                # check if we have any free processes, and even a little bit of memory
                if not self.canRun(1, 1, runningMem, runningProcs): 
                    time.sleep(POLLING_INTERVAL)
                    continue
                
                # check for available stages
                i = p.getRunnableStageIndex()                 
                if i == None:
                    logger.debug("No runnable stages. Sleeping...")
                    time.sleep(POLLING_INTERVAL)
                    # increase the idle time by POLLING_INTERVAL
                    self.idle_time += POLLING_INTERVAL
                    continue

                # Before running stage, check usable mem & procs
                logger.debug("Considering stage %i" % i)
                stageMem, stageProcs = p.getStageMem(i), p.getStageProcs(i)
                if self.canRun(stageMem, stageProcs, runningMem, runningProcs):
                    runningMem += stageMem
                    runningProcs += stageProcs            
                    result = pool.apply_async(runStage,(serverURI, clientURI, i))
                    runningChildren.append(ChildProcess(i, result, stageMem, stageProcs))
                    logger.debug("Added stage %i to the running pool." % i)
                else:
                    logger.debug("Not enough resources to run stage %i. " % i) 
                    p.requeue(i)
        except Exception:
            logger.exception("Error during executor polling loop. Shutting down executor...")
            raise
        else:
            logger.info("Killing executor thread.")
        finally:
            """Acquires lock if it doesn't already have it, 
               releases lock either way"""
            executor.mutex.acquire(False)
            executor.mutex.release()
            pool.close()
            pool.join()        
            daemon.shutdown(True)


##########     ---     Start of program     ---     ##########   

if __name__ == "__main__":

    usage = "%prog [options]"
    description = "pipeline executor"

    # command line option handling    
    parser = OptionParser(usage=usage, description=description)
    
    addExecutorOptionGroup(parser)
                      
    (options,args) = parser.parse_args()

    pe = pipelineExecutor(options)
    if options.queue=="pbs":
        roq = q.runOnQueueingSystem(options)
        for i in range(options.num_exec):
            roq.createExecutorJobFile(i)
    elif options.queue=="sge":
        for i in range(options.num_exec):
            pe.submitToQueue()        
    else:
        processes = [Process(target=pe.launchExecutor) for i in range(options.num_exec)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
