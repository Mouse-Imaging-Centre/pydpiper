#!/usr/bin/env python

import Pyro
import time
import sys
import os
from optparse import OptionParser
from datetime import datetime
from multiprocessing import Process, Pool
from subprocess import call
import pydpiper.queueing as q
import traceback
import logging

logger = logging.getLogger(__name__)

POLLING_INTERVAL = 5 # poll for new jobs

Pyro.config.PYRO_MOBILE_CODE=1


#use Pyro.core.CallbackObjBase?? - need further review of documentation
class clientExecutor(Pyro.core.SynchronizedObjBase):
    def __init__(self):
        Pyro.core.SynchronizedObjBase.__init__(self)
        self.continueRunning =  True
    def continueLoop(self):
        return self.continueRunning
    def serverShutdownCall(self, serverShutdown):
        # receive call from server when all stages are processed
        if serverShutdown:
            self.continueRunning = False
         
def runStage(serverURI, clientURI, i):
    # Proc needs its own proxy as it's independent of executor
    p = Pyro.core.getProxyForURI(serverURI)
    s = p.getStage(i)
    
    # Run stage, set finished or failed accordingly  
    try:
        logger.info("Running stage %i: %s, ", i, str(s))
        p.setStageStarted(i, clientURI)
        try:
            r = s.execStage()
        except:
            logger.exception("Exception whilst running stage: %i ", i)   
            p.setStageFailed(i)
        else:
            logger.info("Stage %i finished, return was: %i", i, r)
            if r == 0:
                p.setStageFinished(i)
            else:
                p.setStageFailed(i)

        return [s.getMem(),s.getProcs()] # If completed, return mem & processes back for re-use    
    except:
        logger.exception("Error communicating to server. Stopping executor...")
        sys.exit()        
         
class pipelineExecutor():
    def __init__(self, options):
        #options cannot be null when used to instantiate pipelineExecutor
        self.mem = options.mem
        self.proc = options.proc
        self.queue = options.queue       
        self.ns = options.use_ns
        self.uri = options.urifile
        if self.uri==None:
            self.uri = os.path.abspath(os.curdir + "/" + "uri")
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
            jobname += "pipeline-executor-" + now.strftime("%Y%m%d-%H%M%S")
            # Add options for sge_batch command
            cmd = ["sge_batch", "-J", jobname, "-m", strprocs, "-l", strmem] 
            cmd += ["pipeline_executor.py", "--uri-file", self.uri, "--proc", strprocs, "--mem", str(self.mem)]
            print(cmd)
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
    def launchPipeline(self):  
        """Start executor that will run pipeline stages"""   
        # initialize pipeline_executor as both client and server
        print "Launching pipeline..."       
        Pyro.core.initClient()
        Pyro.core.initServer()
        daemon = Pyro.core.Daemon()
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
                sys.exit()

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
        # loop until the pipeline sets executor.continueLoop() to false
        pool = Pool(processes = self.proc)
        try:
            while executor.continueLoop():
                daemon.handleRequests(0)               
                # Free up resources from any completed (successful or otherwise) stages
                for child in [x for x in runningChildren if x.result.ready()]:
                    logger.debug("Freeing up resources for stage %i." % child.stage)
                    runningMem -= child.mem
                    runningProcs -= child.procs
                    runningChildren.remove(child)

                # check if we have any free processes, and even a little bit of memory
                if not self.canRun(1, 1, runningMem, runningProcs): 
                    time.sleep(POLLING_INTERVAL)
                    continue
                
                # check for available stages
                i = p.getRunnableStageIndex()                 
                if i == None:
                    logger.debug("No runnable stages. Sleeping...")
                    time.sleep(POLLING_INTERVAL)
                    continue

                # Before running stage, check usable mem & procs
                logger.debug("Considering stage %i" % i)
                s = p.getStage(i)
                stageMem, stageProcs = s.getMem(), s.getProcs()
                if self.canRun(stageMem, stageProcs, runningMem, runningProcs):
                    runningMem += stageMem
                    runningProcs += stageProcs            
                    result = pool.apply_async(runStage,(serverURI, clientURI, i))
                    runningChildren.append(ChildProcess(i, result, stageMem, stageProcs))
                    logger.debug("Added stage %i to the running pool." % i)
                else:
                    logger.debug("Not enough resources to run stage %i. " % i) 
                    p.requeue(i)
        except Exception as e:
            logger.exception("Error during executor polling loop. Shutting down executor...")
            daemon.shutdown(True)
            pool.close()
            pool.join()
            sys.exit()
        else:
            logger.info("Server has shutdown. Killing executor thread.")
            pool.close()
            pool.join()        
            daemon.shutdown(True)


##########     ---     Start of program     ---     ##########   

if __name__ == "__main__":
    FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    usage = "%prog [options]"
    description = "pipeline executor"

    # command line option handling    
    parser = OptionParser(usage=usage, description=description)
    
    parser.add_option("--uri-file", dest="urifile",
                      type="string", default=None,
                      help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    parser.add_option("--use-ns", dest="use_ns",
                      action="store_true",
                      help="Use the Pyro NameServer to store object locations")
    parser.add_option("--num-executors", dest="num_exec", 
                      type="int", default=1, 
                      help="Number of independent executors to launch.")
    parser.add_option("--time", dest="time", 
                      type="string", default="2:00:00:00", 
                      help="Wall time to request for each executor in the format dd:hh:mm:ss")
    parser.add_option("--proc", dest="proc", 
                      type="int", default=8,
                      help="Number of processes per executor. If not specified, default is 8. Also sets max value for processor use per executor.")
    parser.add_option("--mem", dest="mem", 
                      type="float", default=16,
                      help="Total amount of requested memory for all processes the executor runs. If not specified, default is 16G.")
    parser.add_option("--ppn", dest="ppn", 
                      type="int", default=8,
                      help="Number of processes per node. Default is 8. Used when --queue=pbs")
    parser.add_option("--queue", dest="queue", 
                      type="string", default=None,
                      help="Use specified queueing system to submit jobs. Default is None.")              
                      
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
        processes = [Process(target=pe.launchPipeline) for i in range(options.num_exec)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
