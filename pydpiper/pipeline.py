#!/usr/bin/env python

import networkx as nx
import Pyro.core
import Pyro.naming
import Queue
import cPickle as pickle
import os
import shutil
import sys
import socket
import time
from datetime import datetime, date
from subprocess import call
from os.path import basename,isdir
from os import mkdir
from multiprocessing import Process, Event
import pipeline_executor as pe

Pyro.config.PYRO_MOBILE_CODE=1 

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

class FileHandling():
    def __init__(self):
        self.outFileName = []
    def removeFileExt(self, input):
        base, ext = os.path.splitext(input)
        return(basename(input).replace(str(ext), ""))
    def createSubDir(self, input_dir, subdir):
        # check for input_dir/subdir format?
        # at this point, assume all / properly accounted for
        _newdir = input_dir + "/" + subdir
        if not isdir(_newdir):
            mkdir(_newdir)
        return (_newdir)
    def createLogDir(self, input_dir):
        _logDir = self.createSubDir(input_dir, "log")
        return (_logDir)
    def createBaseName(self, input_dir, base):
        # assume all / in input_dir accounted for? or add checking
        return (input_dir + "/" + base)
    def createSubDirSubBase(self, input_dir, subdir, input_base):
        _subDir = self.createSubDir(input_dir, subdir)
        _subBase = self.createBaseName(_subDir, input_base)
        return (_subDir, _subBase)
    def createLogDirLogBase(self, input_dir, input_base):
        _logDir = self.createLogDir(input_dir)
        _logBase = self.createBaseName(_logDir, input_base)
        return (_logDir, _logBase)
    def createBackupDir(self, output):
        _backupDir = self.createSubDir(output, "backups")
        return(_backupDir)
    def createOutputFileName(self, argArray):
        self.outFileName = [] #clear out any arguments from previous call	
        for a in argArray:
            self.outFileName.append(str(a))
        return("".join(self.outFileName))
    def createOutputAndLogFiles(self, output_base, log_base, fileType, argArray=None):
        if argArray:
            outArray = [output_base, "_", "_".join(argArray), fileType]
            logArray = [log_base, "_", "_".join(argArray), ".log"] 
        else:
            outArray = [output_base, fileType]
            logArray = [log_base, ".log"]
        outFile = self.createOutputFileName(outArray)
        logFile = self.createOutputFileName(logArray)
        return (outFile, logFile)

class PipelineStage():
    def __init__(self):
        self.mem = 2 # default memory allotted per stage
        self.procs = 1 # default number of processors per stage
        self.inputFiles = [] # the input files for this stage
        self.outputFiles = [] # the output files for this stage
        self.logFile = None # each stage should have only one log file
        self.status = None
        self.name = ""
        self.colour = "black"

    def isFinished(self):
        if self.status == "finished":
            return True
        else:
            return False
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
        if self.inputFiles == other.inputFiles and self.outputFiles == other.outputFiles:
            return True
        else:
            return False
    def __ne__(self, other):
        if self.inputFiles == other.inputFiles and self.outputFiles == other.outputFiles:
            return False
        else:
            return True

class CmdStage(PipelineStage):
    def __init__(self, argArray):
        PipelineStage.__init__(self)
        self.argArray = argArray # the raw input array
        self.cmd = [] # the input array converted to strings
        self.parseArgs()
        self.checkLogFile()
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
    	of = open(self.logFile, 'w')
    	of.write("Running on: " + socket.gethostname() + " at " + datetime.isoformat(datetime.now(), " ") + "\n")
    	of.write(repr(self) + "\n")
    	of.flush()
        returncode = call(self.cmd, stdout=of, stderr=of)
        of.close()
        return(returncode)
    def getHash(self):
        return(hash(" ".join(self.cmd)))
    def __repr__(self):
        return(" ".join(self.cmd))

class Pipeline(Pyro.core.SynchronizedObjBase):
    def __init__(self):
        # initialize the remote objects bits
        Pyro.core.SynchronizedObjBase.__init__(self)
        # the core pipeline is stored in a directed graph. The graph is made
        # up of integer indices
        self.G = nx.DiGraph()
        # an array of the actual stages (PipelineStage objects)
        self.stages = []
        self.nameArray = []
        # a queue of the stages ready to be run - contains indices
        self.runnable = Queue.Queue()
        # the current stage counter
        self.counter = 0
        # hash to keep the output to stage association
        self.outputhash = {}
        # a hash per stage - computed from inputs and outputs or whole command
        self.stagehash = {}
        # an array containing the status per stage
        self.processedStages = []
        # location of backup files for restart if needed
        self.backupFileLocation = None
        # list of registered clients
        self.clients = []
    def addStage(self, stage):
        """adds a stage to the pipeline"""
        # check if stage already exists in pipeline - if so, don't bother

        # check if stage exists - stage uniqueness defined by in- and outputs
        # for base stages and entire command for CmdStages
        h = stage.getHash()
        if self.stagehash.has_key(h):
            pass #stage already exists - nothing to be done
        else: #stage doesn't exist - add it
            # add hash to the dict
            self.stagehash[h] = self.counter
            #self.statusArray[self.counter] = 'notstarted'
            # add the stage itself to the array of stages
            self.stages.append(stage)
            self.nameArray.append(stage.name)
            # add all outputs to the output dictionary
            for o in stage.outputFiles:
                self.outputhash[o] = self.counter
            # add the stage's index to the graph
            self.G.add_node(self.counter, label=stage.name,color=stage.colour)
            # increment the counter for the next stage
            self.counter += 1
    def selfPickle(self):
        if (self.backupFileLocation == None):
            self.setBackupFileLocation()
        pickle.dump(self.G, open(str(self.backupFileLocation) + '/G.pkl', 'wb'))
        pickle.dump(self.stages, open(str(self.backupFileLocation) + '/stages.pkl', 'wb'))
        pickle.dump(self.nameArray, open(str(self.backupFileLocation) + '/nameArray.pkl', 'wb'))
        pickle.dump(self.counter, open(str(self.backupFileLocation) + '/counter.pkl', 'wb'))
        pickle.dump(self.outputhash, open(str(self.backupFileLocation) + '/outputhash.pkl', 'wb'))
        pickle.dump(self.stagehash, open(str(self.backupFileLocation) + '/stagehash.pkl', 'wb'))
        pickle.dump(self.processedStages, open(str(self.backupFileLocation) + '/processedStages.pkl', 'wb'))
        print '\nPipeline pickled.\n'
    def restart(self):
        """Restarts the pipeline from previously pickled backup files."""
        if (self.backupFileLocation == None):
            self.setBackupFileLocation()
            print "Backup location not specified. Looking in the current directory."
        try:
            self.G = pickle.load(open(str(self.backupFileLocation) + '/G.pkl', 'rb'))
            self.stages = pickle.load(open(str(self.backupFileLocation) + '/stages.pkl', 'rb'))
            self.nameArray = pickle.load(open(str(self.backupFileLocation) + '/nameArray.pkl', 'rb'))
            self.counter = pickle.load(open(str(self.backupFileLocation) + '/counter.pkl', 'rb'))
            self.outputhash = pickle.load(open(str(self.backupFileLocation) + '/outputhash.pkl', 'rb'))
            self.stagehash = pickle.load(open(str(self.backupFileLocation) + '/stagehash.pkl', 'rb'))
            self.processedStages = pickle.load(open(str(self.backupFileLocation) + '/processedStages.pkl', 'rb'))
            print 'Successfully reimported old data from backups.'
        except:
            sys.exit("Backup files are not recoverable.  Pipeline restart required.\n")
        print 'Previously completed stages (of ' + str(len(self.stages)) + ' total): '
        done = []
        for i in self.G.nodes_iter():
            if self.stages[i].isFinished() == True:
                done.append(i)
        print str(done)
        self.initialize()
        self.printStages()
    def setBackupFileLocation(self, outputDir=None):
        fh = FileHandling()
        if (outputDir == None):
            # set backups in current directory if directory doesn't currently exist
            outputDir = os.getcwd() 
        self.backupFileLocation = fh.createBackupDir(outputDir)   
    def addPipeline(self, p):
        for s in p.stages:
            self.addStage(s)
    def printStages(self):
        for i in range(len(self.stages)):
            print(str(i) + "  " + str(self.stages[i]))           
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
        print("Create Edges time: " + str(endtime-starttime))
    def computeGraphHeads(self):
        """adds stages with no incomplete predecessors to the runnable queue"""
        graphHeads = []
        for i in self.G.nodes_iter():
            if self.stages[i].isFinished() == False:
                """ either it has 0 predecessors """
                if len(self.G.predecessors(i)) == 0:
                    self.runnable.put(i)
                """ or all of its predecessors are finished """
                if len(self.G.predecessors(i)) != 0:
                    predfinished = True
                    for j in self.G.predecessors(i):
                        if self.stages[j].isFinished() == False:
                            predfinished = False
                    if predfinished == True:
                        self.runnable.put(i) 
                        graphHeads.append(i)
        print "Graph heads: " + str(graphHeads) + "\n"              
    def getStage(self, i):
        """given an index, return the actual pipelineStage object"""
        return(self.stages[i])
    def getRunnableStageIndex(self):
        """returns the next runnable stage, or None"""
        if self.runnable.empty():
            return None
        else:
            index = self.runnable.get()
            self.stages[index].setRunning()
            return(index)
    def checkIfRunnable(self, index):
        """stage added to runnable queue if all predecessors finished"""
        canRun = True
        print("Checking if stage " + str(index) + " is runnable ...")
        if self.stages[index].isFinished() == True:
            canRun = False
        else:
            for i in self.G.predecessors(index):
                print "Predecessor: Stage " + str(i),               
                s = self.getStage(i)
                print " State: " + str(s.status) 
                if s.isFinished() == False:
                    canRun = False
        print("Stage " + str(index) + " Runnable: " + str(canRun) + '\n')
        return(canRun)
    def setStageFinished(self, index):
        """given an index, sets corresponding stage to finished and adds successors to the runnable queue"""
        print("FINISHED STAGE " + str(index) + ": " + str(self.stages[index]))
        self.stages[index].setFinished()
        self.processedStages.append(index)
        self.selfPickle()
        for i in self.G.successors(index):
            if self.checkIfRunnable(i):
                self.runnable.put(i)
    def setStageFailed(self, index):
        self.stages[index].setFailed()
        self.processedStages.append(index)
        for i in nx.dfs_successors(self.G, index).keys():
            self.processedStages.append(index)
    def requeue(self, i):
        # when executors return a stage they can't handle at the moment, put it back on the queue
        self.stages[i].setNone()
        self.runnable.put(i)            
    def initialize(self):
        """called once all stages have been added - computes dependencies and adds graph heads to runnable queue"""
        self.runnable = Queue.Queue()
        self.createEdges()
        self.computeGraphHeads()
    def continueLoop(self):
        # return 1 unless all stages are finished
        return(len(self.stages) > len(self.processedStages))
    def getProcessedStageCount(self):
        return(len(self.processedStages))
    def register(self, client):
        print "CLIENT REGISTERED: " + str(client)
        self.clients.append(client)

def launchPipelineExecutor(options):
    pipelineExecutor = pe.pipelineExecutor()
    # Need to put in memory mgmt here as well. 
    if options.queue=="sge":
        strprocs = str(options.proc) 
        # NOTE: May want to change definition of options.mem, as sge_batch multiplies this by # procs. 
        # Currently designed as total amount of memory needed. May want this to be mem/process or just divide. 
        strmem = "vf=" + str(options.mem) + "G"  
        jobname = "pipeline-" + str(date.today())
        # Add options for sge_batch command
        cmd = ["sge_batch", "-J", jobname, "-m", strprocs, "-l", strmem] 
        # Next line is for development and testing only -- will be removed in final checked in version
        cmd += ["-q", "defdev.q"]
        cmd += ["pipeline_executor.py", "--uri-file", options.urifile, "--proc", strprocs]
        print cmd
        call(cmd)   
    elif options.queue=="scinet":
        print "Specified queueing system == scinet"
    else: 
        #options.queue==None. No queueing system specified, jobs will run locally. 
        pipelineExecutor.launchPipeline(options) 
    
def launchServer(pipeline, options, e):
    Pyro.core.initServer()
    daemon=Pyro.core.Daemon()
    
    #Note: Pyro NameServer must be started for this function to work properly
    if options.use_ns:
        ns=Pyro.naming.NameServerLocator().getNS()
        daemon.useNameServer(ns)
    
    uri=daemon.connect(pipeline, "pipeline")
    print("Daemon is running on port: " + str(daemon.port))
    print("The object's uri is: " + str(uri))

    # If not using Pyro NameServer, must write uri to file for reading by client.
    if not options.use_ns:
        uf = open(options.urifile, 'w')
        uf.write(str(uri))
        uf.close()
    
    e.set()
    
    try:
        daemon.requestLoop(pipeline.continueLoop)
    except:
    	print "Failed in pipelineDaemon"
    	print "Unexpected error: ", sys.exc_info()
    	sys.exit()
    else:
    	print("Pipeline completed. Daemon unregistering " + str(len(pipeline.clients)) + " client(s) and shutting down...\n")
    	for c in pipeline.clients[:]:
    	    clientObj = Pyro.core.getProxyForURI(c)
    	    clientObj.serverShutdownCall(True)
    	    print "Made serverShutdownCall to: " + str(c)
    	    pipeline.clients.remove(c)
    	    print "Client deregistered from server: "  + str(c)
    	daemon.shutdown(True)
    	print("Objects successfully unregistered and daemon shutdown.")

def pipelineDaemon(pipeline, returnEvent, options=None):

    #check for valid pipeline 
    if pipeline.runnable.empty()==None:
        print "Pipeline has no runnable stages. Exiting..."
        sys.exit()

    if options.urifile==None:
        options.urifile = os.curdir + "/" + "uri"    
    
    e = Event()
    process = Process(target=launchServer, args=(pipeline,options,e,))
    process.start()
    e.wait()
    if options.num_exec != 0:
        processes = [Process(target=launchPipelineExecutor, args=(options,)) for i in range(options.num_exec)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    
    returnEvent.set()
