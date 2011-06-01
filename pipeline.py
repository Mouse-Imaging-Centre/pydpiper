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
from datetime import datetime
from subprocess import call

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

class PipelineStage():
    def __init__(self):
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
        if os.path.isdir('backups') == False:
            print "Creating backup directory ..."
            os.mkdir('backups')
        pickle.dump(self.G, open('./backups/G.pkl', 'wb'))
        pickle.dump(self.nameArray, open('./backups/nameArray.pkl', 'wb'))
        pickle.dump(self.counter, open('./backups/counter.pkl', 'wb'))
        pickle.dump(self.outputhash, open('./backups/outputhash.pkl', 'wb'))
        pickle.dump(self.stagehash, open('./backups/stagehash.pkl', 'wb'))
        pickle.dump(self.processedStages, open('./backups/processedStages.pkl', 'wb'))
        print 'File closed'
        print '\n\nPipeline pickled.\n\n'
    def recycle(self):
        self.G = pickle.load(open('./backups/G.pkl', 'rb'))        
        self.nameArray = pickle.load(open('./backups/nameArray.pkl', 'rb'))
        self.counter = pickle.load(open('./backups/counter.pkl', 'rb'))
        self.outputhash = pickle.load(open('./backups/outputhash.pkl', 'rb'))
        self.stagehash = pickle.load(open('./backups/stagehash.pkl', 'rb'))
        self.processedStages = pickle.load(open('./backups/processedStagesgm.pkl', 'rb'))
        print '  Successfully reimported old data from backups.'
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
        if self.stages[index].isFinished() == True:
            canRun = False
        else:
            print("Checking if stage " + str(index) + " is runnable ...")
            for i in self.G.predecessors(index):
                line = "   Predecessor: Stage " + str(i)
                s = self.getStage(i)
                print(line + ", State: " + str(s.status))
                if s.isFinished() == False:
                    canRun = False
                print("Stage " + str(index) + " Runnable: " + str(canRun) + '\n\n')
        return(canRun)
    def setStageFinished(self, index):
        """given an index, sets corresponding stage to finished and adds successors to the runnable queue"""
        print("FINISHED: " + str(self.stages[index]))
        self.stages[index].setFinished()
        self.processedStages.append(index)
        self.selfPickle()

        for i in self.G.successors(index):
            if self.checkIfRunnable(i):
                self.runnable.put(i)
    def setStageFailed(self, index):
        self.processedStages.append(index)
        #for i in nx.dfs_successors(self.G, index).keys():
        #    self.processedStages.append(index)
        
    def initialize(self):
        """called once all stages have been added - computes dependencies and adds graph heads to runnable queue"""
        self.runnable = Queue.Queue()
        self.createEdges()
        self.computeGraphHeads()
    def continueLoop(self):
        # return 1 unless all stages are finished
        return(len(self.stages) > len(self.processedStages))

def pipelineDaemon(pipeline):
    
    #error checking for valid pipeline?

    #Note: NameServer must be started for this function to work properly

    Pyro.core.initServer()  
    daemon=Pyro.core.Daemon()
    ns=Pyro.naming.NameServerLocator().getNS()
    daemon.useNameServer(ns)
    uri=daemon.connect(pipeline,"pipeline")
 
    print("Daemon is running on port: " + str(daemon.port))
    print("The object's uri is: " + str(uri))
    
    try:
    	daemon.requestLoop(pipeline.continueLoop)
    except:
    	sys.exit("daemon.requestLoop did not complete properly. Pipeline may not have completed. Check logs and restart if needed.")
    else:
    	print("Pipeline completed. daemon unregistering objects and shutting down.")
    	daemon.shutdown(True)
    	print("Objects successfully unregistered and daemon shutdown.")

def pipelineNoNSDaemon(pipeline, urifile=None):

    # check for valid pipeline?     

    if urifile==None:
	urifile = os.curdir + "/" + "uri"    

    Pyro.core.initServer()
    daemon=Pyro.core.Daemon()
    uri=daemon.connect(pipeline, "pipeline")
    
    print("Daemon is running on port: " + str(daemon.port))
    print("The object's uri is: " + str(uri))

    uf = open(urifile, 'w')
    uf.write(str(uri))
    uf.close()

    try:
    	daemon.requestLoop(pipeline.continueLoop)
    except:
    	sys.exit("daemon.requestLoop did not complete properly. Pipeline may not have completed. Check logs and restart if needed.")
    else:
    	print("Pipeline completed. daemon unregistering objects and shutting down.")
    	daemon.shutdown(True)
    	print("Objects successfully unregistered and daemon shutdown.")
