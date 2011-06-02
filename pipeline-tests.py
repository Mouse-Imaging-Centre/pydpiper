#!/usr/bin/env python

from pydpiper.pipeline import *
import networkx as nx

def generateFile(i):
    return("filename_" + str(i) + ".mnc")

class TestSimplePipeline():
    def setup_method(self, method):
        self.p = Pipeline()
        startFile = generateFile(0)
        self.p.addStage(CmdStage(["somecommand", InputFile(startFile), OutputFile(generateFile(1))]))
        for i in range(2,100):
            self.p.addStage(CmdStage(["somecommand", InputFile(generateFile(i-1)), OutputFile(generateFile(i))]))
        self.p.initialize()
        nx.write_dot(self.p.G, "simple-test-pipeline.dot")

    def test_graph_head(self):
        """make sure that it finds the graph head correctly"""
        runnableStage = self.p.getRunnableStageIndex()
        assert runnableStage == 0

    def test_dependency(self):
        """make sure that random dependency is correct"""
        s = self.p.G.predecessors(10)
        print "S: " + str(s)
        assert s[0] == 9

    def test_set_runnable(self):
        """make sure that it correctly updates runnable class"""
        s = self.p.getRunnableStageIndex() # should be 0
        self.p.setStageFinished(s)
        s = self.p.getRunnableStageIndex()
        assert s == 1

    def test_stage_failure_pipe_finish(self):
        """make sure that pipeline finishes when stage fails and no successors can run"""
        for i in range(10):
            s = self.p.getRunnableStageIndex()
            self.p.setStageFinished(s)
        s = self.p.getRunnableStageIndex()
        self.p.setStageFailed(s)
        assert self.p.continueLoop() == False
        
    def test_stage_already_exists(self):
        """make sure that if a stage already exists it is not recreated"""
        assert self.p.addStage(CmdStage(["somecommand", InputFile(generateFile(15)), OutputFile(generateFile(16))])) == None
        
class TestBranchedPipeline():
    def setup_method(self, method):
        self.p = Pipeline()
        startFileA = generateFile(1)
        startFileB = generateFile(5)
        self.p.addStage(CmdStage(["headcommand", "0-1", InputFile(generateFile(0)), OutputFile(startFileA)]))
        self.p.addStage(CmdStage(["subcommand", "1-2", InputFile(startFileA), OutputFile(generateFile(2))]))
        self.p.addStage(CmdStage(["subcommand", "1-3", InputFile(startFileA), OutputFile(generateFile(3))]))
        self.p.addStage(CmdStage(["headcommand", "4-5", InputFile(generateFile(4)), OutputFile(startFileB)]))
        self.p.addStage(CmdStage(["subcommand", "5-6", InputFile(startFileB), OutputFile(generateFile(6))]))
        self.p.addStage(CmdStage(["subcommand", "5-7", InputFile(startFileB), OutputFile(generateFile(7))]))
        self.p.initialize()
        nx.write_dot(self.p.G, "branched-test-pipeline.dot")
        
    def test_graph_heads(self):
        """make sure that both graph heads can run without predecessors"""
        runnableStage = self.p.getRunnableStageIndex()
        assert runnableStage == 0
        self.p.setStageFinished(0)
        runnableStage = self.p.getRunnableStageIndex()
        assert runnableStage == 3

    def test_predecessors(self):
        """make sure that dependencies in both subtrees are correct"""
        s = self.p.G.predecessors(1)
        print "S: " + str(s)
        assert s[0] == 0
        s = self.p.G.predecessors(2)
        print "S: " + str(s)
        assert s[0] == 0
        s = self.p.G.predecessors(4)
        print "S: " + str(s)
        assert s[0] == 3
        s = self.p.G.predecessors(5)
        print "S: " + str(s)
        assert s[0] == 3

        
