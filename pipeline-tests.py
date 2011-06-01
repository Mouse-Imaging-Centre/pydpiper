#!/usr/bin/env python

from pipeline import *
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
        for i in range(10):
            s = self.p.getRunnableStageIndex()
            self.p.setStageFinished(s)
        s = self.p.getRunnableStageIndex()
        self.p.setStageFailed(s)
        assert self.p.continueLoop() == False
        

        
