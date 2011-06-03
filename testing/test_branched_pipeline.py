#!/usr/bin/env python

from pydpiper.pipeline import *
import networkx as nx

def generateFile(i):
    return("filename_" + str(i) + ".mnc")
    
class TestBranchedPipeline():
    def setup_method(self, method):
        self.p = Pipeline()
        startFileA = generateFile(1)
        startFileB = generateFile(5)
        self.p.addStage(CmdStage(["headcommand-1", InputFile(generateFile(0)), OutputFile(startFileA)]))
        self.p.addStage(CmdStage(["subcommand-1-2", InputFile(startFileA), OutputFile(generateFile(2))]))
        self.p.addStage(CmdStage(["subcommand-1-3", InputFile(startFileA), OutputFile(generateFile(3))]))
        self.p.addStage(CmdStage(["headcommand-5", InputFile(generateFile(4)), OutputFile(startFileB)]))
        self.p.addStage(CmdStage(["subcommand-5-6", InputFile(startFileB), OutputFile(generateFile(6))]))
        self.p.addStage(CmdStage(["subcommand-5-7", InputFile(startFileB), OutputFile(generateFile(7))]))
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
        
    def test_stage_failure_one_branch(self):
        """make sure that if stage in one tree fails, unrelated stages can still run"""
        s = self.p.getRunnableStageIndex()
        assert s == 0
        self.p.setStageFailed(s)
        s = self.p.getRunnableStageIndex()
        assert s == 3
        assert self.p.continueLoop() == True
        
        

        
