#!/usr/bin/env python

from pydpiper.pipeline import *
import networkx as nx

def generateFile(i):
    return("filename_" + str(i) + ".mnc")
    
class TestSgeScript():
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
    def test_flatten_pipeline_simple(self):
        p = Pipeline()
        p.addStage(CmdStage(["command"]))
        p.initialize()
    
        assert flatten_pipeline(p) == [(0, "command", [])]
     
    def test_flatten_pipeline_branched(self):
        expected = [
    		(0, str(self.p.stages[0]), []),
    		(1, str(self.p.stages[1]), [0]),
    		(2, str(self.p.stages[2]), [0]),
    		(3, str(self.p.stages[3]), []),
    		(4, str(self.p.stages[4]), [3]),
    		(5, str(self.p.stages[5]), [3]),
    	]
        actual = flatten_pipeline(self.p)
    
        assert expected == actual

    def test_sge_script_simple(self):
        p = Pipeline()
        p.addStage(CmdStage(["command"]))
        p.initialize()

        assert sge_script(p) == [
                "qsub -h -N job_0 command",
                "qalter -h U job_0"]

    def test_sge_script_branched(self):
        assert sge_script(self.p) == [
                "qsub -h -N job_0 headcommand-1 filename_0.mnc filename_1.mnc",
                "qsub -h -N job_1 subcommand-1-2 filename_1.mnc filename_2.mnc",
                "qsub -h -N job_2 subcommand-1-3 filename_1.mnc filename_3.mnc",
                "qsub -h -N job_3 headcommand-5 filename_4.mnc filename_5.mnc",
                "qsub -h -N job_4 subcommand-5-6 filename_5.mnc filename_6.mnc",
                "qsub -h -N job_5 subcommand-5-7 filename_5.mnc filename_7.mnc",
                "qalter -hold_jid job_0 job_1",
                "qalter -hold_jid job_0 job_2",
                "qalter -hold_jid job_3 job_4",
                "qalter -hold_jid job_3 job_5",
                "qalter -h U job_0",
                "qalter -h U job_1",
                "qalter -h U job_2",
                "qalter -h U job_3",
                "qalter -h U job_4",
                "qalter -h U job_5",
                ]

