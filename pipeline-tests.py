#!/usr/bin/env python

from pipeline import *

def generateFile(i):
    return("filename_" + str(i) + ".mnc")

def test_simple_pipeline():
    p = Pipeline()
    startFile = generateFile(0)
    p.addStage(CmdStage(["somecommand", startFile, generateFile(1)]))
    for i in range(2,100):
        p.addStage(CmdStage(["somecommand", generateFile(i-1), generateFile(i)]))
    p.initialize()
    runnableStage = p.getRunnableStageIndex()
    assert runnableStage == 0
