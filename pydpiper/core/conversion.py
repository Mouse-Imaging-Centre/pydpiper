import os

from pydpiper.execution.pipeline import CmdStage, InputFile, OutputFile

def convertCmdStage(cmd_stage): # CmdStage -> old CmdStage
    print "\nConverting a new stage to an old one"
    print type(cmd_stage)
    print cmd_stage.inputs
    print cmd_stage.outputs
    def f(s):
        for inp in map(lambda x: x.get_path(), cmd_stage.inputs):
            if inp in s:  
                return InputFile(inp)
        for inp in map(lambda x: x.get_path(), cmd_stage.outputs):
            if inp in s:  
                return OutputFile(inp)
        return s
    print cmd_stage.to_array()
    c = CmdStage(map(f, cmd_stage.to_array()))
    print c.inputFiles
    print c.outputFiles
    c.mem = cmd_stage.memory
    c.runnable_hooks = cmd_stage.when_runnable_hooks
    c.finished_hooks = cmd_stage.when_finished_hooks
    return c
