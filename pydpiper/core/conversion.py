import os

from pydpiper.execution.pipeline import CmdStage, InputFile, OutputFile

def convertCmdStage(cmd_stage): # CmdStage -> old CmdStage
    def f(s):
        if s in map(lambda x: x.path, cmd_stage.inputs):
            return InputFile(s)
        elif s in map(lambda x: x.path, cmd_stage.outputs):
            return OutputFile(s)
        else:
            return s
    c = CmdStage(map(f, cmd_stage.to_array()))
    c.mem = cmd_stage.memory
    c.runnable_hooks = cmd_stage.when_runnable_hooks
    c.finished_hooks = cmd_stage.when_finished_hooks
    return c
