import os

from pydpiper.execution.pipeline import Pipeline, CmdStage, InputFile, OutputFile
from pydpiper.minc.files import MincAtom

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

def directories(stages):
    """Return a set of directories the pipeline must create.
    No need to consider stage inputs - any input already exists
    or is also the output of some previous stage."""
    # TODO what about logfiles/logdirs?
    # What if an output file IS a directory - should there be a special
    # way of tagging this?
    return set((os.path.dirname(o) for s in stages for o in s.outputs))

#def mk_directories(stages):
#    for d in directories(stages):
#        try:
#            os.makedirs(d)
#        except OSError:
#            pass #ugh

def pipeline(stages):
    p = Pipeline()
    for s in stages:
        p.add(s)
    return p
