from pydpiper.execution.pipeline import CmdStage, InputFile, OutputFile

def convertCmdStage(cmd_stage):
    c = CmdStage([])
    c.inputFiles  = map(lambda x: x.get_path(), cmd_stage.inputs)
    c.outputFiles = map(lambda x: x.get_path(), cmd_stage.outputs)
    c.cmd = cmd_stage.to_array()
    c.mem = cmd_stage.memory
    c.runnable_hooks = cmd_stage.when_runnable_hooks
    c.finished_hooks = cmd_stage.when_finished_hooks
    return c
