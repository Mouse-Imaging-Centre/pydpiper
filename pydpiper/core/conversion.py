from pydpiper.execution.pipeline import CmdStage, InputFile, OutputFile

def convertCmdStage(cmd_stage):
    c = CmdStage([])
    c.inputFiles  = [x.path for x in cmd_stage.inputs]
    c.outputFiles = [x.path for x in cmd_stage.outputs]
    c.cmd  = cmd_stage.to_array()
    c.mem  = cmd_stage.memory
    c.procs = cmd_stage.procs
    c.category = cmd_stage.category
    c.name = c.cmd[0]
    c.checkLogFile()
    c._runnable_hooks = cmd_stage.when_runnable_hooks
    c.finished_hooks = cmd_stage.when_finished_hooks
    c.logFile = cmd_stage.log_file
    c.env_vars = cmd_stage.env_vars
    return c
