import time
import pkg_resources
import logging
import networkx as nx
import sys
import os
from typing import NamedTuple, List, Callable, Any

from pydpiper.core.stages import Result
from pydpiper.core.arguments import (CompoundParser, AnnotatedParser, application_parser,
                                     registration_parser, execution_parser, parse)
from pydpiper.execution.pipeline import Pipeline, pipelineDaemon
from pydpiper.execution.queueing import runOnQueueingSystem
from pydpiper.execution.pipeline_executor import addExecutorArgumentGroup, ensure_exec_specified
from pydpiper.core.util import output_directories
from pydpiper.core.conversion import convertCmdStage

PYDPIPER_VERSION = pkg_resources.get_distribution("pydpiper").version  # pylint: disable=E1101

logger = logging.getLogger(__name__)

ExecutionOptions = NamedTuple('ExecutionOptions', [('use_backup_files', bool),
                                                   ('create_graph', bool),
                                                   ('execute', bool)])
# TODO: put remainder of executor args here?


def output_dir(options):
    return options.application.output_directory if options.application.output_directory else os.getcwd()

def write_stages(stages, name):
    """
    writes all pipeline stages to a file
    """
    fileForPrinting = os.path.abspath(os.curdir + "/" + str(name) + "_pipeline_stages.txt")
    pf = open(fileForPrinting, "w")
    for i, stage in enumerate(stages):
        pf.write(str(i) + "  " + str(stage.render()) + "\n")
    pf.close()

def file_graph(stages, pipeline_dir):
    # TODO remove pipeline_dir from node pathnames
    G = nx.DiGraph()
    #files = [f.path for s in stages for f in s.inputs]
    #G.add_nodes_from(files)
    # could filter out boring outputs here...
    for s in stages:
        for x in s.inputs:
            for y in s.outputs:
                G.add_edge(os.path.relpath(x.path, pipeline_dir),
                           os.path.relpath(y.path, pipeline_dir),
                           label=s.to_array()[0])
    return G
    # TODO: nx.write_dot doesn't show `cmd` attribute;
    # need to use something like nx.to_pydot to convert

def ensure_short_output_paths(stages, max_len=245): # magic no. for mincblur, &c.
    for s in stages:
        for o in s.outputs:
            if len(o.path) > max_len:
                raise ValueError("output filename '%s' too long (more than %s chars)" % (o.path, max_len))

def ensure_output_paths_in_dir(stages, d):
    for s in stages:
        for o in s.outputs:
            if os.path.relpath(o.path, d).startswith('..'):
                raise ValueError("output %s of stage %s not contained inside "
                                 "pipeline directory %s" % (o.path, s, d))

#TODO: change this to ...(static_pipeline, options)?
def execute(stages, options):
    """Basically just looks at the arguments and exits if `--no-execute` is specified,
    otherwise dispatch on backend type."""

    # TODO: logger.info('Constructing pipeline...')
    pipeline = Pipeline(stages=[convertCmdStage(s) for s in stages],
                        options=options)

    ensure_short_output_paths(stages)
    ensure_output_paths_in_dir(stages, options.application.output_directory)

    # TODO: print/log version
    reconstruct_command(options)

    write_stages(stages, options.application.pipeline_name)
    
    if options.application.create_graph:
        # TODO: these could have more descriptive names ...
        logger.debug("Writing dot file...")
        nx.write_dot(pipeline.G, str(options.application.pipeline_name) + "_labeled-tree.dot")
        nx.write_dot(file_graph(stages, options.application.output_directory), str(options.application.pipeline_name) + "_labeled-tree-alternate.dot")
        logger.debug("Done.")

    if not options.application.execute:
        print("Not executing the command (--no-execute is specified).\nDone.")
        return

    # TODO: why is this needed now that --version is also handled automatically?
    # --num-executors=0 (<=> --no-execute) could be the default, and you could
    # also refuse to submit scripts for 0 executors ...
    ensure_exec_specified(options.execution.num_exec)

    # TODO: move calls to create_directories into execution functions
    #create_directories(stages)
    
    execution_proc = backend(options)
    execution_proc(pipeline, options)


def mk_application(parsers: List[AnnotatedParser], pipeline: Callable[[Any], Result[Any]]) -> Callable[[], Any]:
    """Wire up a pure-python pipeline application into a command-line application."""
    # TODO the type isn't very precise ...
    p = CompoundParser([AnnotatedParser(parser=application_parser, namespace='application'),
                        AnnotatedParser(parser=registration_parser, namespace='registration'),
                        AnnotatedParser(parser=execution_parser, namespace='execution')
                        ] + parsers)
    def f():
        options = parse(p, sys.argv[1:])
        execute(pipeline(options).stages, options)
    return f


def backend(options):
    return normal_execute if options.execution.local else execution_backends[options.execution.queue_type]

# TODO: should create_directories be added as a method to Pipeline?
def create_directories(stages):
    dirs = output_directories(stages)
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# The old AbstractApplication class has been removed due to its non-obvious API.  In its place,
# we currently provide an `execute` function and some helper functions for command-line parsing.
# In the future, we could also provide higher-order functions which invert control again, although
# with a clearer interface than AbstractApplication.  This would be nice since the user wouldn't have
# to remember to add the executor option group themselves, for example, but would have to be done tastefully.

def normal_execute(pipeline, options):
    # FIXME this is a trivial function; inline pipelineDaemon here
    #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
    logger.info("Starting pipeline daemon...")
    # TODO: make a flag to disable this in case already created, wish to create later, etc.
    create_directories(pipeline.stages) # TODO: or whatever
    pipelineDaemon(pipeline, options, sys.argv[0])
    logger.info("Server has stopped.  Quitting...")

def grid_only_execute(pipeline, options):
    #    if pbs_submit:
    roq = runOnQueueingSystem(options, sys.argv)
    roq.createAndSubmitPbsScripts()
    # TODO: make the local server create the directories (first time only) OR create them before submitting OR submit a separate stage?
    # NOTE we can't add a stage to the pipeline at this point since the pipeline doesn't support any sort of incremental recomputation ...
    logger.info("Finished submitting PBS job scripts...quitting")

execution_backends = { None : normal_execute, 'sge' : normal_execute, 'pbs' : grid_only_execute }

def reconstruct_command(options):
    # TODO: also write down the environment, contents of config files
    reconstruct = ' '.join(sys.argv)
    logger.info("Command is: " + reconstruct)
    logger.info("Command version : " + PYDPIPER_VERSION)
    fileForCommandAndVersion = options.application.pipeline_name + "-command-and-version-" + time.strftime("%d-%m-%Y-at-%H-%m-%S") + ".sh"
    pf = open(fileForCommandAndVersion, "w")
    pf.write("#!/usr/bin/env bash\n")
    pf.write("# Command version is: " + PYDPIPER_VERSION + "\n")
    pf.write("# Command was: \n")
    pf.write(reconstruct + '\n')
    pf.close()
 
