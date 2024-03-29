import concurrent.futures
from collections import defaultdict
import pkg_resources
import logging
import networkx as nx
import os
import sys
import shutil
import subprocess
import time

from typing import NamedTuple, List, Callable, Any

from pydpiper.core.stages import Result
from pydpiper.core.arguments import (CompoundParser, AnnotatedParser, application_parser,
                                     registration_parser, execution_parser, parse)
from pydpiper.execution.pipeline import Pipeline, pipelineDaemon, CmdStage, OutputFile
from pydpiper.execution.queueing import runOnQueueingSystem
from pydpiper.execution.pipeline_executor import ensure_exec_specified
from pydpiper.core.util import output_directories
from pydpiper.core.conversion import convertCmdStage
from pydpiper.minc.registration import can_read_MINC_file

PYDPIPER_VERSION = pkg_resources.get_distribution("pydpiper").version  # pylint: disable=E1101

logger = logging.getLogger(__name__)

ExecutionConf = NamedTuple('ExecutionConf', [('use_backup_files', bool),
                                             ('create_graph', bool),
                                             ('execute', bool)])
# TODO: put remainder of executor args here?


def output_dir(options):
    return options.application.output_directory if options.application.output_directory else os.getcwd()

def write_stages(stages, name):
    """
    writes all pipeline stages to a file
    """
    with open(os.path.join(os.curdir, "%s_pipeline_stages.txt" % name), 'w') as pf:
        for i, stage in enumerate(stages):
            # TODO indices of this enumeration only correspond to stage numbers by "coincidence"
            # (a similar iteration is performed elsewhere with the same results) but this is sort of silly/dangerous
            pf.write(str(i) + "\t" + str(stage.render()) + "\n")

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


def ensure_short_output_paths(stages, max_len=255):  # magic no. for EXT3, EXT4, NFS (?), Linux NAME_MAX, etc.
    # N.B. - at some point we had 245 instead of 255 -- a typo, a program-specific buffer size,
    # or something to do with one of the file systems (NFS, SciNet's IBM GPFS, etc. ...)?
    # TODO check the other parts of the path aren't too long either (much less likely)?
    for s in stages:
        for o in [o.filename_wo_ext for o in s.outputs] + [os.path.basename(s.log_file)]:
            if len(o) > max_len:
                raise ValueError("output filename '%s' of command '%s' too long (more than %s chars)" %
                                 (o, s.render(), max_len))


def ensure_output_paths_in_dir(stages, d):
    # TODO also check the logfiles ... should these be counted as stage outputs (tedious to add by hand ...)?
    not_in_dir = []
    for s in stages:
        for o in s.outputs:
            if os.path.relpath(o.path, d).startswith('..'):
                not_in_dir.append([o.path,s.cmd_to_string()])
    if (not_in_dir):
        # import pdb; pdb.set_trace()
        raise ValueError(["output %s of stage '%s' not contained inside pipeline directory %s"
                          % (item[0], item[1], d) for item in not_in_dir])


# TODO: where should this live - util?
# TODO: write some tests to check that this might be working
# TODO: could generalize to '(non)distinctOn' but nobody would ever use ...
def nondistinct_outputs(stages):
    """
    TODO: move this doctest to a proper test in test/
    >>> c1 = CmdStage(argArray=["touch", OutputFile("/tmp/foo.txt")])
    >>> c2 = CmdStage(argArray=[">",     OutputFile("/tmp/foo.txt")])
    >>> nondistinct_outputs([c1, c2]) == { '/tmp/foo.txt' : set([c1,c2]) }
    True
    """
    m = ((o, s) for s in stages for o in s.outputFiles)
    d = defaultdict(set)
    for o, s in m:
        d[o].add(s)
    bad_outputs = { o : ss for o, ss in d.items() if len(ss) > 1 }
    return bad_outputs


def ensure_distinct_outputs(stages):
    # TODO logfiles as well? (see comment in `ensure_output_paths_in_dir`)
    bad_outputs = nondistinct_outputs(stages)
    if len(bad_outputs) >= 1:
        print("Uh-oh - some files appear as outputs of multiple stages, to wit:", file=sys.stderr)
        for o, ss in bad_outputs.items():
            print("output: %s\nstages:\n" % o, file=sys.stderr)
            for s in ss:
                print("%s\n" % s, file=sys.stderr)
            print("\n", file=sys.stderr)
        raise ValueError("Conflicting outputs:", bad_outputs)


def ensure_commands_exist(stages, use_singularity, container_path, container_args):
    cmds = set((s.to_array()[0] for s in stages))
    if use_singularity:
        if not container_path:
            raise ValueError("container_path needed")
        bad_cmds = [cmd for cmd in cmds
                    if subprocess.run(['singularity', 'exec']
                                       + ([container_args] if container_args else [])
                                       + [container_path, 'which', cmd],
                                      capture_output=True).returncode != 0]
    else:
        bad_cmds = [cmd for cmd in cmds if shutil.which(cmd) is None]
    if len(bad_cmds) > 0:
        raise ValueError("Missing executables: %s" % bad_cmds)

def check_container_options(use_singularity, container_path):
    assert not (use_singularity and not container_path), "need --container-path=... with --use-singularity"


#TODO: change this to ...(static_pipeline, options)?
def execute(stages, options):
    """Basically just looks at the arguments and exits if `--no-execute` is specified,
    otherwise dispatches on backend type."""

    # if options.application.output_directory:
    #     os.chdir(options.application.output_directory)

    # TODO: logger.info('Constructing pipeline...')
    pipeline = Pipeline(stages=[convertCmdStage(s) for s in stages],
                        options=options)

    # TODO: print/log version
    reconstruct_command(options)

    write_stages(stages, options.application.pipeline_name)

    if options.application.create_graph:
        # TODO: these could have more descriptive names ...
        logger.debug("Writing dot file...")
        nx.drawing.nx_agraph.write_dot(pipeline.G, str(options.application.pipeline_name) + "_labeled-tree.dot")
        nx.drawing.nx_agraph.write_dot(file_graph(stages, options.application.output_directory),
                                       str(options.application.pipeline_name) + "_labeled-tree-alternate.dot")
        logger.debug("Done.")

    # TODO some of these checks could come even earlier
    check_container_options(use_singularity=options.execution.use_singularity,
                            container_path=options.execution.container_path)

    if options.execution.check_commands_exist:
        ensure_commands_exist(stages,
                              use_singularity=options.execution.use_singularity,
                              container_path=options.execution.container_path,
                              container_args=options.execution.container_args)

    # for debugging reasons, it's best if these come after writing stages, drawing graph, ...
    ensure_short_output_paths(stages)  # TODO convert to new `CmdStage`s
    ensure_output_paths_in_dir(stages, options.application.output_directory)
    ensure_distinct_outputs([convertCmdStage(s) for s in stages])


    def check_inputs():
        # TODO: probably inefficient to reconstruct inputs from graph here instead of once and for all ...
        # TODO: or check inputs to unfinished stages lying outside the unfinished set, instead of the 'overall' inputs!
        inputs = [ i for s in pipeline.G
                   for i in pipeline.stages[s].inputFiles
                   if pipeline.G.in_degree(s) == 0 ]

        def input_ok(input_file):
            # TODO: the `.endswith` call here is because in the old code the inputs/outputs are strings, not `Stage`s
            minc_ok = can_read_MINC_file(input_file) if input_file.endswith(".mnc") else True
            # TODO: check non-MINC files somehow!  (So far we usually don't encounter this case ...)
            return minc_ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # is 8 a good value?
            bad_inputs = [input_file for input_file, ok in zip(inputs, executor.map(input_ok, inputs)) if not ok]
            if len(bad_inputs) > 0:
                # TODO check that this properly quotes input files, e.g. making extra spaces visible ...
                raise ValueError("bad inputs: %s" % bad_inputs)

    # TODO lots of optimizations/improvements possible here, e.g., check only 'live' ancestors, not original ones
    if options.execution.check_input_files:
        check_inputs()

    # TODO: why is this needed now that --version is also handled automatically?
    # --num-executors=0 (<=> --no-execute) could be the default, and you could
    # also refuse to submit scripts for 0 executors ...

    # TODO: move calls to create_directories into execution functions
    #create_directories(stages)

    execution_proc = backend(options)
    execution_proc(pipeline, options)


def mk_application(parsers: List[AnnotatedParser], pipeline: Callable[[Any], Result[Any]]) -> Callable[[], Any]:
    """Wire up a pure-python pipeline application into a command-line application."""
    # TODO the type isn't very precise ...
    p = CompoundParser([application_parser,
                        registration_parser,
                        execution_parser
                        ] + parsers)
    def f():
        options = parse(p, sys.argv[1:])
        execute(pipeline(options).stages, options)
    return f


def backend(options):
    if options.execution.backend == 'makeflow':
        return makeflow

    if options.execution.submit_server and not options.execution.local:
        return grid_only_execute
    else:
        return pyro_execute

# TODO: should create_directories be added as a method to Pipeline?
def create_directories(stages):
    dirs = output_directories(stages)
    for d in dirs:
        # some files that are created by the pipeline end up in the
        # output directory which can be ''. In that case, one of the
        # directories returned in dirs is in fact '', but we should
        # not try to create it...
        if d != '':
            os.makedirs(d, exist_ok=True)

# The old AbstractApplication class has been removed due to its non-obvious API.  In its place,
# we currently provide an `execute` function and some helper functions for command-line parsing, as well
# as a `mk_application` function which inverts control again, although with a clearer interface (hopefully)
# than AbstractApplication.

def pyro_execute(pipeline, options):
    if not options.application.execute:
        print("Not executing the command (--no-execute is specified).\nDone.")
        return
    ensure_exec_specified(options.execution.num_exec)
    # FIXME this is a trivial function; inline pipelineDaemon here
    #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
    logger.info("Starting pipeline daemon...")
    # TODO: make a flag to disable this in case already created, wish to create later, etc.
    if not options.execution.defer_directory_creation:
        create_directories(pipeline.stages) # TODO: or whatever
    pipelineDaemon(pipeline, options, sys.argv[0])
    logger.info("Server has stopped.  Quitting...")


def generate_makeflow_command(options, outfile):
    return (["makeflow", "--jx", outfile] +
            ([f"-T{options.execution.queue_type}"] if options.execution.queue_type else []) +
            # TODO any way to pass queue_opts?  via `makeflow -B ...` somehow?
            (options.execution.makeflow_opts.split() if options.execution.makeflow_opts else []) +
            ([f"--singularity={options.execution.container_path}",
              f"--singularity-opt={options.execution.container_args}"]
             if options.execution.use_singularity else []) +
            (["--max-remote", str(options.execution.num_exec),
              "--max-local", str(options.execution.num_exec)] if options.execution.num_exec is not None else []) +
            (["--skip-file-check"] if not options.execution.check_input_files else []) +
            ([f"--wrapper={options.execution.cmd_wrapper}"] if options.execution.cmd_wrapper else []) +
            ([f"--disable-heartbeat"] if not options.execution.monitor_heartbeats else []) +
            ([f"--wait-for-files={options.execution.fs_delay}"]
             if options.execution.latency_tolerance else []))


def makeflow(pipeline, options):
    makeflow_file = generate_makeflow(pipeline, options)
    print(f"generated Makeflow file at {makeflow_file}")

    if options.application.execute:
        ensure_exec_specified(options.execution.num_exec)
        import subprocess

        cmd = generate_makeflow_command(options, outfile = makeflow_file)

        logger.info(" ".join(cmd))

        makeflow_stdout_file = f"{makeflow_file}.stdout"

        monitor_cmd = ["makeflow_monitor", f"{makeflow_file}.makeflowlog"]

        # note - we might want to install a sigint handler for all of Pydpiper,
        # but it's quite unclear how this would intereact with the Pyro backend
        # (and with potential library rather than application usage)

        monitor = subprocess.Popen(monitor_cmd)
        try:

          with open(makeflow_stdout_file, 'w') as f:
              res = subprocess.run(cmd, stdout = f, stderr=subprocess.STDOUT)

        except KeyboardInterrupt:
          rc = 130
        else:
          rc = res.returncode
          if rc == 0:
              print("Done.")
          else:
              print(f"makeflow exited with an error; see {makeflow_stdout_file} for details")
        finally:
          monitor.terminate()
          monitor.wait()
          sys.exit(rc)

def makeflow_filename(options):
    return f"{options.application.pipeline_name}_makeflow.jx"

def generate_makeflow(pipeline, options):
    import simplejson

    def remove_none_values(d):
        return { k : v for k, v in d.items() if v is not None }

    output_file = makeflow_filename(options)

    with open(options.application.pipeline_name + "_makeflow.jx", 'w') as f:
        f.write(simplejson.dumps(
          { "rules" :
            [
              remove_none_values({
                "command"     : str(stage),
                "inputs"      : stage.inputFiles + list({os.path.dirname(f) for f in stage.outputFiles}),
                "outputs"     : stage.outputFiles,
                "category"    : stage.category,
                "environment" : stage.env_vars if stage.env_vars != {} else None
              })
              for stage in pipeline.stages
            ] + [ { "command" : f"mkdir -p {d}", "inputs" : [], "outputs" : [d],
                    "local_job" : True, "category" : "mkdir" }
                  for d in list({ os.path.dirname(f) for s in pipeline.stages for f in s.outputFiles }) ],
            "categories" : {
                "default" : {
                    "resources" : { "memory" : int(1024 * options.execution.default_job_mem) }
                },
                "mkdir" : {
                    "resources" : { "memory" : 1, "wall-time" : 10}
                },
                "ANTS" : {
                    "resources" : { "memory" : int(1024 * 12), "wall-time" : 48 * 3600 }
                },
                "minctracc" : {
                    "resources" : { "memory" : int(1024 * 8), "wall-time" : 12 * 3600 }
                },
            }
          },
          indent = '  '
        ))

    return output_file


def grid_only_execute(pipeline, options):
    if options.execution.queue_type != 'pbs':
        raise ValueError("currently we only support submitting the server to PBS/Torque systems")
    roq = runOnQueueingSystem(options, sys.argv)
    roq.createAndSubmitPbsScripts()
    # TODO: make the local server create the directories (first time only) OR create them before submitting OR submit a separate stage?
    # NOTE we can't add a stage to the pipeline at this point since the pipeline doesn't support any sort of incremental recomputation ...
    logger.info("Finished submitting PBS job scripts...quitting")

def reconstruct_command(options):
    # TODO: also write down the environment, contents of config files
    reconstruct = ' '.join(sys.argv)
    logger.info("Command is: " + reconstruct)
    logger.info("Command version : " + PYDPIPER_VERSION)
    fileForCommandAndVersion = options.application.pipeline_name + "-command-and-version-" + time.strftime("%d-%m-%Y-at-%H-%M-%S") + ".sh"
    pf = open(fileForCommandAndVersion, "w")
    pf.write("#!/usr/bin/env bash\n")
    for p in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "PERL5LIB"]:
        pf.write("#export %s=%s\n" % (p, os.getenv(p)))
    pf.write("# Command version is: " + PYDPIPER_VERSION + "\n")
    pf.write("# Command was: \n")
    pf.write(reconstruct + '\n')
    pf.write("# options were: \n# %s" % options)
    pf.close()
 
