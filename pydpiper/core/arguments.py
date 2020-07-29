"""
TODO write some useful documentation
"""
from collections import defaultdict

from pkg_resources import get_distribution  # type: ignore
import warnings
import copy
import os
import time
from configargparse import ArgParser, Namespace  # type: ignore
from typing import Any, Callable, List, Optional
from pydpiper.core.util import AutoEnum, NamedTuple


# TODO: should the pipeline-specific argument handling be located here
# or in that pipeline's module?  Makes more sense (general stuff
# can still go here)
from pydpiper.minc.registration import InputSpace, to_lsq6_conf, RegistrationConf, LSQ12Conf


class LSQ6Method(AutoEnum):
    lsq6_simple = ()
    lsq6_centre_estimation = ()
    lsq6_large_rotations = ()


class PydParser(ArgParser):
    # Some sneakiness... override the format_epilog method
    # to return the epilog verbatim.
    # That way in the help message you can include an example
    # of what an lsq6/nlin protocol should look like, and this
    # won't be mangled when the parser calls `format_epilog`
    # when writing its help.
    def format_epilog(self, _formatter):
        if not self.epilog:
            self.epilog = ""
        return self.epilog


# TODO delete/move to util?
# def id(*args, **kwargs):
#    return args, kwargs

def parse_nullable_int(string: str) -> int:
    if string == "None":
        return None
    else:
        return int(string)


class Parser(object): pass


class AnnotatedParser(object):
    def __init__(self,
                 parser: Parser,
                 namespace: str,
                 # this causes some problems if not supplied - you get '---stuff-like-this'
                 # FIXME currently checking for None avoids this situation, but "" still produces it -
                 # checking for the latter might be better
                 prefix: str = None,  # = "",
                 cast: Callable[[Any], Any] = None) -> None:  # TODO: remove Any
        self.parser = parser  # type: Parser
        self.prefix = prefix  # type: str
        self.namespace = namespace  # type: str
        self.cast = cast      # type: Any


# the leaves of the parse object (these contain the arguments you're interested in)
class BaseParser(Parser):
    def __init__(self, argparser: ArgParser, group_name: str) -> None:
        self.argparser = argparser  # type: ArgParser
        self.group_name = group_name  # type: str


# the internal nodes of the parse object
class CompoundParser(Parser):
    def __init__(self, annotated_parsers: List[AnnotatedParser]) -> None:
        """
        annotated_parsers is a list that can hold
        both BaseParser-s and CompoundParser-s.
        """
        self.parsers = annotated_parsers  # type: List[AnnotatedParser]


# Parser = BaseParser ArgParser | CompoundParser([Annotated Parser]) - rose tree with elts at leaves instead of nodes?
# for more flexibility, you could also add an extra parser at the node, but that doesn't seem to be needed
# (you can always add an extra leaf at that node) (~isomorphic?)

# TODO: What about the situation when you want cross-cutting?  That is, instead of the usual situation
# (everything grows like a tree with more and more prefixes), you have components at very far-apart
# locations in the tree which you wish to control via a single set of options?  For instance,
# what if you have two twolevel models within a pipeline and want to set the second-level LSQ12 on both
# via --second-level-lsq12-max-pairs?  Within one twolevel pipeline, you can do this easily.
# I guess you could add a lsq12 parser in the code calling the two pipelines and use it as a default,
# but this wouldn't happen automagically.  You'd also have a problem with knowing which fields were user-specified
# and which were defaults, although configargparse's `print_values` should help with that.

# TODO tighten up the type somehow by adding type variables to a Parser ...?
# know problems with this setup:
#   - some features don't work, e.g., mutually exclusive groups (arguments are added but the group itself is erased)
#   - will break on arguments not starting with '--'
#   - FIXME reports too many things as positional args ('files') for the parsers that support this
#     (possible solution: get positional args from overall parse somehow ... how to know which parser this is from
#      and enforce that only a single parser accepts such args?)
def parse(parser: Parser, args: List[str]) -> Namespace:
    # TODO: accepting a comma-separated list might allow more flexibility
    default_config_file = os.getenv("PYDPIPER_CONFIG_FILE")
    if default_config_file is not None:
      try:
        with open(default_config_file) as _:
          pass
      except:
        warnings.warn(f"PYDPIPER_CONFIG_FILE is set to '{default_config_file}', which can't be opened.")
    config_files = [default_config_file] if default_config_file else []

    # First, build a parser that's aware of all options
    # (will be used for help/version/error messages).
    # This must be tried _before_ the partial parsing attempts
    # in order to get correct help/version messages.

    main_parser = ArgParser(default_config_files=config_files)

    # TODO: abstract out the recursive travels in go_1 and go_2 into a `walk` function
    def go_1(p, current_prefix):
        if isinstance(p, BaseParser):
            g = main_parser.add_argument_group(p.group_name)
            for a in p.argparser._actions:
                new_a = copy.copy(a)
                ss = copy.deepcopy(new_a.option_strings)
                for ix, s in enumerate(new_a.option_strings):
                    if s.startswith("--"):
                        ss[ix] = "-" + current_prefix + "-" + s[2:]  # "" was "-"
                    else:
                        raise NotImplementedError(
                            "sorry, I only understand flags starting with `--` at the moment, but got %s" % s)
                new_a.option_strings = ss
                g._add_action(new_a)
        elif isinstance(p, CompoundParser):
            for q in p.parsers:
                go_1(q.parser, current_prefix + (('-' + q.prefix) if q.prefix is not None else ''))
        else:
            raise TypeError(
                "parser %s wasn't a %s (%s or %s) but a %s" % (p, Parser, BaseParser, CompoundParser, p.__class__))

    go_1(parser, "")

    # Use this parser to exit with a helpful message if parse fails or --help/--version specified:
    main_parser.parse_args(args)

    # Now, use parse_known_args for each parser in the tree of parsers to fill the appropriate namespace object ...
    def go_2(p, current_prefix, current_ns):
        if isinstance(p, BaseParser):
            new_p = ArgParser(default_config_files=config_files)
            for a in p.argparser._actions:
                new_a = copy.copy(a)
                ss = copy.deepcopy(new_a.option_strings)
                for ix, s in enumerate(new_a.option_strings):
                    if s.startswith("--"):
                        ss[ix] = "-" + current_prefix + "-" + s[2:]
                    else:
                        raise NotImplementedError
                    new_a.option_strings = ss
                new_p._add_action(new_a)
            _used_args, _rest = new_p.parse_known_args(args, namespace=current_ns)
            # add a "_flags" field to each object so we know what flags caused a certain option to be set:
            # (however, note that post-parsing we may munge around ...)
            flags_dict = defaultdict(set)
            for action in new_p._actions:
                for opt in action.option_strings:
                    flags_dict[action.dest].add(opt)
            current_ns.flags_ = Namespace(**flags_dict)
            # TODO: could continue parsing from `_rest` instead of original `args`
        elif isinstance(p, CompoundParser):
            current_ns.flags_ = set()  # could also check for the CompoundParser case and not set flags there,
                                       # since there will never be any
            for q in p.parsers:
                ns = Namespace()
                if q.namespace in current_ns.__dict__:
                    raise ValueError("Namespace field '%s' already in use" % q.namespace)
                    # TODO could also allow, say, a None
                else:
                    # gross but how to write n-ary identity fn that behaves sensibly on single arg??
                    current_ns.__dict__[q.namespace] = ns
                    # FIXME this casting doesn't work for configurations with positional arguments,
                    # which aren't unpacked correctly -- better to use a namedtuple
                    # (making all arguments keyword-only also works, but then you have to supply
                    # often meaningless defaults in the __init__)
                go_2(q.parser, current_prefix=current_prefix + (('-' + q.prefix) if q.prefix is not None else ''),
                     current_ns=ns)
                # If a cast function is provided, apply it to the namespace, possibly doing dynamic type checking
                # and also allowing the checker to provide hinting for the types of the fields
                flags = ns.flags_
                del ns.flags_
                fixed = (q.cast(current_ns.__dict__[q.namespace]) #(q.cast(**vars(current_ns.__dict__[q.namespace]))
                                                    if q.cast else current_ns.__dict__[q.namespace])
                if isinstance(fixed, tuple):
                    fixed = fixed.replace(flags_=flags)
                elif isinstance(fixed, Namespace):
                    setattr(fixed, "flags_", flags)
                else:
                    raise ValueError("currently only Namespace and NamedTuple objects are supported return types from "
                                     "parsing; got %s (a %s)" % (fixed, type(fixed)))
                current_ns.__dict__[q.namespace] = fixed
                # TODO current_ns or current_namespace or ns or namespace?
        else:
            raise TypeError("parser %s wasn't a %s (%s or %s) but a %s" %
                            (p, Parser, BaseParser, CompoundParser, p.__class__))

    main_ns = Namespace()
    go_2(parser, current_prefix="", current_ns=main_ns)
    return main_ns


def _mk_application_parser(p: ArgParser) -> ArgParser:
    # p = ArgParser(add_help=False)
    g = p.add_argument_group("General application options",
                             "General options for all pydpiper applications.")
    #TODO is this broken? Shouldn't this action be "store_true"?
    g.add_argument("--restart", dest="restart",
                   action="store_false", default=True,
                   help="""
                   Restarts the pipeline by checking each finished stage's hash to its hash stored in the
                   finished_stages file. If any stages were changed (perhaps due to different arguments),
                   those stages will have to be re-run. Otherwise, the pipeline will restart from where it was last stopped.
                   [default = %(default)s]
                   """)
    g.add_argument("--smart-restart", dest="smart_restart",
                   action="store_true", default=False,
                   help="""
                   Also check if a stage's input files were modified after its output files, in which case re-run that stage.
                   [default = %(default)s]
                   """)
    g.add_argument("--pipeline-name", dest="pipeline_name", type=str,
                   default=time.strftime("pipeline-%d-%m-%Y-at-%H-%m-%S"),
                   help="Name of pipeline and prefix for models.")

    g.add_argument("--no-restart", dest="restart",
                   action="store_false", help="Opposite of --restart")
    # TODO instead of prefixing all subdirectories (logs, backups, processed, ...)
    # with the pipeline name/date, we could create one identifying directory
    # and put these other directories inside
    g.add_argument("--output-dir", dest="output_directory",
                   type=str,
                   default='',
                   help="Directory where output data and backups will be saved.")
    g.add_argument("--create-graph", dest="create_graph",
                   action="store_true", default=False,
                   help="Create a .dot file with graphical representation of pipeline relationships [default = %(default)s]")
    g.set_defaults(execute=True)
    g.set_defaults(verbose=True)
    g.add_argument("--execute", dest="execute",
                   action="store_true",
                   help="Actually execute the planned commands [default = %(default)s]")
    g.add_argument("--no-execute", dest="execute",
                   action="store_false",
                   help="Opposite of --execute")
    g.add_argument("--version", action="version",
                   version="%(prog)s (" + get_distribution("pydpiper").version + ")")  # pylint: disable=E1101
    g.add_argument("--verbose", dest="verbose",
                   action="store_true",
                   help="Be verbose in what is printed to the screen [default = %(default)s]")
    g.add_argument("--no-verbose", dest="verbose",
                   action="store_false",
                   help="Opposite of --verbose")
    g.add_argument("--files", type=str, nargs='*', metavar='file',
                   help='Files to process')
    g.add_argument("--csv-file", dest="csv_file",
                   type=str, default=None,
                   help="CSV file containing application-specific columns. [Default=%(default)s]")
    # TODO also allow relative to pipeline output dir??
    g.add_argument("--csv-paths-relative-to-wd", dest="csv_paths_relative_to_wd", default=False,
                   action="store_true",
                   help="CSV paths are relative to the working directory, not the CSV file")
    return p


application_parser = AnnotatedParser(parser=BaseParser(_mk_application_parser(ArgParser(add_help=False)),
                                                       "application"),
                                     namespace="application")

def _mk_execution_parser(p: ArgParser) -> ArgParser:
    # parser = ArgParser(add_help=False)
    group = p.add_argument_group("Executor options",
                                 "Options controlling how and where the code is run.")
    group.add_argument("--uri-file", dest="urifile",
                       type=str, default=None,
                       help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    group.add_argument("--use-ns", dest="use_ns",
                       action="store_true",
                       help="Use the Pyro NameServer to store object locations. Currently a Pyro nameserver must be started separately for this to work.")
    group.add_argument("--latency-tolerance", dest="latency_tolerance",
                       type=float, default=600.0,
                       help="Allowed grace period by which an executor may miss a heartbeat tick before being considered failed [Default = %(default)s.")
    group.add_argument("--num-executors", dest="num_exec",
                       type=int, default=-1,
                       help="Number of independent executors to launch. [Default = %(default)s. Code will not run without an explicit number specified.]")
    group.add_argument("--max-failed-executors", dest="max_failed_executors",
                       type=int, default=10,
                       help="Maximum number of failed executors before we stop relaunching. [Default = %(default)s]")
    # TODO: add corresponding --monitor-heartbeats
    group.add_argument("--no-monitor-heartbeats", dest="monitor_heartbeats",
                       action="store_false",
                       help="Don't assume executors have died if they don't check in with the server (NOTE: this can hang your pipeline if an executor crashes).")
    group.add_argument("--time", dest="time", 
                       type=str, default="23:59:59",
                       help="Wall time to request for each server/executor in the format hh:mm:ss. Required only if --queue-type=pbs. Current default on PBS is %(default)s.")
    group.add_argument("--proc", dest="proc",
                       type=int, default=1,
                       help="Number of processes per executor. Also sets max value for processor use per executor. [Default = %(default)s]")
    group.add_argument("--mem", dest="mem",
                       type=float, default=6,
                       help="Total amount of requested memory (in GB) for all processes the executor runs. [Default = %(default)s].")
    group.add_argument("--pe", dest="pe",
                       type=str, default=None,
                       help="Name of the SGE pe, if any. [Default = %(default)s]")
    group.add_argument("--mem-request-attribute", dest="mem_request_attribute",
                       type=str, default=None,
                       help="Name of the resource attribute to request for managing memory limits. [Default = %(default)s]")
    group.add_argument("--greedy", dest="greedy",
                       action="store_true",
                       help="Request the full amount of RAM specified by --mem rather than the (lesser) amount needed by runnable jobs.  Always use this if your executor is assigned a full node.")
    group.add_argument("--ppn", dest="ppn",
                       type=int, default=8,
                       help="Number of processes per node. Used when --queue-type=pbs. [Default = %(default)s].")
    group.add_argument("--queue-name", dest="queue_name", type=str, default=None,
                       help="Name of the queue, e.g., all.q (MICe) or batch (SciNet)")
    group.add_argument("--queue-type", dest="queue_type", type=str, default=None,
                       help="""Queue type to submit jobs, i.e., "sge" or "pbs".  [Default = %(default)s]""")
    group.add_argument("--queue-opts", dest="queue_opts",
                       type=str, default="",
                       help="A string of extra arguments/flags to pass to qsub. [Default = %(default)s]")
    group.add_argument("--executor-start-delay", dest="executor_start_delay", type=int, default=180,
                       help="Seconds before starting remote executors when running the server on the grid")
    group.add_argument("--submit-server", dest="submit_server", action="store_true",
                       help="Submit the server to the grid.  Currently works only with PBS/Torque systems.")
    group.add_argument("--no-submit-server", dest="submit_server", action="store_false",
                       help="Opposite of --submit-server. [default]")
    group.add_argument("--max-idle-time", dest="max_idle_time",
                       type=int, default=1,
                       help="The number of minutes an executor is allowed to continuously sleep, i.e. wait for an available job, while active on a compute node/farm before it kills itself due to resource hogging. [Default = %(default)s]")
    group.add_argument("--time-to-accept-jobs", dest="time_to_accept_jobs",
                       type=int,
                       help="The number of minutes after which an executor will not accept new jobs anymore. This can be useful when running executors on a batch system where other (competing) jobs run for a limited amount of time. The executors can behave in a similar way by given them a rough end time. [Default = %(default)s]")
    group.add_argument('--local', dest="local", action='store_true',
                       help="Don't submit anything to any specified queueing system but instead run as a server/executor")
    group.add_argument("--config-file", type=str, metavar='config_file', is_config_file=True,
                       required=False, help='Config file location')
    group.add_argument("--prologue-file", type=str, metavar='file',
                       help="Location of a shell script to inline into PBS submit script to set paths, load modules, etc.")
    group.add_argument("--min-walltime", dest="min_walltime", type=int, default=0,
                       help="Min walltime (s) allowed by the queuing system [Default = %(default)s]")
    group.add_argument("--max-walltime", dest="max_walltime", type=int, default=None,
                       help="Max walltime (s) allowed for jobs on the queuing system, or infinite if None [Default = %(default)s]")
    group.add_argument("--default-job-mem", dest="default_job_mem",
                       type=float, default=1.75,
                       help="Memory (in GB) to allocate to jobs which don't make a request. [Default=%(default)s]")
    group.add_argument("--memory-factor", dest="memory_factor",
                       type=float, default=1,
                       help="Overall factor by which to scale all memory estimates/requests (including default job memory, "
                            "but not executor totals (--mem)), say due to system differences or overcommitted nodes. "
                            "[Default=%(default)s]")
    group.add_argument("--cmd-wrapper", dest="cmd_wrapper",
                       type=str, default="",
                       help="Wrapper inside of which to run the command, e.g., '/usr/bin/time -v'. [Default='%(default)s']")
    group.add_argument("--check-input-files", dest="check_input_files", action="store_true",
                       help="Check overall pipeline inputs exist and, when applicable, "
                            "are valid MINC files [Default=%(default)s]")
    group.add_argument("--no-check-input-files", dest="check_input_files", action="store_false",
                       help="Opposite of --check-input-files")
    group.set_defaults(check_inputs=True)
    group.add_argument("--check-outputs", dest="check_outputs",
                       action="store_true",
                       help="Check output files exist and error if not [Default=%(default)s]")
    group.add_argument("--no-check-outputs", dest="check_outputs",
                       action="store_false",
                       help="Opposite of --check-outputs.")
    group.set_defaults(check_outputs=False)
    group.add_argument("--fs-delay", dest="fs_delay",
                       type=float, default=5,
                       help="Time (sec) to allow for NFS to become consistent after stage completion [Default=%(default)s]")
    group.add_argument("--executor_wrapper", dest="executor_wrapper",
                       type=str, default="",
                       help="Command inside of which to run the executor. [Default='%(default)s']")
    group.add_argument("--defer-directory-creation", default=False,
                       action="store_true", dest="defer_directory_creation",
                       help="Create relevant directories when a stage is run instead of at startup [Default=%(default)s]")
    group.add_argument("--generate-makeflow", dest="generate_makeflow", default=False, action="store_true",
                       help="generate Makeflow instead of running")
    return p


execution_parser = AnnotatedParser(parser=BaseParser(_mk_execution_parser(ArgParser(add_help=False)),
                                                      'execution'),
                                    namespace="execution")


def _mk_registration_parser(p: ArgParser) -> ArgParser:
    g = p.add_argument_group("General registration options",
                             "....")
    # p = ArgParser(add_help=False)
    g.add_argument("--input-space", dest="input_space",
                   type=lambda x: InputSpace[x],  # type: ignore # mypy/issues/741
                   default=InputSpace.native,
                   # choices=[x for x, _ in InputSpace.__members__.items()],
                   help="Option to specify space of input-files. Can be native (default), lsq6, lsq12. "
                        "Native means that there is no prior formal alignment between the input files "
                        "yet. lsq6 means that the input files have been aligned using translations "
                        "and rotations; the code will continue with a 12 parameter alignment. lsq12 "
                        "means that the input files are fully linearly aligned. Only non-linear "
                        "registrations are performed. [Default=%(default)s]")
    g.add_argument("--resolution", dest="resolution",
                   type=float, default=None,
                   help="Specify the resolution at which you want the registration to be run. "
                        "If not specified, the resolution of the target of your pipeline will "
                        "be used. [Default=%(default)s]")
    g.add_argument("--subject-matter", dest="subject_matter",
                   type=str, default=None,
                   help="Specify the subject matter for the pipeline. This will set the parameters "
                        "for multiple programs based on the overall size of the subject matter. Instead "
                        "of using the resolution of the files. Currently supported option is: \"mousebrain\" "
                        "[Default=%(default)s]")
    return p  # g?


registration_parser = AnnotatedParser(parser=BaseParser(_mk_registration_parser(ArgParser(add_help=False)),
                                                        "registration"),
                                      namespace="registration",
                                      cast=lambda ns: RegistrationConf(**vars(ns)))


def _mk_lsq6_parser(with_nuc : bool = True,
                    with_inormalize : bool = True):
    p = ArgParser(add_help=False)
    p.set_defaults(lsq6_method="lsq6_large_rotations")
    p.set_defaults(nuc = True if with_nuc else False)
    p.set_defaults(inormalize = True if with_inormalize else False)
    p.set_defaults(copy_header_info=False)
    # TODO: should this actually be part of the LSQ6 component?  What would it return in this case?
    p.set_defaults(run_lsq6=True)
    p.add_argument("--run-lsq6", dest="run_lsq6",
                   action="store_true",
                   help="Actually run the 6 parameter alignment [default = %(default)s]")
    p.add_argument("--no-run-lsq6", dest="run_lsq6",
                   action="store_false",
                   help="Opposite of --run-lsq6")
    # TODO should be part of some mutually exclusive group ...
    p.add_argument("--init-model", dest="init_model",
                   type=str, default=None,
                   help="File in standard space in the initial model. The initial model "
                        "can also have a file in native space and potentially a transformation "
                        "file. See our wiki (https://wiki.mouseimaging.ca/) for detailed "
                        "information on initial models. [Default = %(default)s]")
    p.add_argument("--lsq6-target", dest="lsq6_target",
                   type=str, default=None,
                   help="File to be used as the target for the initial (often 6-parameter) alignment. "
                        "[Default = %(default)s]")
    p.add_argument("--bootstrap", dest="bootstrap",
                   action="store_true", default=False,
                   help="Use the first input file to the pipeline as the target for the "
                        "initial (often 6-parameter) alignment. [Default = %(default)s]")
    # TODO: add information about the pride of models to the code in such a way that it
    # is reflected on GitHub
    p.add_argument("--pride-of-models", dest="pride_of_models",
                   type=str, default=None,
                   help="(selected longitudinal pipelines only!) Specify a csv file that contains the mapping of "
                        "all your initial models at different time points. The idea is that you might "
                        "want to use different initial models for the time points in your data. "
                        "The csv file should have one column called \"model_file\", and one column "
                        "called \"time_point\". The time points can be given in either integer values "
                        "or float values. Each model file should point to the file in standard space "
                        "for that particular model.  [Default = %(default)s]")
    # TODO: do we need to implement this option? This was for Kieran Short, but the procedure
    # he will be using in the future most likely will not involve this option.
    # group.add_argument("--lsq6-alternate-data-prefix", dest="lsq6_alternate_prefix",
    #                   type=str, default=None,
    #                   help="Specify a prefix for an augmented data set to use for the 6 parameter "
    #                   "alignment. Assumptions: there is a matching alternate file for each regular input "
    #                   "file, e.g. input files are: input_1.mnc input_2.mnc ... input_n.mnc. If the "
    #                   "string provided for this flag is \"aug_\", then the following files should exist: "
    #                   "aug_input_1.mnc aug_input_2.mnc ... aug_input_n.mnc. These files are assumed to be "
    #                   "in the same orientation/location as the regular input files.  They will be used for "
    #                   "for the 6 parameter alignment. The transformations will then be used to transform "
    #                   "the regular input files, with which the pipeline will continue.")
    p.add_argument("--lsq6-simple", dest="lsq6_method",
                   action="store_const", const="lsq6_simple",
                   help="Run a 6 parameter alignment assuming that the input files are roughly "
                        "aligned: same space, similar orientation. Keep in mind that if you use an "
                        "initial model with both a standard and a native space, the assumption is "
                        "that the input files are already roughly aligned to the native space. "
                        "Three iterations are run: 1st is 17 times stepsize blur, 2nd is 9 times "
                        "stepsize gradient, 3rd is 4 times stepsize blur. [Default = %(default)s]")
    p.add_argument("--lsq6-centre-estimation", dest="lsq6_method",
                   action="store_const", const="lsq6_centre_estimation",
                   help="Run a 6 parameter alignment assuming that the input files have a "
                        "similar orientation, but are scanned in different coils/spaces. [Default = %(default)s]")
    p.add_argument("--lsq6-large-rotations", dest="lsq6_method",
                   action="store_const", const="lsq6_large_rotations",
                   help="Run a 6 parameter alignment assuming that the input files have a random "
                        "orientation and are scanned in different coils/spaces. A brute force search over "
                        "the x,y,z rotation space is performed to find the best 6 parameter alignment. "
                        "[Default = %(default)s]")
    p.add_argument("--lsq6-large-rotations-tmp-dir", dest="rotation_tmp_dir",
                   type=str, default="/dev/shm/",
                   help="Specify the directory that rotational_minctracc.py uses for temporary files. "
                        "By default we use /dev/shm/, because this program involves a lot of I/O, and "
                        "this is probably one of the fastest way to provide this. [Default = %(default)s]")
    p.add_argument("--lsq6-large-rotations-parameters", dest="rotation_params",
                   type=str, default="5,4,10,8",
                   help="Settings for the large rotation alignment. factor=factor based on smallest file "
                        "resolution: 1) blur factor, 2) resample step size factor, 3) registration step size "
                        "factor, 4) w_translations factor  ***** if you are working with mouse brain data "
                        " the defaults do not have to be based on the file resolution; a default set of "
                        " settings works for all mouse brain. In order to use those setting, specify: "
                        "\"mousebrain\" as the argument for this option. ***** [default = %(default)s]")
    p.add_argument("--lsq6-rotational-range", dest="rotation_range",
                   type=int, default=50,
                   help="Settings for the rotational range in degrees when running the large rotation "
                        "alignment. [Default = %(default)s]")
    p.add_argument("--lsq6-rotational-interval", dest="rotation_interval",
                   type=int, default=10,
                   help="Settings for the rotational interval in degrees when running the large rotation "
                        "alignment. [Default = %(default)s]")
    p.add_argument("--nuc", dest="nuc",
                   action="store_true",
                   help="Perform non-uniformity correction. [Default = %(default)s]")
    p.add_argument("--no-nuc", dest="nuc",
                   action="store_false",
                   help="If specified, do not perform non-uniformity correction. Opposite of --nuc.")
    p.add_argument("--inormalize", dest="inormalize",
                   action="store_true",
                   help="Normalize the intensities after lsq6 alignment and nuc, if done. "
                        "[Default = %(default)s] ")
    p.add_argument("--no-inormalize", dest="inormalize",
                   action="store_false",
                   help="If specified, do not perform intensity normalization. Opposite of --inormalize.")
    p.add_argument("--copy-header-info-to-average", dest="copy_header_info",
                   action="store_true",
                   help="Copy the MINC header information of the first input file into the "
                        "average that is created. [Default = %(default)s] ")
    p.add_argument("--no-copy-header-info-to-average", dest="copy_header_info",
                   action="store_false",
                   help="Opposite of --copy-header-info-to-average.")
    p.add_argument("--lsq6-protocol", dest="protocol_file",
                   type=str, default=None,
                   help="Specify an lsq6 protocol that overrides the default setting for stages in "
                        "the 6 parameter minctracc call. Parameters must be specified as in the following \n"
                        "example: applications_testing/test_data/minctracc_example_linear_protocol.csv \n"
                        "[Default = %(default)s].")
    return p


# the cast is basically mandatory, while it's easy to change the namespace if needed, so
# upgraded this from a BaseParser to an AnnotatedParser
lsq6_parser = AnnotatedParser(parser=BaseParser(_mk_lsq6_parser(), "LSQ6"), namespace="lsq6", cast=to_lsq6_conf)


def _mk_stats_parser():
    p = ArgParser(add_help=False)
    # p.add_argument_group("Statistics options",
    #                      "Options for calculating statistics.")
    default_fwhms = "0.2"
    p.set_defaults(stats_kernels=default_fwhms)
    p.set_defaults(calc_stats=True)
    p.add_argument("--calc-stats", dest="calc_stats",
                   action="store_true",
                   help="Calculate statistics at the end of the registration. [Default = %(default)s]")
    p.add_argument("--no-calc-stats", dest="calc_stats",
                   action="store_false",
                   help="If specified, statistics are not calculated. Opposite of --calc-stats.")
    p.add_argument("--stats-kernels", dest="stats_kernels",
                   type=str,
                   help="comma separated list of blurring kernels for analysis. [Default = %(default)s].")
    return p


_stats_parser = BaseParser(_mk_stats_parser(), "stats")
stats_parser = AnnotatedParser(parser=_stats_parser, namespace="stats")


def _mk_chain_parser():
    p = ArgParser(add_help=False)
    p.add_argument("--csv-file", dest="csv_file",
                   type=str, required=True,
                   help="The spreadsheet with information about your input data. "
                        "For the registration chain you are required to have the "
                        "following columns in your csv file: \" subject_id\", "
                        "\"timepoint\", and \"filename\". Optionally you can have "
                        "a column called \"is_common\" that indicates that a scan "
                        "is to be used for the common time point registration "
                        "using a 1, and 0 otherwise.")
    p.add_argument("--common-time-point", dest="common_time_point",
                   type=int, default=None,
                   help="The time point at which the inter-subject registration will be "
                        "performed. I.e., the time point that will link the subjects together. "
                        "If you want to use the last time point from each of your input files, "
                        "(they might differ per input file) specify -1. If the common time "
                        "is not specified, the assumption is that the spreadsheet contains "
                        "the mapping using the \"is_common\" column. [Default = %(default)s]")
    p.add_argument("--common-time-point-name", dest="common_time_point_name",
                   type=str, default="common",
                   help="Option to specify a name for the common time point. This is useful for the "
                        "creation of more readable output file names. Default is \"common\". Note "
                        "that the common time point is the one created by an iterative group-wise "
                        "registration (inter-subject).")
    return p


_chain_parser = BaseParser(_mk_chain_parser(), "chain")
chain_parser = AnnotatedParser(parser=_chain_parser, namespace="chain")  # TODO cast to ChainConf

# TODO: probably doesn't belong here ... do we need to move them again to the
# modules where complementary code is?
def _mk_lsq12_parser():
    p = ArgParser(add_help=False)
    # group = parser.add_argument_group("LSQ12 registration options",
    #                                  "Options for performing a pairwise, affine registration")
    p.set_defaults(run_lsq12=True)
    p.set_defaults(generate_tournament_style_lsq12_avg=False)
    p.add_argument("--run-lsq12", dest="run_lsq12",
                   action="store_true",
                   help="Actually run the 12 parameter alignment [default = %(default)s]")
    p.add_argument("--no-run-lsq12", dest="run_lsq12",
                   action="store_false",
                   help="Opposite of --run-lsq12")
    p.add_argument("--lsq12-max-pairs", dest="max_pairs",
                   type=parse_nullable_int, default=25,
                   help="Maximum number of pairs to register together ('None' implies all pairs). "
                        "[Default = %(default)s]")
    p.add_argument("--lsq12-likefile", dest="like_file",
                   type=str, default=None,
                   help="Can optionally specify a 'like'-file for resampling at the end of pairwise "
                        "alignment. Default is None, which means that the input file will be used. "
                        "[Default = %(default)s]")
    p.add_argument("--lsq12-protocol", dest="protocol",
                   type=str,
                   help="Can optionally specify a registration protocol that is different from defaults. "
                        "Parameters must be specified as in the following example: \n"
                        "applications_testing/test_data/minctracc_example_linear_protocol.csv \n"
                        "[Default = %(default)s].")
    #p.add_argument("--generate-tournament-style-lsq12-avg", dest="generate_tournament_style_lsq12_avg",
    #               action="store_true",
    #               help="Instead of creating the average of the lsq12 resampled files "
    #                    "by simply averaging them directly, create an iterative average "
    #                    "as follows. Perform a non linear registration between pairs "
    #                    "of files. Resample each file halfway along that transformation "
    #                    "in order for them to end up in the middle. Average those two files. "
    #                    "Then continue on to the next level as in a tournament. [default = %(default)s]")
    #p.add_argument("--no-generate-tournament-style-lsq12-avg", dest="generate_tournament_style_lsq12_avg",
    #               action="store_false",
    #               help="Opposite of --generate-tournament-style-lsq12-avg")
    return p


def to_lsq12_conf(lsq12_args : Namespace) -> LSQ12Conf:
    return LSQ12Conf(**lsq12_args.__dict__)

_lsq12_parser = BaseParser(_mk_lsq12_parser(), "LSQ12")
lsq12_parser = AnnotatedParser(parser=_lsq12_parser, namespace="lsq12") #, cast=to_lsq12_conf)

def _mk_nlin_parser(p: ArgParser):
    group = p.add_argument_group("Nonlinear registration options",
                                 "Options for performing a non-linear registration")
    group.add_argument("--registration-method", dest="reg_method",
                       default="ANTS", choices=["ANTS",
                                                "antsRegistration",
                                                "minctracc"],
                       help="Specify algorithm used for non-linear registrations. "
                            "[Default = %(default)s]")
    # TODO wire up the choices here in reg_method and reg_strategy to the actual ones ...
    group.add_argument("--registration-strategy", dest="reg_strategy",
                        default="build_model", choices=['build_model', 'pairwise', 'tournament',
                                                        'tournament_and_build_model', 'pairwise_and_build_model'],
                        help="Process used for model construction [Default = %(default)s")
    group.add_argument("--nlin-protocol", dest="nlin_protocol",
                       type=str, default=None,
                       help="Can optionally specify a registration protocol that is different from defaults. "
                            "Parameters must be specified as in either or the following examples: \n"
                            "applications_testing/test_data/minctracc_example_nlin_protocol.csv \n"
                            "applications_testing/test_data/mincANTS_example_nlin_protocol.csv \n"
                            "[Default = %(default)s]")
    group.add_argument("--use-robust-averaging", dest="use_robust_averaging", action='store_true',
                       help="use robust intensity averaging if possible")
    p.set_defaults(use_robust_averaging = False)
    return p

NLINConf = NamedTuple('NLINConf', [('reg_method', str),  # TODO make this an enumerated type
                                   ('nlin_protocol', Optional[str]),
                                   ('nlin_pairwise', bool)])

def to_nlin_conf(nlin_args : Namespace) -> NLINConf:
    return NLINConf(**nlin_args.__dict__)

_nlin_parser = BaseParser(_mk_nlin_parser(ArgParser(add_help=False)), group_name='nlin')
nlin_parser = AnnotatedParser(parser=_nlin_parser, namespace="nlin")  #, cast=to_nlin_conf)
# TODO cast doesn't work for NLIN.py parser with extra --target flag


def _mk_segmentation_parser(parser : ArgParser, default : bool):
    group = parser.add_argument_group("Segmentation", "Segmentation options.")
    group.add_argument("--run-maget", action='store_true', dest="run_maget",
                       help="Run MAGeT segmentation. [default = %(default)s]")
    group.add_argument("--no-run-maget", dest="run_maget",
                       action='store_false', help="Don't run MAGeT segmentation")
    parser.set_defaults(run_maget=True)
    return parser


# FIXME change this default ?!
segmentation_parser = AnnotatedParser(parser=BaseParser(_mk_segmentation_parser(ArgParser(add_help=False),
                                                                                default=True),
                                                        "segmentation"),
                                      namespace="segmentation")



