'''
This file will contain argument option groups that can be used 
by PydPiper applications. Some option groups might be mandatory/highly
recommended to add to your application: e.g. the arguments that deal
with the execution of your application.
'''
from configargparse import ArgParser, Namespace
from atom.api import Atom, Enum, Instance, Int, Str

from collections import namedtuple
from pkg_resources import get_distribution
import copy
import os
import sys
import time

from pydpiper.core.util import raise_

# TODO: should the pipeline-specific argument handling be located here
# or in that pipeline's module?  Makes more sense (general stuff
# can still go here)

class PydParser(ArgParser):
    # Some sneakiness... override the format_epilog method
    # to return the epilog verbatim.
    # That way in the help message you can include an example
    # of what an lsq6/nlin protocol should look like, and this
    # won't be mangled when the parser calls `format_epilog`
    # when writing its help.
    def format_epilog(self, formatter):
        if not self.epilog:
            self.epilog = ""
        return self.epilog

# TODO delete/move to util?
def id(*args, **kwargs):
    return args, kwargs

def nullable_int(string):
    if string == "None":
        return None
    else:
        return int(string)

#Annotated = namedtuple('Annotated', ['it', 'prefix', 'namespace'])
class Annotated(Atom):
    it        = Instance(object, factory=lambda : raise_(ValueError("must provide a parser")))
    prefix    = Str("")
    namespace = Str("", factory=lambda : raise_(ValueError("must provide a namespace")))
    proc      = Instance(object, factory=lambda : None) #lambda y: y)

class Parser: pass

class BaseParser(Parser):
    def __init__(self, argparser): # Some(
        self.argparser = argparser

class CompoundParser(Parser):
    def __init__(self, annotated_parsers):
        self.parsers  = annotated_parsers

#combine_parsers = CompoundParsers

#Parser = BaseParser ArgParser | CompoundParser([Annotated Parser]) - rose tree with elts at leaves instead of nodes?
# for more flexibility, you could also add an extra parser at the node, but that doesn't seem to be needed
# (you can always add an extra leaf at that node) (~isomorphic?)

# TODO: What about the situation when you want cross-cutting?  That is, instead of the usual situation
# (everything grows like a tree with more and more prefixes), you have components at very far-apart
# locations in the tree which you wish to control via a single set of options?  For instance,
# what if you have two twolevel models within a pipeline and want to set the second-level LSQ12 on both
# via --second-level-lsq12-max-pairs?  Within one twolevel pipeline, you can do this easily.
# I guess you could add a lsq12 parser in the code calling the two pipelines and use it as a default,
# but this wouldn't happen automagically.

def parse(parser, args):
    default_config_file = os.getenv("PYDPIPER_CONFIG_FILE") #TODO: accepting a comma-separated list might allow more flexibility
    config_files = [default_config_file] if default_config_file else []

    # First, build a parser that's aware of all options
    # (will be used for help/version/error messages).
    # This must be tried _before_ the partial parsing attempts
    # in order to get correct help/version messages.

    main_parser = ArgParser(default_config_files=config_files)

    # TODO: abstract out the recursive travels in go_1 and go_2
    def go_1(p, current_prefix):
        if isinstance(p, BaseParser):
            for a in p.argparser._actions:
                a = copy.copy(a)
                ss = copy.deepcopy(a.option_strings)
                for ix, s in enumerate(a.option_strings):
                    if s.startswith("--"):
                        ss[ix] = "-" + current_prefix + "-" + s[2:]
                    else:
                        raise NotImplementedError
                a.option_strings = ss
                main_parser._add_action(a)
        elif isinstance(p, CompoundParser):
            for q in p.parsers:
                go_1(q.it, current_prefix + "-" + q.prefix)
        else:
            raise TypeError("parser %s wasn't a %s (%s or %s) but a %s" % (p, Parser, BaseParser, CompoundParser, p.__class__))

    go_1(parser, "")

    # Use this parser to exit with a helpful message if parse fails or --help/--version specified:
    main_parser.parse_args(args)

    # Now, use parse_known_args for each parser in the tree of parsers to fill the appropriate namespace object ...
    def go_2(p, current_prefix, current_ns):
        if isinstance(p, BaseParser):
            new_p = ArgParser(default_config_files=config_files)
            for ix, a in enumerate(p.argparser._actions):
                new_a = copy.copy(a)
                ss = copy.deepcopy(new_a.option_strings)
                for ix, s in enumerate(new_a.option_strings):
                    if s.startswith("--"):
                        ss[ix] = "-" + current_prefix + "-" + s[2:]
                    else:
                        raise NotImplementedError
                    new_a.option_strings = ss
                new_p._add_action(new_a)
            used_args, _rest = new_p.parse_known_args(args, namespace=current_ns)  # TODO: could continue parsing from `_rest` instead of original `args`
        elif isinstance(p, CompoundParser):
            for q in p.parsers:
                ns = Namespace()
                if q.namespace in current_ns.__dict__:
                    raise ValueError("Namespace field '%s' already in use" % q.namespace)
                else:
                    current_ns.__dict__[q.namespace] = q.proc(**vars(ns)) if q.proc else ns # gross but how to write n-ary identity fn that behaves sensibly on single arg??
                go_2(q.it, current_prefix=current_prefix + "-" + q.prefix, current_ns=ns) # TODO current_ns or current_namespace or ns or namespace?
        else:
            raise TypeError("parser %s wasn't a %s (%s or %s) but a %s" % (p, Parser, BaseParser, CompoundParser, p.__class__))

    main_ns = Namespace()
    go_2(parser, current_prefix="", current_ns=main_ns)
    return(main_ns)

def with_parser(p):
    return lambda args: parse(p, args)

mbm_p = CompoundParser([Annotated(it=lsq6_p, prefix='lsq6', namespace="lsq6"), Annotated(it=lsq12_p, namespace="lsq12", prefix="lsq12", proc=Lsq12Conf)])
two_mbms = CompoundParser([Annotated(it=mbm_p, prefix="mbm1", namespace="mbm1"), Annotated(it=mbm_p, prefix="mbm2")])  #, namespace="mbm2")])
four_mbms = CompoundParser([Annotated(it=two_mbms, prefix="first-two-mbms", namespace="first-two"), Annotated(it=two_mbms, prefix="next-two-mbms", namespace="next-two")])

result = with_parser(four_mbms)(["--first-two-mbms-mbm1-lsq12-max-pairs", "10"]) #(['--lsq6-rotation-interval=30'])

def addApplicationArgumentGroup(parser):
    """
    The arguments that all applications share:
    --pipeline-name
    --restart
    --no-restart
    --output-dir
    --create-graph
    --execute
    --no-execute
    --version
    --verbose
    --no-verbose
    files (left over arguments (0 or more is allowed)
    """
    group = parser.add_argument_group("General application options", "General options for all pydpiper applications.")
    group.add_argument("--restart", dest="restart", 
                       action="store_false", default=True,
                       help="Restart pipeline using backup files. [default = %(default)s]")
    group.add_argument("--pipeline-name", dest="pipeline_name", type=str,
                       default=time.strftime("pipeline-%d-%m-%Y-at-%H-%m-%S"),
                       help="Name of pipeline and prefix for models.")

    group.add_argument("--no-restart", dest="restart", 
                        action="store_false", help="Opposite of --restart")
    # TODO instead of prefixing all subdirectories (logs, backups, processed, ...)
    # with the pipeline name/date, we could create one identifying directory
    # and put these other directories inside
    group.add_argument("--output-dir", dest="output_directory",
                       type=str, default='',
                       help="Directory where output data and backups will be saved.")
    group.add_argument("--create-graph", dest="create_graph",
                       action="store_true", default=False,
                       help="Create a .dot file with graphical representation of pipeline relationships [default = %(default)s]")
    parser.set_defaults(execute=True)
    parser.set_defaults(verbose=False)
    group.add_argument("--execute", dest="execute",
                       action="store_true",
                       help="Actually execute the planned commands [default = %(default)s]")
    group.add_argument("--no-execute", dest="execute",
                       action="store_false",
                       help="Opposite of --execute")
    group.add_argument("--version", action="version",
                       version="%(prog)s ("+get_distribution("pydpiper").version+")", # pylint: disable=E1101
                   ) #    help="Print the version number and exit.")
    group.add_argument("--verbose", dest="verbose",
                       action="store_true",
                       help="Be verbose in what is printed to the screen [default = %(default)s]")
    group.add_argument("--no-verbose", dest="verbose",
                       action="store_false",
                       help="Opposite of --verbose [default]")
    group.add_argument("files", type=str, nargs='*', metavar='file',
                        help='Files to process')



def addExecutorArgumentGroup(parser, prefix=None):
    group = parser.add_argument_group("Executor options",
                        "Options controlling how and where the code is run.")
    group.add_argument("--uri-file", dest="urifile",
                       type=str, default=None,
                       help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    group.add_argument("--use-ns", dest="use_ns",
                       action="store_true",
                       help="Use the Pyro NameServer to store object locations. Currently a Pyro nameserver must be started separately for this to work.")
    group.add_argument("--latency-tolerance", dest="latency_tolerance",
                       type=float, default=15.0,
                       help="Allowed grace period by which an executor may miss a heartbeat tick before being considered failed [Default = %(default)s.")
    group.add_argument("--num-executors", dest="num_exec", 
                       type=int, default=-1, 
                       help="Number of independent executors to launch. [Default = %(default)s. Code will not run without an explicit number specified.]")
    group.add_argument("--max-failed-executors", dest="max_failed_executors",
                      type=int, default=2,
                      help="Maximum number of failed executors before we stop relaunching. [Default = %(default)s]")
    # TODO: add corresponding --monitor-heartbeats
    group.add_argument("--no-monitor-heartbeats", dest="monitor_heartbeats",
                      action="store_false",
                      help="Don't assume executors have died if they don't check in with the server (NOTE: this can hang your pipeline if an executor crashes).")
    group.add_argument("--time", dest="time", 
                       type=str, default=None,
                       help="Wall time to request for each server/executor in the format hh:mm:ss. Required only if --queue-type=pbs. Current default on PBS is 48:00:00.")
    group.add_argument("--proc", dest="proc", 
                       type=int, default=1,
                       help="Number of processes per executor. Also sets max value for processor use per executor. [Default = %(default)s]")
    group.add_argument("--mem", dest="mem", 
                       type=float, default=6,
                       help="Total amount of requested memory (in GB) for all processes the executor runs. [Default = %(default)s].")
    group.add_argument("--pe", dest="pe",
                       type=str, default=None,
                       help="Name of the SGE pe, if any. [Default = %(default)s]")
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
    group.add_argument("--time-to-seppuku", dest="time_to_seppuku", 
                       type=int, default=1,
                       help="The number of minutes an executor is allowed to continuously sleep, i.e. wait for an available job, while active on a compute node/farm before it kills itself due to resource hogging. [Default = %(default)s]")
    group.add_argument("--time-to-accept-jobs", dest="time_to_accept_jobs", 
                       type=int,
                       help="The number of minutes after which an executor will not accept new jobs anymore. This can be useful when running executors on a batch system where other (competing) jobs run for a limited amount of time. The executors can behave in a similar way by given them a rough end time. [Default = %(default)s]")
    group.add_argument('--local', dest="local", action='store_true', help="Don't submit anything to any specified queueing system but instead run as a server/executor")
    group.add_argument("--config-file", type=str, metavar='config_file', is_config_file=True,
                       required=False, help='Config file location')
    group.add_argument("--prologue-file", type=str, metavar='file',
                       help="Location of a shell script to inline into PBS submit script to set paths, load modules, etc.")
    group.add_argument("--min-walltime", dest="min_walltime", type=int, default = 0,
            help="Min walltime (s) allowed by the queuing system [Default = %(default)s]")
    group.add_argument("--max-walltime", dest="max_walltime", type=int, default = None,
            help="Max walltime (s) allowed for jobs on the queuing system, or infinite if None [Default = %(default)s]")
    group.add_argument("--default-job-mem", dest="default_job_mem",
                       type=float, default = 1.75,
                       help="Memory (in GB) to allocate to jobs which don't make a request. [Default=%(default)s]")

def addGeneralRegistrationArgumentGroup(parser):
    group = parser.add_argument_group("General registration options",
                                      "....")
    group.add_argument("--input-space", dest="input_space",
                       choices=['native', 'lsq6', 'lsq12'], default="native", 
                       help="Option to specify space of input-files. Can be native (default), lsq6, lsq12. "
                            "Native means that there is no prior formal alignent between the input files " 
                            "yet. lsq6 means that the input files have been aligned using translations "
                            "and rotations; the code will continue with a 12 parameter alignment. lsq12 " 
                            "means that the input files are fully linearly aligned. Only non linear "
                            "registrations are performed.")
    group.add_argument("--resolution", dest="resolution", type=float,
                        help="Resolution to run the pipeline "
                        "(or determined by initial target if unspecified)")

# TODO: where should this live?
class RegistrationConf(Atom):
    input_space = Enum('native', 'lsq6', 'lsq12')
    resolution  = Instance(float)


def addStatsArgumentGroup(parser):
    group = parser.add_argument_group("Statistics options", 
                          "Options for calculating statistics.")
    default_fwhms = ['0.5','0.2','0.1']
    group.add_argument("--stats-kernels", dest="stats_kernels",
                       type=','.split, default=[0.5,0.2,0.1],
                       help="comma separated list of blurring kernels for analysis. Default is: %s" % ','.join(default_fwhms))


def addRegistrationChainArgumentGroup(parser):
    group = parser.add_argument_group("Registration chain options",
                        "Options for processing longitudinal data.")
#    addGeneralRegistrationArguments(group)
    group.add_argument("--csv-file", dest="csv_file",
                       type=str, required=True,
                       help="The spreadsheet with information about your input data. "
                            "For the registration chain you are required to have the "
                            "following columns in your csv file: \" subject_id\", "
                            "\"timepoint\", and \"filename\". Optionally you can have "
                            "a column called \"is_common\" that indicates that a scan "
                            "is to be used for the common time point registration"
                            "using a 1, and 0 otherwise.")
    group.add_argument("--common-time-point", dest="common_time_point",
                       type=int, default=None,
                       help="The time point at which the inter-subject registration will be "
                            "performed. I.e., the time point that will link the subjects together. "
                            "If you want to use the last time point from each of your input files, "
                            "(they might differ per input file) specify -1. If the common time "
                            "is not specified, the assumption is that the spreadsheet contains "
                            "the mapping using the \"is_common\" column. [Default = %(default)s]")
    group.add_argument("--common-time-point-name", dest="common_time_point_name",
                       type=str, default="common", 
                       help="Option to specify a name for the common time point. This is useful for the "
                            "creation of more readable output file names. Default is \"common\". Note "
                            "that the common time point is the one created by an iterative group-wise " 
                            "registration (inter-subject).")


core_pieces = [(addApplicationArgumentGroup, 'application'),
               (addExecutorArgumentGroup,    'execution')]

# TODO probably doesn't belong here ...
def addLSQ12ArgumentGroup(prefix):
    prefix = "" if prefix in ["", None] else (prefix + '-')
    def f(parser):
        """option group for the command line argument parser"""
        group = parser.add_argument_group("LSQ12 registration options",
                            "Options for performing a pairwise, affine registration")
        group.add_argument("--%slsq12-max-pairs" % prefix, dest="lsq12_max_pairs",
                           type=nullable_int, default=25,
                           help="Maximum number of pairs to register together ('None' implies all pairs).  [Default = %(default)s]")
        group.add_argument("--%slsq12-likefile" % prefix, dest="lsq12_likeFile",
                           type=str, default=None,
                           help="Can optionally specify a like file for resampling at the end of pairwise "
                           "alignment. Default is None, which means that the input file will be used. [Default = %(default)s]")
        group.add_argument("--%slsq12-subject-matter" % prefix, dest="lsq12_subject_matter",
                           type=str, default=None,
                           help="Can specify the subject matter for the pipeline. This will set the parameters "
                           "for the 12 parameter alignment based on the subject matter rather than the file "
                           "resolution. Currently supported option is: \"mousebrain\". [Default = %(default)s].")
        group.add_argument("--%slsq12-protocol" % prefix, dest="lsq12_protocol",
                           type=str, default=None,
                           help="Can optionally specify a registration protocol that is different from defaults. "
                           "Parameters must be specified as in the following example: \n"
                           "applications_testing/test_data/minctracc_example_linear_protocol.csv \n"
                           "[Default = %(default)s].")
        parser.add_argument_group(group)
    return f

