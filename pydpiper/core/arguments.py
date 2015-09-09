'''
This file will contain argument option groups that can be used 
by PydPiper applications. Some option groups might be mandatory/highly
recommended to add to your application: e.g. the arguments that deal
with the execution of your application.
'''
import time

# TODO: most (if not all) of the following options don't do anything yet
# we should come up with a good way to deal with all this. Given that
# Jason wants to be able to connect each of the applications together
# in new code (e.g., by simply calling chain(...args...) or MBM(,,,args...)
# there needs to be another way to create the options hash? So that you could
# write applications that require little to no command line arguments?



def addApplicationArgumentGroup(parser):
    """
    The arguments that all applications share:
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
    group.add_argument("--no-restart", dest="restart", 
                               action="store_false", help="Opposite of --restart")
    # TODO instead of prefixing all subdirectories (logs, backups, processed, ...)
    # with the pipeline name/date, we could create one identifying directory
    # and put these other directories inside
    group.add_argument("--output-dir", dest="output_directory",
                               type=str, default=None,
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
    group.add_argument("--version", dest="show_version",
                               action="store_true",
                               help="Print the version number and exit.")
    group.add_argument("--verbose", dest="verbose",
                               action="store_true",
                               help="Be verbose in what is printed to the screen [default = %(default)s]")
    group.add_argument("--no-verbose", dest="verbose",
                               action="store_false",
                               help="Opposite of --verbose [default]")
    group.add_argument("files", type=str, nargs='*', metavar='file',
                        help='Files to process')



def addExecutorArgumentGroup(parser):
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
                         "General options for running various types of registrations.")
    group.add_argument("--pipeline-name", dest="pipeline_name", type=str,
                       default=time.strftime("pipeline-%d-%m-%Y-at-%H-%m-%S"),
                       help="Name of pipeline and prefix for models.")
    group.add_argument("--input-space", dest="input_space",
                       type=str, default="native", 
                       help="Option to specify space of input-files. Can be native (default), lsq6, lsq12. "
                            "Native means that there is no prior formal alignent between the input files " 
                            "yet. lsq6 means that the input files have been aligned using translations "
                            "and rotations; the code will continue with a 12 parameter alignment. lsq12 " 
                            "means that the input files are fully linearly aligned. Only non linear "
                            "registrations are performed.")

def addStatsArgumentGroup(parser):
    group = parser.add_argument_group("Statistics options", 
                          "Options for calculating statistics.")
    group.add_argument("--stats-kernels", dest="stats_kernels",
                       type=str, default="0.5,0.2,0.1", 
                       help="comma separated list of blurring kernels for analysis. Default is: 0.5,0.2,0.1")


def addRegistrationChainArgumentGroup(parser):
    group = parser.add_argument_group("Registration chain options",
                        "Options for processing longitudinal data.")
    group.add_argument("--csv-file", dest="csv_file",
                       type=str, default=None,
                       help="The spreadsheet with information about your input data. "
                            "For the registration chain you are required to have the "
                            "following columns in your csv file: \" subject_id\", "
                            "\"timepoint\", and \"filename\". Optionally you can have "
                            "a column called \"is_common\" that indicates that a subject "
                            "is to be used for the common time point using a 1, and 0 "
                            "otherwise.")
    group.add_argument("--common-time-point", dest="common_time_point",
                       type=int, default=None,
                       help="The time point at which the inter-subject registration will be "
                            "performed. I.e., the time point that will link the subjects together. "
                            "If you want to use the last time point from each of your input files, "
                            "(they might differ per input file) specify -1. If the common time "
                            "is not specified, the assumption is that the spreadsheet contains "
                            "the mapping using the \"is_common\" column. [Default = %(default)s]")
    group.add_argument("--common-time-point-name", dest="common_name",
                       type=str, default="common", 
                       help="Option to specify a name for the common time point. This is useful for the "
                            "creation of more readable output file names. Default is \"common\". Note "
                            "that the common time point is the one created by an iterative group-wise " 
                            "registration (inter-subject).")

