'''
This file will contain argument option groups that can be used 
by PydPiper applications. Some option groups might be mandatory/highly
recommended to add to your application: e.g. the arguments that deal
with the execution of your application.
'''

def addApplicationArgumentGroup(parser):
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
    group.add_argument("--num-executors", dest="num_exec", 
                       type=int, default=-1, 
                       help="Number of independent executors to launch. [Default = %(default)s. Code will not run without an explicit number specified.]")
    group.add_argument("--max-failed-executors", dest="max_failed_executors",
                      type=int, default=2,
                      help="Maximum number of failed executors before we stop relaunching. [Default = %(default)s]")
    # TODO add corresponding --monitor-heartbeats
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
    group.add_argument("--queue", dest="queue", 
                       type=str, default=None,
                       help="[DEPRECATED; use --queue-type instead.]  Use specified queueing system to submit jobs. Default is None.")              
    group.add_argument("--sge-queue-opts", dest="sge_queue_opts", 
                       type=str, default=None,
                       help="[DEPRECATED; use --queue-name instead.]  For --queue=sge, allows you to specify different queues. [Default = %(default)s]")
    group.add_argument("--queue-opts", dest="queue_opts",
                       type=str, default="",
                       help="A string of extra arguments/flags to pass to qsub. [Default = %(default)s]")
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




