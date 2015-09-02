from configargparse import ArgParser
from datetime import datetime
import time # TODO why both datetime and time?
from pkg_resources import get_distribution
import logging
import networkx as nx
import sys
import os

from pydpiper.execution.pipeline import Pipeline, pipelineDaemon
from pydpiper.execution.queueing import runOnQueueingSystem
from pydpiper.execution.file_handling import makedirsIgnoreExisting
from pydpiper.execution.pipeline_executor import addExecutorArgumentGroup, noExecSpecified
from pydpiper.core.conversion import directories

from   atom.api import Atom
import atom.api as atom

class ExecutionOptions(Atom):
    use_backup_files = Bool(True)
    create_graph     = Bool(False)
    execute          = Bool(True)
    # ... TODO put remainder of executor args, ..., here?

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

# Some sneakiness... Using the following lines, it's possible
# to add an epilog to the parser that is written to screen
# verbatim. That way in the help file you can show an example
# of what an lsq6/nlin protocol should look like.
class MyParser(ArgParser):
    def format_epilog(self, formatter):
        if not self.epilog:
            self.epilog = ""
        return self.epilog

def create_parser():
    default_config_file = os.getenv("PYDPIPER_CONFIG_FILE")
    files = [default_config_file] if default_config_file is not None else []
    parser = MyParser(default_config_files=files)
    addExecutorArgumentGroup(parser)
    addApplicationArgumentGroup(parser)
    return parser

def build_pipeline(stages, options):
    p = Pipeline(options)
    for s in stages:
        p.addStage(s)
    return p

def ensure_outputs_distinct(stages):
    raise NotImplemented

def start(stages, options):
    """Basically just looks at the arguments and exits if `--version`, `--no-execute`, &c., are specified,
    otherwise calls `execute`."""

    logger = logging.getLogger(__name__)

    ensure_outputs_distinct(stages)

    pipeline = build_pipeline(stages, options)

    if options.create_graph:
        logger.debug("Writing dot file...")
        nx.write_dot(pipeline.G, str(options.pipeline_name) + "_labeled-tree.dot")
        logger.debug("Done.")

    dirs = directories(stages)
    # TODO should provide option to turn this off if already created
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError as e:
            # FIXME check it's from the directory already existing
            pass

    execute(pipeline, options)

# The old AbstractApplication class has been removed due to its API being non-obvious.  Instead,
# we currently provide an `execute` function and some helper functions for command-line parsing.
# In the future, we could also provide higher-order functions which invert control again, although
# with a clearer interface than AbstractApplication.  This would be nice since the user wouldn't have
# to remember to add the executor option group themselves, for example, but would have to be done tastefully.

def execute(args):
    """Convert a set of stages into a pipeline and execute it with the given options,
    which should include those specified in applicationArgumentGroup, executorArgumentGroup."""
    raise NotImplemented
    #TODO dispatch on options.queue_type/options.local ...

# TODO totally broken
def reconstructCommand():
    # TODO also write down the environment
    reconstruct = ' '.join(sys.argv)
    logger.info("Command is: " + reconstruct)
    logger.info("Command version : " + self.__version__)
    fileForCommandAndVersion = self.options.pipeline_name + "-command-and-version-" + time.strftime("%d-%m-%Y-at-%H-%m-%S") + ".sh"
    pf = open(fileForCommandAndVersion, "w")
    pf.write("#!/usr/bin/env bash\n")
    pf.write("# Command version is: " + self.__version__ + "\n")
    pf.write("# Command was: \n")
    pf.write(reconstruct + '\n')
    pf.close()
 
class AbstractApplication(object):
    def _print_version(self):
        if self.options.show_version:
            print(self.__version__)
            sys.exit()
    
    # FIXME check that only one server is running with a given output directory
    def _setup_directories(self):
        """Output and backup directories setup here."""
        if not self.options.output_directory:
            self.outputDir = os.getcwd()
        else:
            self.outputDir = makedirsIgnoreExisting(self.options.output_directory)
        self.pipeline.setBackupFileLocation(self.outputDir)

       
    def start(self):
        logger.info("Calling `start`")
        self._setup_options()
        self.setup_options()
        self.options = self.parser.parse_args()
        self.args = self.options.files

        self._print_version()

        # Check to make sure some executors have been specified if we are 
        # actually going to run:
        if self.options.execute:
            noExecSpecified(self.options.num_exec)
             
        self._setup_pipeline(self.options)
        self._setup_directories()
        
        self.appName = self.setup_appName()
        self.setup_logger()

        # NB this doesn't capture environment variables
        # or contents of any config file so isn't really complete
        self.reconstructCommand()

        pbs_submit = (self.options.queue == "pbs" or \
                      self.options.queue_type == "pbs") \
                     and not self.options.local

        if (self.options.execute and not pbs_submit) or self.options.create_graph:
            logger.debug("Calling `run`")
            self.run()
            logger.debug("Calling `initialize`")
            self.pipeline.initialize()
            self.pipeline.printStages(self.options.pipeline_name)
        if not self.options.execute:
            print("Not executing the command (--no-execute is specified).\nDone.")
            return
        
        if pbs_submit:
            roq = runOnQueueingSystem(self.options, sys.argv)
            roq.createAndSubmitPbsScripts()
            logger.info("Finished submitting PBS job scripts...quitting")
            return 
                
        #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
        # if use_ns is specified, Pyro NameServer must be started. 
        logger.info("Starting pipeline daemon...")
        pipelineDaemon(self.pipeline, self.options, sys.argv[0])
        logger.info("Server has stopped.  Quitting...")

    def setup_logger(self):
        """sets logging info specific to application"""
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s %(process)d/%(threadName)s: %(message)s'
        now = datetime.now().strftime("%Y-%m-%d-at-%H:%M:%S")
        FILENAME = str(self.appName) + "-" + now + '-pid-' + str(os.getpid())  + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)
