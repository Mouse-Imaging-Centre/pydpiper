from argparse import ArgumentParser
from pydpiper.pipeline import Pipeline, pipelineDaemon
from pydpiper.queueing import runOnQueueingSystem
from pydpiper.file_handling import makedirsIgnoreExisting
from pydpiper.pipeline_executor import addExecutorArgumentGroup, noExecSpecified
from datetime import datetime
from pkg_resources import get_distribution
import logging
import networkx as nx
import sys
import os

logger = logging.getLogger(__name__)

def addApplicationArgumentGroup(parser):
    group = parser.add_argument_group("General application options", "General options for all pydpiper applications.")
    group.add_argument("--restart", dest="restart", 
                               action="store_false", default=True,
                               help="Restart pipeline using backup files. [default = %default]")
    group.add_argument("--no-restart", dest="restart", 
                               action="store_false", help="Opposite of --restart")
    group.add_argument("--output-dir", dest="output_directory",
                               type=str, default=None,
                               help="Directory where output data and backups will be saved.")
    group.add_argument("--create-graph", dest="create_graph",
                               action="store_true", default=False,
                               help="Create a .dot file with graphical representation of pipeline relationships [default = %default]")
    parser.set_defaults(execute=True)
    parser.set_defaults(verbose=False)
    group.add_argument("--execute", dest="execute",
                               action="store_true",
                               help="Actually execute the planned commands [default = %default]")
    group.add_argument("--no-execute", dest="execute",
                               action="store_false",
                               help="Opposite of --execute")
    group.add_argument("--version", dest="show_version",
                               action="store_true",
                               help="Print the version number and exit.")
    group.add_argument("--verbose", dest="verbose",
                               action="store_true",
                               help="Be verbose in what is printed to the screen [default = %default]")
    group.add_argument("--no-verbose", dest="verbose",
                               action="store_false",
                               help="Opposite of --verbose [default]")
    group.add_argument("files", type=str, nargs='+', metavar='file',
                        help='Files to process')

# Some sneakiness... Using the following lines, it's possible
# to add an epilog to the parser that is written to screen
# verbatim. That way in the help file you can show an example
# of what an lsq6/nlin protocol should look like.
class MyParser(ArgumentParser):
    def format_epilog(self, formatter):
        if not self.epilog:
            self.epilog = ""
        return self.epilog

class AbstractApplication(object):
    """Framework class for writing applications for PydPiper. 
    
       This class defines the default behaviour for accepting common command-line options, and executing the application
       under various queueing systems. 
       
       Subclasses should extend the following methods:
           setup_appName()
           setup_logger() [optional, default method is defined here]
           setup_options()
           run()
    
       Usage: 
          class MyApplication(AbstractApplication):
                ... 
           
          if __name__ == "__main__":
              application = MyApplication()
              application.start()
    """
    def __init__(self):
        self.parser = MyParser()
        self.__version__ = get_distribution("pydpiper").version
    
    def _setup_options(self):
            # PydPiper options
        addExecutorArgumentGroup(self.parser)
        addApplicationArgumentGroup(self.parser)
    
    def _print_version(self):
        if self.options.show_version:
            print self.__version__
            sys.exit()
    
    def _setup_pipeline(self):
        self.pipeline = Pipeline()
        
    def _setup_directories(self):
        """Output and backup directories setup here."""
        if not self.options.output_directory:
            self.outputDir = os.getcwd()
        else:
            self.outputDir = makedirsIgnoreExisting(self.options.output_directory)
        self.pipeline.setBackupFileLocation(self.outputDir)
    
    def reconstructCommand(self):    
        reconstruct = ""
        for i in range(len(sys.argv)):
            reconstruct += sys.argv[i] + " "
        logger.info("Command is: " + reconstruct)
        logger.info("Command version : " + self.__version__)
        # also, because this is probably a better file for it (also has similar
        # naming conventions as the pipeline-stages.txt file:
        fileForCommandAndVersion = os.path.abspath(os.curdir + "/" + self.appName + "-pipeline-command-and-version.txt")
        pf = open(fileForCommandAndVersion, "w")
        pf.write("Command is: " + reconstruct + "\n")
        pf.write("Command version is: " + self.__version__ + "\n")
        pf.close()
        
    def start(self):
        self._setup_options()
        self.setup_options()
        
        self.options = self.parser.parse_args()
        self.args = self.options.files
        
        self._print_version()   
        
        #Check to make sure some executors have been specified. 
        noExecSpecified(self.options.num_exec)
             
        self._setup_pipeline()
        self._setup_directories()
        
        self.appName = self.setup_appName()
        self.setup_logger()
        
        if self.options.scinet or self.options.queue == "pbs" or self.options.queue_type == "pbs":
            roq = runOnQueueingSystem(self.options, sys.argv)
            roq.createAndSubmitPbsScripts()
            logger.info("Finished submitting PBS job scripts...quitting")
            return 

        self.reconstructCommand()
        logger.debug("Calling `run`")
        self.run()
        logger.debug("Calling `initialize`")
        self.pipeline.initialize()
        self.pipeline.printStages(self.appName)
                            
        if self.options.create_graph:
            logger.debug("Writing dot file...")
            nx.write_dot(self.pipeline.G, "labeled-tree.dot")
            logger.debug("Done.")
                
        if not self.options.execute:
            print "Not executing the command (--no-execute is specified).\nDone."
            return
        
        #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
        # if use_ns is specified, Pyro NameServer must be started. 
        logger.info("Starting pipeline daemon...")
        pipelineDaemon(self.pipeline, self.options, sys.argv[0])
        logger.info("Server has stopped.  Quitting...")

    def setup_appName(self):
        """sets the name of the application"""
        pass

    def setup_logger(self):
        """sets logging info specific to application"""
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s %(process)d/%(threadName)s: %(message)s'
        now = datetime.now().strftime("%Y-%m-%d-at-%H:%M:%S")
        FILENAME = str(self.appName) + "-" + now + '-pid-' + str(os.getpid())  + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)

    def setup_options(self):
        """Set up the self.options option parser with options this application needs."""
        pass
    
    def run(self):
        """Run this application.
        
           """
        pass
