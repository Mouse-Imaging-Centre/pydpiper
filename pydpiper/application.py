from optparse import OptionParser,OptionGroup
from pydpiper.pipeline import Pipeline, pipelineDaemon
from pydpiper.queueing import runOnQueueingSystem
from pydpiper.file_handling import makedirsIgnoreExisting
from pydpiper.pipeline_executor import addExecutorOptionGroup
from datetime import datetime
from pkg_resources import get_distribution
import Pyro
import logging
import networkx as nx
import sys
import os

logger = logging.getLogger(__name__)

def addApplicationOptionGroup(parser):
    group = OptionGroup(parser,  "General application options", "General options for all pydpiper applications.")
    group.add_option("--restart", dest="restart", 
                               action="store_true",
                               help="Restart pipeline using backup files.")
    group.add_option("--output-dir", dest="output_directory",
                               type="string", default=None,
                               help="Directory where output data and backups will be saved.")
    group.add_option("--create-graph", dest="create_graph",
                               action="store_true", default=False,
                               help="Create a .dot file with graphical representation of pipeline relationships [default = %default]")
    parser.set_defaults(execute=True)
    group.add_option("--execute", dest="execute",
                               action="store_true",
                               help="Actually execute the planned commands [default]")
    group.add_option("--no-execute", dest="execute",
                               action="store_false",
                               help="Opposite of --execute")
    group.add_option("--version", dest="show_version",
                               action="store_true",
                               help="Print the version number and exit.")
    parser.add_option_group(group)

# Some sneakiness... Using the following lines, it's possible
# to add an epilog to the parser that is written to screen
# verbatim. That way in the help file you can show an example
# of what an lsq6/nlin protocol should look like.
class MyParser(OptionParser):
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
              application = ConcreteApplication()
              application.start()
    """
    def __init__(self):
        Pyro.config.PYRO_MOBILE_CODE=1 
        self.parser = MyParser()
        self.__version__ = get_distribution("pydpiper").version
    
    def _setup_options(self):
            # PydPiper options
        addExecutorOptionGroup(self.parser)
        addApplicationOptionGroup(self.parser)
    
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
        
    def start(self):
        self._setup_options()
        self.setup_options()
        
        self.options, self.args = self.parser.parse_args()
        
        self._print_version()        
        self._setup_pipeline()
        self._setup_directories()
        
        self.appName = self.setup_appName()
        self.setup_logger()
        
        if self.options.queue=="pbs":
            roq = runOnQueueingSystem(self.options, sys.argv)
            roq.createPbsScripts()
            return 
        
        if self.options.restart:
            logger.info("Restarting pipeline from pickled files.")
            self.pipeline.restart()
            self.pipeline.initialize()
            self.pipeline.printStages(self.appName)
        else:
            self.reconstructCommand()
            self.run()
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
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
        now = datetime.now()  
        FILENAME = str(self.appName) + "-" + now.strftime("%Y%m%d-%H%M%S%f") + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)

    def setup_options(self):
        """Set up the self.options option parser with options this application needs."""
        pass
    
    def run(self):
        """Run this application.
        
           """
        pass
