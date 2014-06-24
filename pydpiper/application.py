from optparse import OptionParser,OptionGroup
from pydpiper.pipeline import Pipeline, pipelineDaemon
from pydpiper.queueing import runOnQueueingSystem
from pydpiper.file_handling import makedirsIgnoreExisting
from datetime import datetime
from pkg_resources import get_distribution
import Pyro
import logging
import networkx as nx
import sys
import os

logger = logging.getLogger(__name__)

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
        basic_group = OptionGroup(self.parser,  "Basic execution control",
                                  "Options controlling how and where the code is run.")
        basic_group.add_option("--uri-file", dest="urifile",
                               type="string", default=None,
                               help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
        basic_group.add_option("--use-ns", dest="use_ns",
                               action="store_true",
                               help="Use the Pyro NameServer to store object locations")
        basic_group.add_option("--create-graph", dest="create_graph",
                               action="store_true", default=False,
                               help="Create a .dot file with graphical representation of pipeline relationships [default = %default]")
        basic_group.add_option("--num-executors", dest="num_exec", 
                               type="int", default=0, 
                               help="Launch executors automatically without having to run pipeline_excutor.py independently.")
        basic_group.add_option("--time", dest="time", 
                               type="string", default="2:00:00:00", 
                               help="Wall time to request for each executor in the format dd:hh:mm:ss")
        basic_group.add_option("--proc", dest="proc", 
                               type="int", default=8,
                               help="Number of processes per executor. Default is 8. Also sets max value for processor use per executor.")
        basic_group.add_option("--mem", dest="mem", 
                               type="float", default=16,
                               help="Total amount of requested memory. Default is 16G.")
        basic_group.add_option("--ppn", dest="ppn", 
                               type="int", default=8,
                               help="Number of processes per node. Default is 8. Used when --queue=pbs")
        basic_group.add_option("--queue", dest="queue", 
                               type="string", default=None,
                               help="Use specified queueing system to submit jobs. Default is None.")
        basic_group.add_option("--sge-queue-opts", dest="sge_queue_opts", 
                               type="string", default=None,
                               help="For --queue=sge, allows you to specify different queues. If not specified, default is used.")
        basic_group.add_option("--time-to-seppuku", dest="time_to_seppuku", 
                               type="int", default=None,
                               help="The number of minutes an executor is allowed to continuously sleep, i.e. wait for an available job, while active on a compute node/farm before it kills itself due to resource hogging. (Default=None, which means it will not kill itself for this reason)")
        basic_group.add_option("--time-to-accept-jobs", dest="time_to_accept_jobs", 
                               type="int", default=None,
                               help="The number of minutes after which an executor will not accept new jobs anymore. This can be useful when running executors on a batch system where other (competing) jobs run for a limited amount of time. The executors can behave in a similar way by given them a rough end time. (Default=None, which means that the executor will always accept jobs)")
        basic_group.add_option("--restart", dest="restart", 
                               action="store_true",
                               help="Restart pipeline using backup files.")
        basic_group.add_option("--output-dir", dest="output_directory",
                               type="string", default=None,
                               help="Directory where output data and backups will be saved.")
        self.parser.set_defaults(execute=True)
        basic_group.add_option("--execute", dest="execute",
                               action="store_true",
                               help="Actually execute the planned commands [default]")
        basic_group.add_option("--no-execute", dest="execute",
                               action="store_false",
                               help="Opposite of --execute")
        basic_group.add_option("--version", dest="show_version",
                               action="store_true",
                               help="Print the version number and exit.")
        self.parser.add_option_group(basic_group)
    
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
