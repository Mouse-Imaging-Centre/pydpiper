#!/usr/bin/env python

def pytest_funcarg__setupopts(request):
    return OptsSetup(request)

def pytest_addoption(parser):
    parser.addoption("--uri-file", dest="urifile",
                     type="string", default=None,
                     help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    parser.addoption("--use-ns", dest="use_ns",
                     action="store_true",
                     help="Use the Pyro NameServer to store object locations")
    parser.addoption("--create-graph", dest="create_graph",
                     action="store_true",
                     help="Create a .dot file with graphical representation of pipeline relationships")
    parser.addoption("--num-executors", dest="num_exec", 
                     type="int", default=0, 
                     help="Launch executors automatically without having to run pipeline_excutor.py independently.")
    parser.addoption("--time", dest="time", 
                     type="string", default="2:00:00:00", 
                     help="Wall time to request for each executor in the format dd:hh:mm:ss")
    parser.addoption("--proc", dest="proc", 
                     type="int", default=8,
                     help="Number of processes per executor. Default is 8. Also sets max value for processor use per executor. Overridden if --num-executors not specified.")
    parser.addoption("--mem", dest="mem", 
                     type="float", default=16,
                     help="Total amount of requested memory. Default is 8G. Overridden if --num-executors not specified.")
    parser.addoption("--queue", dest="queue", 
                     type="string", default=None,
                     help="Use specified queueing system to submit jobs. Default is None.")
    parser.addoption("--restart", dest="restart", 
                     action="store_true",
                     help="Restart pipeline using backup files.")
    parser.addoption("--backup-dir", dest="backup_directory",
                     type="string", default=".pipeline-backup",
                     help="Directory where this pipeline backup should be stored.")   
    
class OptsSetup():
    def __init__(self, request):
        self.config = request.config
        
    def returnAllOptions(self):
        return self.config.option
        
    def getNumExecutors(self):
        return self.config.option.num_exec
    
    def getTime(self):
        return self.config.option.time
    
    def getProc(self):
        return self.config.option.proc
    
    def getMem(self):
        return self.config.option.mem
    
    def getQueue(self):
        return self.config.option.queue
    
    def getRestart(self):
        return self.config.option.restart
    
    def getBackupDir(self):
        return self.config.option.backup_directory
    
    def returnSampleArgs(self):
        sampleArgArray = ["TestProgName.py", "img_A.mnc", "img_B.mnc"]
        return sampleArgArray