#!/usr/bin/env python

from os.path import abspath
from optparse import OptionGroup
from datetime import date
from pydpiper.pipeline import CmdStage, Pipeline
from pydpiper.file_handling import createLogFile, createSubDir, makedirsIgnoreExisting
from pydpiper.application import AbstractApplication
from pydpiper_apps.minc_tools.registration_file_handling import RegistrationPipeFH
from pydpiper_apps.minc_tools.registration_functions import addGenRegOptionGroup, initializeInputFiles
from pydpiper_apps.minc_tools.minc_atoms import blur, mincresample, mincANTS, mincAverage
from pyminc.volumes.factory import volumeFromFile

# Probably want separate minctracc/ANTS classes. May or may not want to inherit from common base. 
class NonlinearRegistration(AbstractApplication):
    def setup_options(self):
        group = OptionGroup(self.parser, "Nonlinear registration options", 
                        "Options for performing a non-linear registration")
        group.add_option("--lsq12-avg", dest="lsq12_avg",
                      type="string", default=None,
                      help="Starting target for non-linear alignment.")
        group.add_option("--lsq12-mask", dest="lsq12_mask",
                      type="string", default="None", 
                      help="Optional mask for target.")
        self.parser.add_option_group(group)
        """Add option groups from specific modules"""
        addGenRegOptionGroup(self.parser)
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_appName(self):
        appName = "Nonlinear-registration"
        return appName

    def run(self):
        options = self.options
        args = self.args
        
        """NOTE: Some of this code below is duplicated from MBM.py. 
           We'll want to condense into a function!"""
           
        """Make main pipeline directories"""
        pipeDir = makedirsIgnoreExisting(options.pipeline_dir)
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
        nlinDirectory = createSubDir(pipeDir, pipeName + "_nlin")
        processedDirectory = createSubDir(pipeDir, pipeName + "_processed")
        
        """Initialize input files (from args) and initial target"""
        inputFiles = initializeInputFiles(args, processedDirectory, maskDir=options.mask_dir)
        initialTarget = RegistrationPipeFH(options.lsq12_avg, mask=options.lsq12_mask, basedir=nlinDirectory)
        
        """Want an if statement here for minctracc?"""
        ants = NLINANTS(inputFiles, initialTarget, nlinDirectory, 3)
        ants.iterate()
        self.pipeline.addPipeline(ants.p)
    
class NLINANTS(object):
    def __init__(self, inputArray, targetFH, nlinOutputDir, numberOfGenerations):
        self.p = Pipeline()
        """Initial inputs should be an array of fileHandlers with lastBasevol in lsq12 space"""
        self.inputs = inputArray
        """Initial target should be the file handler for the lsq12 average"""
        self.target = targetFH 
        """Output directory should be _nlin """
        self.nlinDir = nlinOutputDir
        """number of generations/iterations"""
        self.generations = numberOfGenerations
        """Empty array that we will fill with averages as we create them"""
        self.nlinAvg = [] 
        """Create the blurring resolution from the file resolution"""
        self.ANTSBlur = volumeFromFile(self.target.getLastBasevol()).separations[0]
        
        """Below are ANTS parameters for each generation that differ from defaults."""
        self.iterations = ["100x100x100x0", "100x100x100x40", "100x100x100x150"]
        self.useMask = [False, True, True]
        
        """ In init, need to add the following things:
              ANTSBlur may want to be set elsewhere, but should be set from resolution.
              Need a function that reads registration params that override above. 
        """
            
    #Possible base class for all of these modules (eg LSQ6,12,nlin) with similar structure
    # separate class for minctracc and mincANTS?
    def iterate(self):
        for i in range(self.generations):
            tblur = blur(self.target, self.ANTSBlur, gradient=True)              
            self.p.addStage(tblur)
            filesToAvg = []
            for inputFH in self.inputs:
                iblur = blur(inputFH, self.ANTSBlur, gradient=True)
                self.p.addStage(iblur)
                ma = mincANTS(inputFH, 
                              self.target, 
                              defaultDir="tmp", 
                              iterations=self.iterations[i], 
                              useMask=self.useMask[i])
                self.p.addStage(ma)
                rs = mincresample(inputFH, self.target, likeFile=self.target, defaultDir="tmp")
                #Do we need to resample any masks?
                filesToAvg.append(rs.outputFiles[0])
                self.p.addStage(rs)
            """Because we don't reset lastBasevol on each inputFH, call mincAverage with files only.
               We create fileHandler first though, so we have log directory.
               This solution seems a bit hackish--may want to modify?  
            """
            nlinOutput = abspath(self.nlinDir) + "/" + "nlin-%g.mnc" % (i+1)
            nlinFH = RegistrationPipeFH(nlinOutput, basedir=self.nlinDir)
            avgLog = createLogFile(nlinFH.logDir, nlinOutput)
            avg = mincAverage(filesToAvg, nlinOutput, logFile=avgLog)
            self.p.addStage(avg)
            """Reset target for next iteration and add to array"""
            self.target = nlinFH
            self.nlinAvg.append(nlinFH)
        
            
if __name__ == "__main__":
    
    application = NonlinearRegistration()
    application.start()
            
            
            
            