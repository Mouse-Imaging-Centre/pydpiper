#!/usr/bin/env python

from os.path import abspath
from optparse import OptionGroup
from datetime import date
from pydpiper.pipeline import CmdStage, Pipeline
from pydpiper.file_handling import createLogFile, createSubDir, makedirsIgnoreExisting, removeBaseAndExtension
from pydpiper.application import AbstractApplication
from pydpiper_apps.minc_tools.registration_file_handling import RegistrationPipeFH
from pydpiper_apps.minc_tools.registration_functions import addGenRegOptionGroup, initializeInputFiles
from pydpiper_apps.minc_tools.minc_atoms import blur, mincresample, mincANTS, mincAverage, mincAverageDisp, minctracc
from pydpiper_apps.minc_tools.stats_tools import addStatsOptions, CalcStats
from pyminc.volumes.factory import volumeFromFile
import sys
import logging

logger = logging.getLogger(__name__)

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
        addStatsOptions(self.parser)
        
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
        
        """Based on cmdline option, register with minctracc or mincANTS"""
        if options.reg_method=="mincANTS":
            ants = NLINANTS(inputFiles, initialTarget, nlinDirectory, 3)
            ants.iterate()
            self.pipeline.addPipeline(ants.p)
            self.nlinAvg = ants.nlinAvg
        elif options.reg_method == "minctracc":
            tracc = NLINminctracc(inputFiles, initialTarget, nlinDirectory, 6)
            tracc.iterate()
            self.pipeline.addPipeline(tracc.p)
            self.nlinAvg = tracc.nlinAvg
        else:
            logger.error("Incorrect registration method specified: " + options.reg_method)
            sys.exit()
            
        """Calculate statistics between final nlin average and individual mice"""
        if options.calc_stats:
            """Get blurs from command line option and put into array"""
            blurs = []
            for i in options.stats_kernels.split(","):
                blurs.append(float(i))
            """Choose final average from array of nlin averages"""
            numGens = len(self.nlinAvg)
            finalNlin = self.nlinAvg[numGens-1]
            """For each input file, calculate statistics"""
            for inputFH in inputFiles:
                stats = CalcStats(inputFH, finalNlin, blurs, inputFiles)
                stats.fullStatsCalc()
                self.p.addPipeline(stats.p)
            
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
        
        """Below are ANTS parameters for each generation that differ from defaults.
           Defaults in mincANTS class currently derived from SyN[0.2] protocol.
           These parameters are for the standard mincANTS protocol in 
           the current iteration of build-model.
        """
        self.iterations = ["100x100x100x0", "100x100x100x20", "100x100x100x50"]
        self.transformationModel = ["SyN[0.5]", "SyN[0.4]", "SyN[0.4]"]
        self.regularization = ["Gauss[5,1]", "Gauss[5,1]", "Gauss[5,1]"]
        self.useMask = [False, True, True]
        
        
        """ In init, need to add the following things:
              ANTSBlur may want to be set elsewhere, but should be set from resolution.
              Need a function that reads registration params that override above. 
        """
            
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
                              transformation_model = self.transformationModel[i],
                              regularization=self.regularization[i],
                              useMask=self.useMask[i])
                self.p.addStage(ma)
                rs = mincresample(inputFH, self.target, likeFile=self.target, defaultDir="tmp", argArray=["-sinc"])
                #Do we need to resample any masks?
                filesToAvg.append(rs.outputFiles[0])
                self.p.addStage(rs)
            
            """Because we don't reset lastBasevol on each inputFH, call mincAverage with files only.
               We create fileHandler first though, so we have log directory.
               This solution seems a bit hackish--may want to modify?  
               Additionally, we are currently using the full RegistrationPipeFH class, but ultimately
               we'll want to create a third class that is somewhere between a full and base class. 
            """
            nlinOutput = abspath(self.nlinDir) + "/" + "nlin-%g.mnc" % (i+1)
            nlinFH = RegistrationPipeFH(nlinOutput, mask=self.target.getMask(), basedir=self.nlinDir)
            logBase = removeBaseAndExtension(nlinOutput)
            avgLog = createLogFile(nlinFH.logDir, logBase)
            avg = mincAverage(filesToAvg, nlinOutput, logFile=avgLog)
            self.p.addStage(avg)
            """Reset target for next iteration and add to array"""
            self.target = nlinFH
            self.nlinAvg.append(nlinFH)

class NLINminctracc(object):
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
        """Blur and step size parameters will be created from the file resolution."""
        self.fileRes = volumeFromFile(self.target.getLastBasevol()).separations[0]    
        
        """ 
            Default minctracc parameters for 6 generations. 
            As with the NLINANTS class, we need a way to set these generally. 
        """
        self.blurs = [self.fileRes*5.0, self.fileRes*(10.0/3.0), self.fileRes*(10.0/3.0),
                      self.fileRes*(10.0/3.0), self.fileRes*(5.0/3.0), self.fileRes]
        self.steps = [self.fileRes*(35.0/3.0), self.fileRes*10.0, self.fileRes*(25.0/3.0),
                      self.fileRes*4.0, self.fileRes*2.0, self.fileRes]
        self.iterations = [20,6,8,8,8,8]
        self.simplex = [5,2,2,2,2,2]
        
    def iterate(self):
        for i in range(self.generations):
            """Create file handler for nlin average for each generation"""
            nlinOutput = abspath(self.nlinDir) + "/" + "nlin-%g.mnc" % (i+1)
            nlinFH = RegistrationPipeFH(nlinOutput, mask=self.target.getMask(), basedir=self.nlinDir)
            tblur = blur(self.target, self.blurs[i], gradient=True)              
            self.p.addStage(tblur)
            filesToAvg = []
            for inputFH in self.inputs:
                iblur = blur(inputFH, self.blurs[i], gradient=True)
                self.p.addStage(iblur)
                """Run two stages: once with the blur and once with the gradient"""
                mta = minctracc(inputFH, 
                                self.target, 
                                defaultDir="tmp", 
                                blur=self.blurs[i],
                                gradient=False,
                                iterations=self.iterations[i],
                                step=self.steps[i],
                                weight=0.8, 
                                stiffness=0.98,
                                similarity=0.8,
                                simplex=self.simplex[i])
                self.p.addStage(mta)
                mtb = minctracc(inputFH, 
                                self.target, 
                                defaultDir="tmp", 
                                blur=self.blurs[i],
                                gradient=True,
                                iterations=self.iterations[i],
                                step=self.steps[i],
                                weight=0.8, 
                                stiffness=0.98,
                                similarity=0.8,
                                simplex=self.simplex[i])
                self.p.addStage(mtb)
                """Need to set last xfm so that next generation will use it as the input transform"""
                inputFH.setLastXfm(nlinFH, mtb.outputFiles[0])
                rs = mincresample(inputFH, self.target, likeFile=self.target, defaultDir="tmp", argArray=["-sinc"])
                #Do we need to resample any masks?
                filesToAvg.append(rs.outputFiles[0])
                self.p.addStage(rs)
                
            """Because we don't reset lastBasevol on each inputFH, call mincAverage with files only.
               File handler has been created above. 
               This solution seems a bit hackish--may want to modify?  
            """
            logBase = removeBaseAndExtension(nlinOutput)
            avgLog = createLogFile(nlinFH.logDir, logBase)
            avg = mincAverage(filesToAvg, nlinOutput, logFile=avgLog)
            self.p.addStage(avg)
            """Reset target for next iteration and add to array"""
            self.target = nlinFH
            self.nlinAvg.append(nlinFH)
        
            
if __name__ == "__main__":
    
    application = NonlinearRegistration()
    application.start()
            
            
            
            