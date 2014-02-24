#!/usr/bin/env python

from os.path import abspath
from optparse import OptionGroup
from pydpiper.pipeline import Pipeline
from pydpiper.file_handling import createBaseName, createLogFile, removeBaseAndExtension
from pydpiper.application import AbstractApplication
from atoms_and_modules.registration_file_handling import RegistrationPipeFH
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.minc_parameters as mp
from atoms_and_modules.minc_atoms import blur, mincresample, mincANTS, mincAverage, minctracc
from atoms_and_modules.stats_tools import addStatsOptions, CalcStats
import sys
import csv
import logging

logger = logging.getLogger(__name__)

def addNlinRegOptionGroup(parser):
    """option group for the command line argument parser"""
    group = OptionGroup(parser, "Nonlinear registration options",
                        "Options for performing a non-linear registration")
    group.add_option("--target-avg", dest="target_avg",
                     type="string", default=None,
                     help="Starting target for non-linear alignment. (Often in lsq12 space)")
    group.add_option("--target-mask", dest="target_mask",
                     type="string", default=None,
                     help="Optional mask for target.")
    group.add_option("--nlin-protocol", dest="nlin_protocol",
                     type="string", default=None,
                     help="Can optionally specify a nonlinear protocol that is different from defaults. Default is None.")
    parser.add_option_group(group)
    
def finalGenerationFileNames(inputFH):
    """Set up and return filenames for final nlin generation, since we don't want to use defaults here.
       The naming of the final resampled files/transforms will be the same regardless of registration
       protocol (minctracc vs mincANTS) or number of generations. 
    """
    registerDir = inputFH.setOutputDirectory("transforms")
    registerFileName = removeBaseAndExtension(inputFH.basename) + "-final-nlin.xfm"
    registerOutput = createBaseName(registerDir, registerFileName)
    resampleDir = inputFH.setOutputDirectory("resampled")
    resampleFileName = removeBaseAndExtension(inputFH.basename) + "-resampled-final-nlin.mnc"
    resampleOutput = createBaseName(resampleDir, resampleFileName)
    return (registerOutput, resampleOutput)

def initNLINModule(inputFiles, initialTarget, nlinDir, nlin_protocol, reg_method):
    if reg_method=="mincANTS":
        nlinModule = NLINANTS(inputFiles, initialTarget, nlinDir, nlin_protocol)
    elif reg_method=="minctracc":
        nlinModule = NLINminctracc(inputFiles, initialTarget, nlinDir, nlin_protocol)
    else:
        logger.error("Incorrect registration method specified: " + reg_method)
        sys.exit()
    return nlinModule

class NonlinearRegistration(AbstractApplication):
    """
        This class performs an iterative non-linear registration between one or more files
        and a single target. It currently supports two non-linear registration programs:
        minctracc (mni_autoreg, McGill) and mincANTS (Advanced Normalization Tools, U Penn). 
        Optionally, statistics may be calculated at the end of the registration.  
        
        Source:
            One or more files with or without a mask. 
        
        Target:
            Single input file with or without a mask. 
            
        This application was designed to be called on inputs and targets that have already 
        undergone an affine registration. (e.g. translations, rotations, scales and shears)
        This is sometimes referred to as LSQ12. However, this class is generic enough that 
        this iterative alignment could be called on sources and a target that are not in
        LSQ12 space, although it is likely that the alignment will be less successful. 
          
    """
    def setup_options(self):
        """Add option groups from specific modules"""
        rf.addGenRegOptionGroup(self.parser)
        addNlinRegOptionGroup(self.parser)
        addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_appName(self):
        appName = "Nonlinear-registration"
        return appName

    def run(self):
        options = self.options
        args = self.args
        
        # Setup output directories for non-linear registration.        
        dirs = rf.setupDirectories(self.outputDir, options.pipeline_name, module="NLIN")
        
        """Initialize input files (from args) and initial target"""
        inputFiles = rf.initializeInputFiles(args, dirs.processedDir, maskDir=options.mask_dir)
        if options.target_avg: 
            initialTarget = RegistrationPipeFH(options.target_avg, 
                                               mask=options.target_mask, 
                                               basedir=dirs.nlinDir)
        else:
            # if no target is specified, create an average from the inputs
            targetName = abspath(self.outputDir) + "/" + "initial-target.mnc" 
            initialTarget = RegistrationPipeFH(targetName, basedir=self.outputDir)
            avg = mincAverage(inputFiles, 
                              initialTarget, 
                              output=targetName,
                              defaultDir=self.outputDir)
            self.pipeline.addStage(avg)
        
        """Based on options.reg_method, register with minctracc or mincANTS"""
        nlinModule = initNLINModule(inputFiles, 
                                    initialTarget, 
                                    dirs.nlinDir, 
                                    options.nlin_protocol, 
                                    options.reg_method)
        nlinModule.iterate()
        self.pipeline.addPipeline(nlinModule.p)
        self.nlinAverages = nlinModule.nlinAverages
            
        """Calculate statistics between final nlin average and individual mice"""
        if options.calc_stats:
            """Get blurs from command line option and put into array"""
            blurs = []
            for i in options.stats_kernels.split(","):
                blurs.append(float(i))
            """Choose final average from array of nlin averages"""
            numGens = len(self.nlinAverages)
            finalNlin = self.nlinAverages[numGens-1]
            """For each input file, calculate statistics from finalNlin to input"""
            for inputFH in inputFiles:
                stats = CalcStats(inputFH, finalNlin, blurs, inputFiles)
                stats.fullStatsCalc()
                self.pipeline.addPipeline(stats.p)

class NLINBase(object):
    """
        This is the parent class for any iterative non-linear registration. 
        
        Subclasses should extend the following methods:
            addBlurStage()
            regAndResample()
        
    """
    def __init__(self, inputArray, targetFH, nlinOutputDir):
        self.p = Pipeline()
        """Initial inputs should be an array of fileHandlers with lastBasevol in lsq12 space"""
        self.inputs = inputArray
        """Initial target should be the file handler for the lsq12 average"""
        self.target = targetFH 
        """Output directory should be _nlin """
        self.nlinDir = nlinOutputDir
        """Empty array that we will fill with averages as we create them"""
        self.nlinAverages = [] 
        """Create the blurring resolution from the file resolution"""
        try: # the attempt to access the minc volume will fail if it doesn't yet exist at pipeline creation
            self.fileRes = rf.getFinestResolution(self.target)
        except: 
            # if it indeed failed, get resolution from the original file specified for 
            # one of the input files, which should exist. 
            # Can be overwritten by the user through specifying a nonlinear protocol.
            self.fileRes = rf.getFinestResolution(self.inputs[0].inputFileName)
        
        # Create new nlin group for each input prior to registration
        for i in range(len(self.inputs)):
            self.inputs[i].newGroup(groupName="nlin")
    
    def addBlurStage(self):
        """
            Add blurs to pipeline. Because blurs are handled differently by
            parameter arrays in minctracc and mincANTS subclasses, they are added
            to the pipeline via function call. 
        """
        pass
    
    def regAndResample(self):
        """Registration and resampling calls"""
        pass
    
    def iterate(self):
        for i in range(self.generations):
            nlinOutput = abspath(self.nlinDir) + "/" + "nlin-%g.mnc" % (i+1)
            nlinFH = RegistrationPipeFH(nlinOutput, mask=self.target.getMask(), basedir=self.nlinDir)
            self.addBlurStage(self.target, i)
            filesToAvg = []
            for inputFH in self.inputs:
                self.addBlurStage(inputFH, i) 
                self.regAndResample(inputFH, i, filesToAvg, nlinFH)
            
            """Because we don't reset lastBasevol on each inputFH, call mincAverage with files only.
               We create fileHandler first though, so we have log directory.
               This solution seems a bit hackish--may want to modify?  
               Additionally, we are currently using the full RegistrationPipeFH class, but ultimately
               we'll want to create a third class that is somewhere between a full and base class. 
            """
            logBase = removeBaseAndExtension(nlinOutput)
            avgLog = createLogFile(nlinFH.logDir, logBase)
            avg = mincAverage(filesToAvg, nlinOutput, logFile=avgLog)
            self.p.addStage(avg)
            """Reset target for next iteration and add to array"""
            self.target = nlinFH
            self.nlinAverages.append(nlinFH)
            """Create a final nlin group to add to the inputFH.
               lastBasevol = by default, will grab the lastBasevol used in these calculations (e.g. lsq12)
               setLastXfm between final nlin average and inputFH will be set for stats calculations.
            """
            if i == (self.generations -1):
                for inputFH in self.inputs:
                    """NOTE: The last xfm being set below is NOT the result of a registration between
                       inputFH and nlinFH, but rather is the output transform from the previous generation's
                       average."""
                    finalXfm = inputFH.getLastXfm(self.nlinAverages[self.generations-2])
                    inputFH.newGroup(groupName="final")
                    inputFH.setLastXfm(nlinFH, finalXfm)
    
class NLINANTS(NLINBase):
    """
        This class does an iterative non-linear registration using the mincANTS
        registration protocol. The default number of generations is three. 
    """
    def __init__(self, inputArray, targetFH, nlinOutputDir, nlin_protocol=None):
        NLINBase.__init__(self, inputArray, targetFH, nlinOutputDir)
        
        """Setup parameters, either as defaults, or read from a .csv"""
        params = mp.setMincANTSParams(self.fileRes, nlin_protocol)
    
        self.blurs = params.blurs
        self.gradient = params.gradient
        self.similarityMetric = params.similarityMetric
        self.weight = params.weight
        self.radiusHisto = params.radiusHisto
        self.transformationModel = params.transformationModel
        self.regularization = params.regularization
        self.iterations = params.iterations
        self.useMask = params.useMask
        self.generations = params.generations
    
    def addBlurStage(self, FH, i):
        for j in self.blurs[i]:
            if j != -1:
                tblur = blur(FH, j, gradient=True)              
                self.p.addStage(tblur)
    
    def regAndResample(self, inputFH, i, filesToAvg, nlinFH):
        """For last generation, override default output names. 
           Note that the defaultDir specified in the mincANTS call
           is ignored in this instance. """
        if i == (self.generations -1):
            registerOutput, resampleOutput = finalGenerationFileNames(inputFH)
        else:
            registerOutput = None
            resampleOutput = None
        ma = mincANTS(inputFH, 
                      self.target, 
                      output=registerOutput,
                      defaultDir="tmp",
                      blur=self.blurs[i],
                      gradient=self.gradient[i], 
                      similarity_metric=self.similarityMetric[i],
                      weight=self.weight[i], 
                      iterations=self.iterations[i],
                      radius_or_histo=self.radiusHisto[i],
                      transformation_model = self.transformationModel[i],
                      regularization=self.regularization[i],
                      useMask=self.useMask[i])
        self.p.addStage(ma)
        rs = mincresample(inputFH, 
                          self.target, 
                          likeFile=self.target, 
                          output=resampleOutput, 
                          defaultDir="tmp", 
                          argArray=["-sinc"])
        #Do we need to resample any masks?
        filesToAvg.append(rs.outputFiles[0])
        self.p.addStage(rs)

class NLINminctracc(NLINBase):
    """
        This class does an iterative non-linear registration using the minctracc
        registration protocol. Default number of generations is 6. 
    """
    def __init__(self, inputArray, targetFH, nlinOutputDir, nlin_protocol=None):
        NLINBase.__init__(self, inputArray, targetFH, nlinOutputDir)
        
        """Setup parameters, either as defaults, or read from a .csv"""
        params = mp.setNlinMinctraccParams(self.fileRes, nlin_protocol=nlin_protocol)
        self.blurs = params.blurs
        self.stepSize = params.stepSize
        self.iterations = params.iterations
        self.simplex = params.simplex
        self.useGradient = params.useGradient
        self.optimization = params.optimization
        self.generations = params.generations
    
    
    def addBlurStage(self, FH, i):
        tblur = blur(FH, self.blurs[i], gradient=True)              
        self.p.addStage(tblur)
    
    def regAndResample(self, inputFH, i, filesToAvg, nlinFH):
        """For last generation, override default output names. 
           Note that the defaultDir specified in the minctracc call
           is ignored in this instance. """
        if i == (self.generations -1):
            if not self.useGradient[i]:
                firstRegOutput, resampleOutput = finalGenerationFileNames(inputFH)
            else:
                firstRegOutput = None
                registerOutput, resampleOutput = finalGenerationFileNames(inputFH)
        else:
            firstRegOutput = None
            registerOutput = None
            resampleOutput = None
        """If self.useGradient is True, then we call minctracc twice: once
            with a gradient and once without. Otherwise, we call only once
            without a gradient. """
        mta = minctracc(inputFH, 
                        self.target, 
                        defaultDir="tmp",
                        output=firstRegOutput, 
                        blur=self.blurs[i],
                        gradient=False,
                        iterations=self.iterations[i],
                        step=self.stepSize[i],
                        weight=0.8, 
                        stiffness=0.98,
                        similarity=0.8,
                        simplex=self.simplex[i])
        self.p.addStage(mta)
        if self.useGradient[i]:
            mtb = minctracc(inputFH, 
                            self.target, 
                            output=registerOutput,
                            defaultDir="tmp", 
                            blur=self.blurs[i],
                            gradient=True,
                            iterations=self.iterations[i],
                            step=self.stepSize[i],
                            weight=0.8, 
                            stiffness=0.98,
                            similarity=0.8,
                            simplex=self.simplex[i])
            self.p.addStage(mtb)
            """Need to set last xfm so that next generation will use it as the input transform"""
            inputFH.setLastXfm(nlinFH, mtb.outputFiles[0])
            rs = mincresample(inputFH, 
                                self.target, 
                                likeFile=self.target, 
                                output=resampleOutput,
                                defaultDir="tmp", 
                                argArray=["-sinc"])
            #Do we need to resample any masks?
            filesToAvg.append(rs.outputFiles[0])
            self.p.addStage(rs)
            
if __name__ == "__main__":
    
    application = NonlinearRegistration()
    application.start()
            
            
            
            
