#!/usr/bin/env python

from os.path import abspath, isfile
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
    group.add_option("--registration-method", dest="reg_method",
                      default="mincANTS", type="string",
                      help="Specify whether to use minctracc or mincANTS for non-linear registrations. "
                           "Default is mincANTS (and minctracc when running MAGeT.py).")
    group.add_option("--nlin-protocol", dest="nlin_protocol",
                     type="string", default=None,
                     help="Can optionally specify a registration protocol that is different from defaults. "
                     "Parameters must be specified as in either or the following examples: \n"
                     "applications_testing/test_data/minctracc_example_nlin_protocol.csv \n"
                     "applications_testing/test_data/mincANTS_example_nlin_protocol.csv \n"
                     "Default is None.")
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
        
        #Initialize input files (from args)
        inputFiles = rf.initializeInputFiles(args, dirs.processedDir, maskDir=options.mask_dir)
        
        #Setup initial target and run iterative non-linear registration
        if options.target_avg:
            createAvg=False
        else:
            createAvg=True
        nlinObj = initializeAndRunNLIN(self.outputDir,
                                       inputFiles,
                                       dirs.nlinDir,
                                       avgPrefix=options.pipeline_name,
                                       createAvg=createAvg,
                                       targetAvg=options.target_avg,
                                       targetMask=options.target_mask,
                                       nlin_protocol=options.nlin_protocol,
                                       reg_method=options.reg_method)
        
        self.pipeline.addPipeline(nlinObj.p)
        self.nlinAverages = nlinObj.nlinAverages
            
        """Calculate statistics between final nlin average and individual mice"""
        if options.calc_stats:
            """Choose final average from array of nlin averages"""
            finalNlin = self.nlinAverages[-1]
            """For each input file, calculate statistics from finalNlin to input"""
            for inputFH in inputFiles:
                stats = CalcStats(inputFH, finalNlin, options.stats_kernels)
                self.pipeline.addPipeline(stats.p)

class initializeAndRunNLIN(object):
    """Class to setup target average (if needed), 
       instantiate correct version of NLIN class,
       and run NLIN registration."""
    def __init__(self, 
                  targetOutputDir, #Output directory for files related to initial target (often _lsq12)
                  inputFiles, 
                  nlinDir, 
                  avgPrefix, #Prefix for nlin-1.mnc, ... nlin-k.mnc 
                  createAvg=True, #True=call mincAvg, False=targetAvg already exists
                  targetAvg=None, #Optional path to initial target - passing name does not guarantee existence
                  targetMask=None, #Optional path to mask for initial target
                  nlin_protocol=None,
                  reg_method=None):
        self.p = Pipeline()
        self.targetOutputDir = targetOutputDir
        self.inputFiles = inputFiles
        self.nlinDir = nlinDir
        self.avgPrefix = avgPrefix
        self.createAvg = createAvg
        self.targetAvg = targetAvg
        self.targetMask = targetMask
        self.nlin_protocol = nlin_protocol
        self.reg_method = reg_method
        
        # setup initialTarget (if needed) and initialize non-linear module
        self.setupTarget()
        self.initNlinModule()
        
        #iterate through non-linear registration and setup averages
        self.nlinModule.iterate()
        self.p.addPipeline(self.nlinModule.p)
        self.nlinAverages = self.nlinModule.nlinAverages
        self.nlinParams = self.nlinModule.nlinParams
        
    def setupTarget(self):
        if self.targetAvg:
            if isinstance(self.targetAvg, str): 
                self.initialTarget = RegistrationPipeFH(self.targetAvg, 
                                                        mask=self.targetMask, 
                                                        basedir=self.targetOutputDir)
                self.outputAvg = self.targetAvg
            elif isinstance(self.targetAvg, RegistrationPipeFH):
                self.initialTarget = self.targetAvg
                self.outputAvg = self.targetAvg.getLastBasevol()
                if not self.initialTarget.getMask():
                    if self.targetMask:
                        self.initialTarget.setMask(self.targetMask)
            else:
                print "You have passed a target average that is neither a string nor a file handler: " + str(self.targetAvg)
                print "Exiting..."
        else:
            self.targetAvg = abspath(self.targetOutputDir) + "/" + "initial-target.mnc" 
            self.initialTarget = RegistrationPipeFH(self.targetAvg, 
                                                    mask=self.targetMask, 
                                                    basedir=self.targetOutputDir)
            self.outputAvg = self.targetAvg
        if self.createAvg:
            avg = mincAverage(self.inputFiles, 
                              self.initialTarget, 
                              output=self.outputAvg,
                              defaultDir=self.targetOutputDir)
            self.p.addStage(avg)
            
    def initNlinModule(self):
        if self.reg_method=="mincANTS":
            self.nlinModule = NLINANTS(self.inputFiles, self.initialTarget, self.nlinDir, self.avgPrefix, self.nlin_protocol)
        elif self.reg_method=="minctracc":
            self.nlinModule = NLINminctracc(self.inputFiles, self.initialTarget, self.nlinDir, self.avgPrefix, self.nlin_protocol)
        else:
            logger.error("Incorrect registration method specified: " + self.reg_method)
            sys.exit()

class NLINBase(object):
    """
        This is the parent class for any iterative non-linear registration. 
        
        Subclasses should extend the following methods:
            addBlurStage()
            regAndResample()
        
    """
    def __init__(self, inputArray, targetFH, nlinOutputDir, avgPrefix, nlin_protocol):
        self.p = Pipeline()
        """Initial inputs should be an array of fileHandlers with lastBasevol in lsq12 space"""
        self.inputs = inputArray
        """Initial target should be the file handler for the lsq12 average"""
        self.target = targetFH 
        """Output directory should be _nlin """
        self.nlinDir = nlinOutputDir
        """Prefix to pre-pend to averages at each generation"""
        self.avgPrefix = avgPrefix
        """Empty array that we will fill with averages as we create them"""
        self.nlinAverages = [] 
        """Create the blurring resolution from the file resolution"""
        if nlin_protocol==None:
            self.fileRes = rf.returnFinestResolution(self.inputs[0]) 
        else:
            self.fileRes = None
        
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
            outputName = "nlin-%g.mnc" % (i+1)
            if self.avgPrefix:
                outputName = str(self.avgPrefix) + "-" + outputName
            nlinOutput = abspath(self.nlinDir) + "/" + outputName
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
    def __init__(self, inputArray, targetFH, nlinOutputDir, avgPrefix, nlin_protocol=None):
        NLINBase.__init__(self, inputArray, targetFH, nlinOutputDir, avgPrefix, nlin_protocol)
        
        """Setup parameters, either as defaults, or read from a .csv"""
        self.nlinParams = mp.setMincANTSParams(self.fileRes, reg_protocol=nlin_protocol)
    
        self.blurs = self.nlinParams.blurs
        self.gradient = self.nlinParams.gradient
        self.similarityMetric = self.nlinParams.similarityMetric
        self.weight = self.nlinParams.weight
        self.radiusHisto = self.nlinParams.radiusHisto
        self.transformationModel = self.nlinParams.transformationModel
        self.regularization = self.nlinParams.regularization
        self.iterations = self.nlinParams.iterations
        self.useMask = self.nlinParams.useMask
        self.generations = self.nlinParams.generations
    
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
        if i == (self.generations -1):
            inputFH.setLastBasevol(newBaseVol=rs.outputFiles[0])

class NLINminctracc(NLINBase):
    """
        This class does an iterative non-linear registration using the minctracc
        registration protocol. Default number of generations is 6. 
    """
    def __init__(self, inputArray, targetFH, nlinOutputDir, avgPrefix, nlin_protocol=None):
        NLINBase.__init__(self, inputArray, targetFH, nlinOutputDir, avgPrefix, nlin_protocol)
        
        """Setup parameters, either as defaults, or read from a .csv"""
        self.nlinParams = mp.setNlinMinctraccParams(self.fileRes, reg_protocol=nlin_protocol)
        self.blurs = self.nlinParams.blurs
        self.stepSize = self.nlinParams.stepSize
        self.iterations = self.nlinParams.iterations
        self.simplex = self.nlinParams.simplex
        self.useGradient = self.nlinParams.useGradient
        self.optimization = self.nlinParams.optimization
        self.generations = self.nlinParams.generations
        self.w_translations = self.nlinParams.w_translations
        self.stiffness = self.nlinParams.stiffness
        self.weight = self.nlinParams.weight
        self.similarity = self.nlinParams.similarity
    
    
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
                        weight=self.weight[i], 
                        stiffness=self.stiffness[i],
                        similarity=self.similarity[i],
                        w_translations = self.w_translations[i],
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
                            weight=self.weight[i], 
                            stiffness=self.stiffness[i],
                            similarity=self.similarity[i],
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
        if i == (self.generations -1):
            inputFH.setLastBasevol(newBaseVol=rs.outputFiles[0])
            
if __name__ == "__main__":
    
    application = NonlinearRegistration()
    application.start()
            
            
            
            
