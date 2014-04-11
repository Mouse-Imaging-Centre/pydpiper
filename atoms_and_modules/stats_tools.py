from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
from atoms_and_modules.registration_functions import isFileHandler
from atoms_and_modules.minc_atoms import xfmConcat, xfmInvert
import pydpiper.file_handling as fh
from optparse import OptionGroup
import sys
import Pyro

Pyro.config.PYRO_MOBILE_CODE=1

def addStatsOptions(parser):
    group = OptionGroup(parser, "Statistics options", 
                        "Options for calculating statistics.")
    group.add_option("--calc-stats", dest="calc_stats",
                      action="store_true",
                      help="Calculate statistics at the end of the registration. [Default]")
    group.add_option("--no-calc-stats", dest="calc_stats",
                      action="store_false", 
                      help="If specified, statistics are not calculated. Opposite of --calc-stats.")
    group.add_option("--stats-kernels", dest="stats_kernels",
                      type="string", default="1.0,0.5,0.2,0.1", 
                      help="comma separated list of blurring kernels for analysis. Default is: 1.0,0.5,0.2,0.1")
    parser.set_defaults(calc_stats=True)
    parser.add_option_group(group)
    
def createOutputFileName(iFH, xfm, outputDir, nameExt):
    outDir = iFH.setOutputDirectory(outputDir)
    outBase = fh.removeBaseAndExtension(xfm) + nameExt
    outputFile = fh.createBaseName(outDir, outBase)
    return outputFile

class StatsGroup(object):
    """This group saves the key output from each instance for CalcStats, 
       so it can easily be retrieved later."""
    def __init__(self):
        self.relativeJacobians = {}
        self.absoluteJacobians = {}        
        
class CalcStats(object):
    """Statistics calculation between an input and target. 
       This class calculates multiple displacement fields, relative and absolute jacobians.  
       General functionality as follows:
       1. Class instantiated with input, target and statsKernels. Note that here, the statsKernels 
          specified are blurs used to smooth the displacement fields prior to additional calculations. 
          They may be a string of comma separated values or an array of doubles. 
       2. An additional transform may also be included to calculate absolute 
          jacobians to a different space, as is described in the __init__ function, 
          documentation and elsewhere in the code.  
       3. If needed, invert transform between input and target in setupXfms(). This is necessary
          as this class assumes that the target is the reference space, from which all stats
          are calculated. 
       4. Call fullStatsCalc in calling class after instantiation. This calculates linear and 
          pure nonlinear displacement before calculating jacobians.    
       5. Ability to recenter displacements using an average may be re-added in the future. 
    """
    def __init__(self, inputFH, targetFH, statsKernels, additionalXfm=None):
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.blurs = []
        self.setupBlurs(statsKernels)
        self.statsGroup = StatsGroup()
        self.setupXfms()
        """ additionalXfm is an optional transform that may be specified. If it is, 
            it is concatenated with the lastXfm from input to target. This additional
            transform must also be in the same direction as the lastXfm (e.g. input to target)
            Example usage: if the lastXfm from input to target goes from lsq12 to nlin space
            and you would like to calculate the absolute jacobians to lsq6 space, the additional
            transform specified may be the lsq6 to lsq12 transform from input to target. 
        """
        self.additionalXfm = additionalXfm
        
    def setupBlurs(self, statsKernels):
        if isinstance(statsKernels, list):
            self.blurs = statsKernels
        elif isinstance(statsKernels, str):
            for i in statsKernels.split(","):
                self.blurs.append(float(i))
        else:
            print "Improper type of blurring kernels specified for stats calculation: " + str(statsKernels)
            sys.exit()
           
    def setupXfms(self):
        self.xfm = self.inputFH.getLastXfm(self.targetFH)
        if not self.xfm:
            print "Cannot calculate statistics. No transform between input and target specified."
            sys.exit()
        else:
            self.invXfm = self.targetFH.getLastXfm(self.inputFH)
            if not self.invXfm:
                xi = xfmInvert(self.xfm, FH=self.inputFH)
                self.p.addStage(xi)
                self.invXfm = xi.outputFiles[0]
    
    def fullStatsCalc(self):
        self.linAndNlinDisplacement()
        self.calcDetAndLogDet(useFullDisp=False)  # Calculate relative jacobians
        self.calcDetAndLogDet(useFullDisp=True)   # Calculate absolute jacobians
    
    def calcFullDisplacement(self):
        """Calculate full displacement from target to input. If an
           additionaXfm is specified, it is concatenated to self.xfm here """
        if self.additionalXfm:
            outXfm = createOutputFileName(self.inputFH, self.xfm, "transforms", "_with_additional.xfm")
            xc = xfmConcat([self.additionalXfm, self.xfm], outXfm, fh.logFromFile(self.inputFH.logDir, outXfm))
            self.p.addStage(xc)
            xi = xfmInvert(xc.outputFiles[0], FH=self.inputFH)
            self.p.addStage(xi)
            fullDisp = mincDisplacement(self.targetFH, self.inputFH, transform=xi.outputFiles[0])
        else:
            fullDisp = mincDisplacement(self.targetFH, self.inputFH, transform=self.invXfm)
        self.p.addStage(fullDisp)
        self.fullDisp = fullDisp.outputFiles[0]
    
    def calcNlinDisplacement(self):
        """Calculate pure non-linear displacement from target to input
           1. Concatenate self.invXfm (target to input xfm) and self.linearPartOfNlinXfm   
           2. Compute mincDisplacement on this transform. 
        """
        pureNlinXfm = createOutputFileName(self.inputFH, self.invXfm, "transforms", "_pure_nlin.xfm")
        xc = xfmConcat([self.invXfm, self.linearPartOfNlinXfm], 
                       pureNlinXfm, fh.logFromFile(self.inputFH.logDir, pureNlinXfm))
        self.p.addStage(xc)
        nlinDisp = mincDisplacement(self.targetFH, self.inputFH, transform=pureNlinXfm)
        self.p.addStage(nlinDisp)
        self.nlinDisp = nlinDisp.outputFiles[0]
        
    def linAndNlinDisplacement(self):
        """
           Calculation of full and pure non-linear displacements.
           The former is used to calculate absolute jacobians,
           the latter to calculate relative. The direction of the
           transforms and displacements is defined in each subclass.  
        """
        
        #1. Calculate linear part of non-linear xfm from input to target. 
        # This is necessary prior to calculating the pure nonlinear displacement
        lpnl = linearPartofNlin(self.inputFH, self.targetFH)
        self.p.addStage(lpnl)
        self.linearPartOfNlinXfm = lpnl.outputFiles[0]
        
        # 2. Calculate the pure non-linear displacement
        self.calcNlinDisplacement()
        
        # 3. Calculate the full displacement
        self.calcFullDisplacement()

             
    def calcDetAndLogDet(self, useFullDisp=False):  
        if useFullDisp:
            dispToUse = self.fullDisp #absolute jacobians
        else:
            dispToUse = self.nlinDisp #relative jacobians
        """Insert -1 at beginning of blurs array to include the calculation of unblurred jacobians."""
        self.blurs.insert(0,-1)    
        for b in self.blurs:
            """Create base name for determinant calculation."""
            outputBase = fh.removeBaseAndExtension(dispToUse).split("_displacement")[0]
            """Calculate smoothed deformation field for all blurs other than -1"""
            if b != -1:
                fwhm = "--fwhm=" + str(b)
                outSmooth = fh.createBaseName(self.inputFH.tmpDir, 
                                       outputBase + "_smooth_displacement_fwhm" + str(b) + ".mnc")
                cmd = ["smooth_vector", "--clobber", "--filter", fwhm, 
                       InputFile(dispToUse), OutputFile(outSmooth)]
                smoothVec = CmdStage(cmd)
                smoothVec.setLogFile(LogFile(fh.logFromFile(self.inputFH.logDir, outSmooth)))
                self.p.addStage(smoothVec)
                """Set input for determinant calculation."""
                inputDet = outSmooth
                nameAddendum = "_fwhm" + str(b)
            else:
                inputDet = dispToUse
                nameAddendum = ""
            outputDet = fh.createBaseName(self.inputFH.tmpDir, 
                                          outputBase + "_determinant" + nameAddendum + ".mnc")
            outDetShift = fh.createBaseName(self.inputFH.tmpDir, 
                                          outputBase + "_det_plus1" + nameAddendum + ".mnc")
            
            if useFullDisp: 
                #absolute jacobians
                outLogDet = fh.createBaseName(self.inputFH.statsDir, 
                                          outputBase + "_absolute_log_determinant" + nameAddendum + ".mnc")
            else:
                #relative jacobians
                outLogDet = fh.createBaseName(self.inputFH.statsDir, 
                                          outputBase + "_relative_log_determinant" + nameAddendum + ".mnc")
            
            """Calculate the determinant, then add 1 (per mincblob weirdness)"""
            
            cmd = ["mincblob", "-clobber", "-determinant", InputFile(inputDet), OutputFile(outputDet)]
            det = CmdStage(cmd)
            det.setLogFile(LogFile(fh.logFromFile(self.inputFH.logDir, outputDet)))
            self.p.addStage(det)
            
            cmd = ["mincmath", "-clobber", "-2", "-const", str(1), "-add", 
                   InputFile(outputDet), OutputFile(outDetShift)]
            det = CmdStage(cmd)
            det.setLogFile(LogFile(fh.logFromFile(self.inputFH.logDir, outDetShift)))
            self.p.addStage(det)
            
            """Calculate log determinant (jacobian) and add to statsGroup."""
            cmd = ["mincmath", "-clobber", "-2", "-log", InputFile(outDetShift), OutputFile(outLogDet)]
            det = CmdStage(cmd)
            det.setLogFile(LogFile(fh.logFromFile(self.inputFH.logDir, outLogDet)))
            self.p.addStage(det)
            if useFullDisp:
                self.statsGroup.absoluteJacobians[b] = outLogDet
            else:
                self.statsGroup.relativeJacobians[b] = outLogDet

class CalcChainStats(CalcStats):
    """This class calculates multiple displacement fields, absolute and relative
       jacobians. IT DOES NOT allow for adding an additional transform, as in the
       base class (CalcStats). This child class is designed specifically for the 
       registration chain application (or similar) and has less complexity than CalcStats()"""
    def __init__(self, inputFH, targetFH, statsKernels):
        CalcStats.__init__(self, inputFH, targetFH, statsKernels)
    
    def setupXfms(self):
        self.xfm = self.inputFH.getLastXfm(self.targetFH)
        if not self.xfm:
            print "Cannot calculate statistics. No transform between input and target specified."
            sys.exit()
    
    def calcFullDisplacement(self):
        """Calculates the full displacement from input to target without removing 
           the linear part. Note that inputFH is deliberately specified twice in the
           mincDisplacement call: Once as the input space, and once for the location 
           of the log files. """
        fullDisp = mincDisplacement(self.inputFH, self.inputFH, transform=self.xfm)
        self.p.addStage(fullDisp)
        self.fullDisp = fullDisp.outputFiles[0]
        
    def calcNlinDisplacement(self):
        """Calculate pure non-linear displacement from input to target 
           1. Invert the linear transform, so we get the linear xfm from target to input.
           2. Concatenate the full non-linear (input to target) transform with the
              linear target to input transform.
           3. Calculate the displacement on this transform. """
        xi = xfmInvert(self.linearPartOfNlinXfm, FH=self.inputFH)
        self.p.addStage(xi)
        
        pureNlinXfm = createOutputFileName(self.inputFH, self.xfm, "transforms", "_pure_nlin.xfm")
        xc = xfmConcat([self.xfm, xi.outputFiles[0]], 
                       pureNlinXfm, fh.logFromFile(self.inputFH.logDir, pureNlinXfm))
        self.p.addStage(xc)
        nlinDisp = mincDisplacement(self.inputFH, self.inputFH, transform=pureNlinXfm)
        self.p.addStage(nlinDisp)
        self.nlinDisp = nlinDisp.outputFiles[0]
    
class linearPartofNlin(CmdStage):
    def __init__(self, inputFH, targetFH, defaultDir="transforms"):
        CmdStage.__init__(self, None)
        
        try:  
            if isFileHandler(inputFH, targetFH):
                self.inFile = inputFH.getLastBasevol()  
                self.mask = inputFH.getMask() 
                self.xfm = inputFH.getLastXfm(targetFH)     
                self.outfile = self.setOutputFile(inputFH, defaultDir)
                self.logFile = fh.logFromFile(inputFH.logDir, self.outfile)
            else:
                print ("linear part of nlin currently only works using file handlers. "
                       "Exception being raised.")
                raise
    
        except:
            print "Failed in putting together linearPartofNlin command"
            print "Unexpected error: ", sys.exc_info()
            
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        
    def addDefaults(self):
        self.inputFiles += [self.inFile, self.xfm]   
        self.outputFiles += [self.outfile]       
        self.cmd += ["lin_from_nlin",
                     "-clobber", "-lsq12"] 
        if self.mask: 
            self.inputFiles += [self.mask]
            self.cmd += ["-mask", self.mask]
                 
    def finalizeCommand(self):
        self.cmd += [self.inFile, self.xfm, self.outfile]   
    def setName(self):
        self.name = "lin_from_nlin " 
    def setOutputFile(self, inFile, defaultDir):
        outDir = inFile.setOutputDirectory(defaultDir)
        outBase = (fh.removeBaseAndExtension(self.xfm) + "_linear_part.xfm")
        outputFile = fh.createBaseName(outDir, outBase)
        return(outputFile)  

class mincDisplacement(CmdStage):
    """This class calculates the displacement from an input
       volume, using a specified transform from this input to
       another volume. Must specify input volume, transform from
       that volume to a target, and an outputFH, which is where 
       the output and log files should be stored. The outputFH 
       and inputFH may be the same volume. A default directory
       for the output may optionally be specified, but is tmp if
       unspecified.  
    """
    def __init__(self, inputFH, outputFH, transform, defaultDir="tmp"):
        CmdStage.__init__(self, None)
        try:  
            if isFileHandler(inputFH, outputFH):
                self.inFile = inputFH.getLastBasevol()  
                self.xfm = transform
                self.outfile = createOutputFileName(outputFH, self.xfm, defaultDir, "_displacement.mnc")
                self.logFile = fh.logFromFile(outputFH.logDir, self.outfile)
            else:
                print ("minc_displacement only works using file handlers. "
                       "Exception being raised.")
                raise
    
        except:
            print "Failed in putting together minc_displacement command"
            print "Unexpected error: ", sys.exc_info()
            
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        
    def addDefaults(self):
        self.inputFiles += [self.inFile, self.xfm]   
        self.outputFiles += [self.outfile]       
        self.cmd += ["minc_displacement", "-clobber"] 
                 
    def finalizeCommand(self):
        self.cmd += [self.inFile, self.xfm, self.outfile]    
    def setName(self):
        self.name = "minc_displacement " 
