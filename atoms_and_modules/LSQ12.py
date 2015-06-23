#!/usr/bin/env python

from __future__ import print_function
from pydpiper.application import AbstractApplication
from pydpiper.pipeline import Pipeline, InputFile, OutputFile, LogFile, CmdStage
from pydpiper.file_handling import createBaseName, logFromFile
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.minc_parameters as mp
from atoms_and_modules.registration_file_handling import RegistrationPipeFH, RegistrationFHBase
from os.path import basename, abspath
import sys
import logging
import random

logger = logging.getLogger(__name__)

def addLSQ12ArgumentGroup(parser):
    """option group for the command line argument parser"""
    group = parser.add_argument_group("LSQ12 registration options",
                        "Options for performing a pairwise, affine registration")
    group.add_argument("--lsq12-max-pairs", dest="lsq12_max_pairs",
                       type=int, default=None,
                       help="Maximum number of pairs to register together.  [Default = %(default)s]")
    group.add_argument("--lsq12-likefile", dest="lsq12_likeFile",
                       type=str, default=None,
                       help="Can optionally specify a like file for resampling at the end of pairwise "
                       "alignment. Default is None, which means that the input file will be used. [Default = %(default)s]")
    group.add_argument("--lsq12-subject-matter", dest="lsq12_subject_matter",
                       type=str, default=None,
                       help="Can specify the subject matter for the pipeline. This will set the parameters "
                       "for the 12 parameter alignment based on the subject matter rather than the file "
                       "resolution. Currently supported option is: \"mousebrain\". [Default = %(default)s].")
    group.add_argument("--lsq12-protocol", dest="lsq12_protocol",
                       type=str, default=None,
                       help="Can optionally specify a registration protocol that is different from defaults. "
                       "Parameters must be specified as in the following example: \n"
                       "applications_testing/test_data/minctracc_example_linear_protocol.csv \n"
                       "[Default = %(default)s].")
    parser.add_argument_group(group)

class LSQ12Registration(AbstractApplication):
    """
        This class performs an iterative, pairwise affine registration on all subjects specified,
        or if a maximum number of pairs is specified via the lsq12_max_pairs command line option,
        then each subject will only be registered to a subset of the other subjects, which will 
        be chosen via a random number generator.    
        
        After aligning all pairs of subjects, an average transform is created for each subject
        and 
          
    """
    def setup_options(self):
        """Add option groups from specific modules"""
        rf.addGenRegArgumentGroup(self.parser)
        addLSQ12ArgumentGroup(self.parser)
         
    def setup_appName(self):
        appName = "LSQ12-registration"
        return appName

    def run(self):
        options = self.options
        args = self.args

        rf.checkInputFiles(args)
        
        # Setup output directories for LSQ12 registration.        
        dirs = rf.setupDirectories(self.outputDir, options.pipeline_name, module="LSQ12")
        
        #Initialize input files from args
        inputFiles = rf.initializeInputFiles(args, dirs.processedDir, maskDir=options.mask_dir)

        # we potentially use the resolution of the input files to determine some
        # minctracc parameters and blurring kernels (if no protocol or subject matter is specified)
        resolutionForLSQ12 = None
        
        #Set up like file for resampling, if one is specified
        if options.lsq12_likeFile:
            likeFH = RegistrationFHBase(abspath(options.lsq12_likeFile), 
                                        basedir=dirs.lsq12Dir)
            resolutionForLSQ12 = rf.returnFinestResolution(likeFH)
        else:
            likeFH = None
            # if there is not a like file, we can use the resolution of
            # the first input file. The assumption being that all input
            # files have the same resolution
            resolutionForLSQ12 = rf.returnFinestResolution(inputFiles[0])

        
        #Iterative LSQ12 model building
        lsq12 = FullLSQ12(inputFiles,
                          outputDir=dirs.lsq12Dir,
                          queue_type=options.queue_type,
                          likeFile=likeFH,
                          maxPairs=options.lsq12_max_pairs, 
                          lsq12_protocol=options.lsq12_protocol, 
                          subject_matter=options.lsq12_subject_matter,
                          resolution=resolutionForLSQ12)
        lsq12.iterate()
        self.pipeline.addPipeline(lsq12.p)
        

class FullLSQ12(object):
    """
        This class takes an array of input file handlers along with an optionally specified 
        protocol and does 12-parameter alignment and averaging of all of the pairs. 
        
        Required arguments:
        inputArray = array of file handlers to be registered
        outputDir = an output directory to place the final average from this registration
       
        Optional arguments include: 
        --likeFile = a file handler that can be used as a likeFile for resampling
            each input into the final lsq12 space. If none is specified, the input
            will be used
        --maxPairs = maximum number of pairs to register. If this pair is specified, 
            then each subject will only be registered to a subset of the other subjects.
        --lsq2_protocol = an optional csv file to specify a protocol that overrides the defaults.
        --subject_matter = currently supports "mousebrain". If this is specified, the parameter for
        the minctracc registrations are set based on defaults for mouse brains instead of the file
        resolution. 
    """
    
    def __init__(self, inputArray, 
                 outputDir,
                 queue_type,
                 likeFile=None, 
                 maxPairs=None,
                 lsq12_protocol=None,
                 subject_matter=None,
                 resolution=None):
        self.p = Pipeline()
        self.queue_type = queue_type
        """Initial inputs should be an array of fileHandlers with lastBasevol in lsq12 space"""
        self.inputs = inputArray
        """Output directory should be _nlin """
        self.lsq12Dir = outputDir
        """likeFile for resampling"""
        self.likeFile=likeFile
        """Maximum number of pairs to calculate"""
        self.maxPairs = maxPairs
        """Final lsq12 average"""
        self.lsq12Avg = None
        """Final lsq12 average file handler (e.g. the file handler associated with lsq12Avg)"""
        self.lsq12AvgFH = None
        """ Dictionary of lsq12 average transforms, which will include one per input.
            Key is input file handler and value is string pointing to final average lsq12
            transform for that particular subject. 
            These xfms may be used subsequently for statistics calculations. """
        self.lsq12AvgXfms = {}
        self.fileRes = None

        """Create the blurring resolution from the file resolution"""
        if (subject_matter==None and resolution==None):
            print("\nError: the FullLSQ12 module was called without specifying the resolution that it should be run at, and without specifying a subject matter. Please indicate one of the two. Exiting...\n")
            sys.exit()
        elif (subject_matter and resolution):
            # subject matter has precedence over resolution
            self.fileRes = None
        elif resolution:
            self.fileRes = resolution
            
        
        """"Set up parameter array"""
        self.lsq12Params = mp.setLSQ12MinctraccParams(self.fileRes, 
                                                      subject_matter=subject_matter, 
                                                      reg_protocol=lsq12_protocol)
        self.blurs = self.lsq12Params.blurs
        self.stepSize = self.lsq12Params.stepSize
        self.useGradient = self.lsq12Params.useGradient
        self.simplex = self.lsq12Params.simplex
        self.w_translations = self.lsq12Params.w_translations
        self.generations = self.lsq12Params.generations
        
        # Create new lsq12 group for each input prior to registration
        for inputF in self.inputs:
            inputF.newGroup(groupName="lsq12")
         
    def iterate(self):
        xfmsToAvg = {}
        lsq12ResampledFiles = {}
        for inputFH in self.inputs:
            """Create an array of xfms, to compute an average lsq12 xfm for each input"""
            xfmsToAvg[inputFH] = []
            if self.maxPairs is not None:
                if self.maxPairs >= len(self.inputs) - 1:
                    # -1 prevents unnecessary sampling in the case self.maxPairs = len(self.inputs) - 1
                    inputs = self.inputs
                else:
                    random.seed(tuple(map(lambda fh: fh.inputFileName, self.inputs)))
                    # if inputFH is included in the sample, we will register against one fewer target
                    inputs = random.sample(filter(lambda fh: fh != inputFH, self.inputs), self.maxPairs)
            else:
                inputs = self.inputs
            for targetFH in inputs:
                if inputFH != targetFH:
                    lsq12 = LSQ12(inputFH,
                                  targetFH,
                                  blurs=self.blurs,
                                  step=self.stepSize,
                                  gradient=self.useGradient,
                                  simplex=self.simplex,
                                  w_translations=self.w_translations)
                    self.p.addPipeline(lsq12.p)
                    xfmsToAvg[inputFH].append(inputFH.getLastXfm(targetFH))

            """Create average xfm for inputFH using xfmsToAvg array"""
            avgXfmOutput = createBaseName(inputFH.transformsDir, inputFH.basename + "-avg-lsq12.xfm")
            cmd = ["xfmavg", "-verbose", "-clobber"] \
                  + map(InputFile, xfmsToAvg[inputFH]) + [OutputFile(avgXfmOutput)]
            #for i in range(len(xfmsToAvg[inputFH])):
            #    cmd.append(InputFile(xfmsToAvg[inputFH][i]))
            # '-clobber' works around #157, but is probably better in general       
            #cmd.append(OutputFile(avgXfmOutput))
            xfmavg = CmdStage(cmd)
            xfmavg.setLogFile(LogFile(logFromFile(inputFH.logDir, avgXfmOutput)))
            self.p.addStage(xfmavg)
            self.lsq12AvgXfms[inputFH] = avgXfmOutput
            """ resample brain and add to array for mincAveraging"""
            if not self.likeFile:
                likeFile=inputFH
            else:
                likeFile=self.likeFile
            rslOutput = createBaseName(inputFH.resampledDir, inputFH.basename + "-resampled-lsq12.mnc")
            res = ma.mincresample(inputFH, 
                                  inputFH,
                                  transform=avgXfmOutput, 
                                  likeFile=likeFile, 
                                  output=rslOutput,
                                  argArray=["-sinc"])   
            self.p.addStage(res)
            lsq12ResampledFiles[inputFH] = rslOutput
        """ After all registrations complete, setLastBasevol for each subject to be
            resampled file in lsq12 space. We can then call mincAverage on fileHandlers,
            as it will use the lastBasevol for each by default."""
        for inputFH in self.inputs:
            inputFH.setLastBasevol(lsq12ResampledFiles[inputFH])
        """ mincAverage all resampled brains and put in lsq12Directory""" 
        self.lsq12Avg = abspath(self.lsq12Dir) + "/" + basename(self.lsq12Dir) + "-pairs.mnc" 
        self.lsq12AvgFH = RegistrationPipeFH(self.lsq12Avg, basedir=self.lsq12Dir)
        avg = ma.average(inputs, queue_type=self.queue_type,
                         outputAvg=self.lsq12AvgFH,
                         output=self.lsq12Avg,
                         defaultDir=self.lsq12Dir)
        self.p.addStage(avg)
            
class LSQ12(object):
    """Basic LSQ12 class. 
    
    This class takes an input FileHandler and a targetFileHandler as required inputs. A series of
    minctracc calls will then produce the 12-parameter alignment. The number of minctracc calls 
    and their parameters are controlled by four further arguments to the constructor:
    
    blurs: an array of floats containing the FWHM of the blurring kernel to be used for each call
    gradient: an array of booleans stating whether we should use the blur (False) or gradient (True) of each blur
    step: an array of floats containing the step used by minctracc in each call
    simplex: an array of floats containing the simplex used by minctracc in each call.
    
    The number of entries in those three (blurs, step, simplex) input arguments determines the number
    of minctracc calls executed in this module. For example, the following call:
    LSQ12(inputFH, targetFH, blurs=[10,5,2], gradient=[False,True,True], step=[4,4,4], simplex=[20,20,20])
    will result in three successive minctracc calls, each initialized with the output transform of the 
    previous call.
    """
    def __init__(self,
                 inputFH,
                 targetFH, 
                 blurs=[0.3, 0.2, 0.15], 
                 step=[1,0.5,0.333333333333333],
                 gradient=[False,True,False],
                 simplex=[3,1.5,1],
                 w_translations=[0.4,0.4,0.4],
                 defaultDir="tmp"):                                      

        # TO DO: Might want to take this out and pass in # of generations, since
        # checking happens there. 
        if len(blurs) == len(step) == len(simplex):
            # do nothing - all lengths are the same and we're therefore happy
            pass
        else:
            logger.error("The same number of entries are required for blurs, step, and simplex in LSQ12")
            sys.exit()
                
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.blurs = blurs
        self.step = step
        self.blurs = blurs
        self.gradient = gradient
        self.simplex = simplex
        self.w_translations = w_translations
        self.defaultDir = defaultDir
            
        self.blurFiles()
        self.buildPipeline()
    
    def blurFiles(self):
        for b in self.blurs:
            if b != -1:
                tblur = ma.blur(self.targetFH, b, gradient=True)
                iblur = ma.blur(self.inputFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
        
    def buildPipeline(self):
        for i in range(len(self.blurs)):    
            linearStage = ma.minctracc(self.inputFH, 
                                       self.targetFH, 
                                       blur=self.blurs[i], 
                                       defaultDir=self.defaultDir,
                                       gradient=self.gradient[i],                                     
                                       linearparam="lsq12",
                                       step=self.step[i],
                                       w_translations=self.w_translations[i],
                                       simplex=self.simplex[i])
            self.p.addStage(linearStage)

if __name__ == "__main__":
    
    application = LSQ12Registration()
    application.start()
            
