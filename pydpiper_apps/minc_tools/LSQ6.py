#!/usr/bin/env python

from pydpiper.application import AbstractApplication
from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.hierarchical_minctracc as hm
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.minc_modules as mm
from os.path import splitext, abspath
import logging
import Pyro
from optparse import OptionGroup
import sys
import re
import os
from datetime import date

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1

class LSQ6Registration(AbstractApplication):
    """ 
        This class handles a 6 parameter (rigid) registration between one or more files
        and a single target. 
        
        Source:
            One or more files with or without a mask
        
        Target:
            Single input file:
            The target can be a single input file with or without a mask
            
            Initial model:
            An initial model can be specified.  Here, you are required 
            to have a mask and an optional transformation.  This option 
            can be used if the "native space" (scanner space) is different
            from the "standard space", the space where you want to register
            your files.  Alternatively, the standard space can be in the
            same space as your native space, but have a tighter crop to
            save disk space and processing time.
               
        Complexity of registration:
            1) single coil, same position for all files.  If for example all 
            samples are scanned in the same coil and the orientation is fixed,
            e.g., the subjects always enter the scanner in the same position.
            This implies that the input files are more or less aligned to begin
            with.
            
            2) multiple coils, same position/orientation for all files.  Think 
            here for example about multiple coil live mice scans.  Even though
            the files from different coils reside in a different part in space,
            the orientation is the same.  This means that aligning the centre 
            of gravity for the input files gets them in option 1)
            
            3) multiple (or single) coil with random scan orientation.  In this 
            case, the input files are in the most random orientation/location.  
            The procedure here is to do a brute force search in the x,y,z rotation
            space in order to find the best alignment. 
    """
    def setup_options(self):
        group = OptionGroup(self.parser, "LSQ6-registration options", 
                        "Options for performing a 6 parameter (rigid) registration.")
        group.add_option("--target", dest="target",
                         type="string", default=None,
                         help="File to be used as the target for the 6 parameter alignment.")
        group.add_option("--init-model", dest="init_model",
                         type="string", default=None,
                         help="File in standard space in the initial model. The initial model can also have a file in native space and potentially a transformation file. See our wiki for detailed information on initial models.")
        self.parser.set_defaults(lsq6_method="lsq6_large_rotations")
        group.add_option("--lsq6-simple", dest="lsq6_method",
                         action="store_const", const="lsq6_simple",
                         help="Run a 6 parameter alignment assuming that the input files are roughly aligned: same space, similar orientation. [default: --lsq6-large-rotations]")
        group.add_option("--lsq6-centre-estimation", dest="lsq6_method",
                         action="store_const", const="lsq6_centre_estimation",
                         help="Run a 6 parameter alignment assuming that the input files have a similar orientation, but are scanned in different coils/spaces. [default: --lsq6-large-rotations]")
        group.add_option("--lsq6-large-rotations", dest="lsq6_method",
                         action="store_const", const="lsq6_large_rotations",
                         help="Run a 6 parameter alignment assuming that the input files have a random orientation and are scanned in different coils/spaces. A brute force search over the x,y,z rotation space is performed to find the best 6 parameter alignment. [default: --lsq6-large-rotations]")
        group.add_option("--lsq6-large-rotations-parameters", dest="large_rotation_parameters",
                         type="string", default="10,4,10,8,50,10",
                         help="Settings for the large rotation alignment. factor=factor based on smallest file resolution: 1) blur factor, 2) resample step size factor, 3) registration step size factor, 4) w_translations factor, 5) rotational range in degrees, 6) rotational interval in degrees. [default: %default]")
        self.parser.add_option_group(group)
        """Add option groups from specific modules"""
        rf.addGenRegOptionGroup(self.parser)
        self.parser.set_usage("%prog [options] [--target target.mnc or --init-model /init/model/file.mnc] input file(s)") 

    def setup_appName(self):
        appName = "LSQ6-registration"
        return appName
    
    def run(self):
        options = self.options
        args = self.args
        
        # initial error handling:  verify that at least one input file is specified and that it is a MINC file
        if(len(args) < 1):
            print "Error: no source image provided\n"
            sys.exit()
        for i in range(len(args)):
            ext = splitext(args[i])[1]
            if(re.match(".mnc", ext) == None):
                print "Error: input file is not a MINC file:, ", args[i], "\n"
                sys.exit()

        # verify that we have some sort of target specified
        if(options.init_model == None and options.target == None):
            print "Error: please specify either a target file for the registration (--target), or an initial model (--init-model)\n"
            sys.exit()   
        if(options.init_model != None and options.target != None):
            print "Error: please specify only one of the options: --target  --init-model\n"
            sys.exit()
        
        mainDirectory = None
        if(options.output_directory == None):
            mainDirectory = os.getcwd()
        else:
            mainDirectory = fh.makedirsIgnoreExisting(options.output_directory)
        
        """Make main pipeline directories"""
        # TODO: change this as soon as mbm-redesign has been merged into master!!
        #pipeDir = fh.makedirsIgnoreExisting(options.pipeline_dir)
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
        #nlinDirectory = createSubDir(pipeDir, pipeName + "_nlin")
        lsq6Directory = fh.createSubDir(mainDirectory, pipeName + "_lsq6")
        processedDirectory = fh.createSubDir(mainDirectory, pipeName + "_processed")
        
        # create file handles for the input file(s) 
        inputFiles = rf.initializeInputFiles(args, mainDirectory=processedDirectory)

        initModel = None
        if(options.target != None):
            targetPipeFH = rfh.RegistrationPipeFH(abspath(options.target), basedir=lsq6Directory)
        else: # options.init_model != None  
            initModel = rf.setupInitModel(options.init_model, mainDirectory)
            if (initModel[1] != None):
                # we have a target in "native" space 
                targetPipeFH = initModel[1]
            else:
                # we will use the target in "standard" space
                targetPipeFH = initModel[0]
        
        """
            Option 1) run a simple lsq6: the input files are assumed to be in the
            same space and roughly in the same orientation.
        """
        if(options.lsq6_method == "lsq6_simple"):
            lsq6minctracc =  LSQ6Minctracc(inputFiles,
                                           targetPipeFH,
                                           initial_model    = initModel,
                                           lsq6OutputDir    = lsq6Directory,
                                           inital_transform = "identity")
            lsq6minctracc.createLSQ6Transformation()
            lsq6minctracc.finalize()
            self.pipeline.addPipeline(lsq6minctracc.p)
        
        """
            Option 2) run an lsq6 registration where the centre of the input files
            is estimated.  Orientation is assumed to be similar, space is not.
        """
        if(options.lsq6_method == "lsq6_centre_estimation"):
            lsq6minctracc =  LSQ6Minctracc(inputFiles,
                                           targetPipeFH,
                                           initial_model     = initModel,
                                           lsq6OutputDir     = lsq6Directory,
                                           initial_transform = "estimate")
            lsq6minctracc.createLSQ6Transformation()
            lsq6minctracc.finalize()
            self.pipeline.addPipeline(lsq6minctracc.p)
        """
            Option 3) run a brute force rotational minctracc.  Input files can be
            in any random orientation and space.
        """
        if(options.lsq6_method == "lsq6_large_rotations"):
            lsq6rot = LSQ6RotationalMinctracc(inputFiles,
                                             targetPipeFH,
                                             initial_model = initModel,
                                             lsq6OutputDir = lsq6Directory,
                                             large_rotation_parameters = options.large_rotation_parameters)
            lsq6rot.createLSQ6Transformation()
            lsq6rot.finalize()
            self.pipeline.addPipeline(lsq6rot.p)


class LSQ6Base(object):
    """
        This is the parent class for any lsq6 registration method.  It implements the following:
        
        - check input files (at the moment assumed to be file handlers
        - check whether an lsq6 directory is provided when more than 1 input file is given
        - creates a new group for the input files: "lsq6"
        
        (the child classes will fill in the function createLSQ6Transform)
        
        - handle a potential ...native_to_standard.xfm transform from an initial model
        - resample the input files
        - create an average if at least 2 input files are given
    """
    def __init__(self,
                 inputFiles,
                 targetFile,
                 initial_model = None,
                 lsq6OutputDir = None):
        self.p              = Pipeline()
        self.inputs         = inputFiles
        self.target         = targetFile
        self.initial_model  = initial_model
        self.lsq6OutputDir  = lsq6OutputDir 
        self.lsq6Avg        = None # will contain file handler to lsq6 average
        self.filesToAvg     = [] # store the resampled lsq6 files in order to create an average at the end
        
        self.check_inputs()
        self.check_lsq6_folder()
        self.setLSQ6GroupToInputs()
        
        
    def check_inputs(self):
        """
            Currently only file handlers are covered in these classes.  This function
            check whether both input and target files are given as file handlers 
        """
        # input file checking...    
        if(not(type(self.inputs) is list)):
            if(not rf.isFileHandler(self.inputs)):
                print "My apologies... the LSQ6 modules currently only work with file handlers (input file)...\nGoodbye"
                sys.exit()
            else:
                # for ease of use, turn into a list
                self.inputs = [self.inputs]
        else:
            for inputfile in self.inputs:
                if(not rf.isFileHandler(inputfile)):
                    print "My apologies... the LSQ6 modules currently only work with file handlers (input files)...\nGoodbye"
                    sys.exit()
        if(not rf.isFileHandler(self.target)):
            print "My apologies... the LSQ6 modules currently only work with file handlers (target file)...\nGoodbye"
            sys.exit()
    
    def check_lsq6_folder(self):
        """
            Make sure that the output directory for the lsq6 average is 
            defined is at least 2 input files are provided
        """
        if(len(self.inputs) > 1 and not(self.lsq6OutputDir)):
            print "Error: ", len(self.inputs), " input files were provided to the LSQ6 module but no output directory for the average was given. Don't know where to put it...\nGoodbye."
            sys.exit()
        
        # just in case this directory was not created yet
        if(self.lsq6OutputDir):
            fh.makedirsIgnoreExisting(self.lsq6OutputDir)

    def setLSQ6GroupToInputs(self):
        """
            make sure that by default any input file handler is 
            given the group "lsq6"
        """
        # create a new group to indicate in the output file names that this is the lsq6 stage
        for i in range(len(self.inputs)):
            self.inputs[i].newGroup(groupName="lsq6")

    def createLSQ6Transformation(self):
        """
            This function is to be filled in the child classes.  For instance
            a rotational minctracc could be called here, or a hierarchical
            6 parameter minctracc call
        """
        pass

    def addNativeToStandardFromInitModel(self):
        """
            If an initial model is used, we might have to concatenate
            the 6 parameter transformation that was created with the 
            transformation from the inital model.  This function does
            that if necessary
        """
        if(self.initial_model != None):
            if(self.initial_model[2] != None):
                for inputFH in self.inputs:
                    prevxfm = inputFH.getLastXfm(self.target)
                    newxfm = inputFH.registerVolume(self.initial_model[0], "transforms")
                    logFile = fh.logFromFile(inputFH.logDir, newxfm)
                    self.p.addStage(ma.xfmConcat([prevxfm, self.initial_model[2]],
                                             newxfm,
                                             logFile))
    
    def resampleInputFiles(self):
        """
            resample input files using the last transformation
        """
        for inputfile in self.inputs:
            likeFileForResample =  self.target
            targetFHforResample = self.target
            if(self.initial_model != None):
                if(self.initial_model[1] != None):
                    likeFileForResample = self.initial_model[0]
                    targetFHforResample = self.initial_model[0]
            rs = ma.mincresample(inputfile,targetFHforResample,likeFile=likeFileForResample)
            self.filesToAvg.append(rs.outputFiles[0])
            self.p.addStage(rs)

    def createAverage(self):
        """
            Create the lsq6 average after all has been done...
        """
        if(len(self.filesToAvg) > 1):
            lsq6AvgOutput = abspath(self.lsq6OutputDir) + "/" + "lsq6_average.mnc"
            # TODO: fix mask issue
            lsq6FH = rfh.RegistrationPipeFH(lsq6AvgOutput, mask=None, basedir=self.lsq6OutputDir)
            logBase = fh.removeBaseAndExtension(lsq6AvgOutput)
            avgLog = fh.createLogFile(lsq6FH.logDir, logBase)
            avg = ma.mincAverage(self.filesToAvg, lsq6AvgOutput, logFile=avgLog)
            self.p.addStage(avg)
            self.lsq6Avg = lsq6FH

    def finalize(self):
        """
            Within one call, take care of the potential initial model transformation,
            the resampling of input files and create an average 
        """
        self.addNativeToStandardFromInitModel()
        self.resampleInputFiles()
        self.createAverage()

class LSQ6RotationalMinctracc(LSQ6Base):
    """
        This class performs an lsq6 registration using rotational minctracc
        from start to end.  That means that it takes care of blurring the
        input files and target, running RotationalMinctracc, resample the 
        input files, and create an average if at least 2 input files are provided.
        
        * Assumptions/input:
            - inputFiles are provided in the form of file handlers
            - targetFile is provided as a file handler
            - large_rotation_parameters as the same as RotationalMinctracc() in minc_atoms,
            please see that module for more information
            - if an initial model is provided, the input to the parameter
            initial_model is assumed to be the output of the function 
            setupInitModel()
            - lsq6OutputDir is a string indicating where the the potential average should go
        
        Output:
            - (self.lsq6Avg) if at least 2 input files were provided, this class will 
            store the file handler for the average in the variable lsq6Avg 
            
    """
    def __init__(self, 
                 inputFiles,
                 targetFile,
                 initial_model = None,
                 lsq6OutputDir = None,
                 large_rotation_parameters="10,4,10,8,50,10"):
        # initialize all the defaults in 
        LSQ6Base.__init__(self, inputFiles, targetFile, initial_model, lsq6OutputDir)
        
        self.parameters     = large_rotation_parameters
        
    def createLSQ6Transformation(self):    
        # We should take care of the appropriate amount of blurring
        # for the input files here 
        parameterList = self.parameters.split(',')
        blurFactor= float(parameterList[0])
        
        blurAtResolution = -1
        # assumption: all files have the same resolution, so we take input file 1
        highestResolution = rf.getFinestResolution(self.inputs[0])
        if(blurFactor != -1):
            blurAtResolution = blurFactor * highestResolution
        
        self.p.addStage(ma.blur(self.target, fwhm=blurAtResolution))
        
        for inputFH in self.inputs:
            
            self.p.addStage(ma.blur(inputFH, fwhm=blurAtResolution))
        
            self.p.addStage((ma.RotationalMinctracc(inputFH,
                                                    self.target,
                                                    blur               = blurAtResolution,
                                                    resample_step      = float(parameterList[1]),
                                                    registration_step  = float(parameterList[2]),
                                                    w_translations     = float(parameterList[3]),
                                                    rotational_range   = int(parameterList[4]),
                                                    rotational_interval= int(parameterList[5]) )))


class LSQ6Minctracc(LSQ6Base):
    """
        TODO: document
        
        * Assumptions/input
            - inputFiles are provided in the form of file handlers
            - targetFile is provided as a file handler
        
    """
    def __init__(self,
                 inputFiles,
                 targetFile,
                 initial_model    = None,
                 lsq6OutputDir    = None,
                 inital_transform = "estimate",
                 lsq6_protocol    = None):
        # initialize all the defaults in 
        LSQ6Base.__init__(self, inputFiles, targetFile, initial_model, lsq6OutputDir)
        self.initial_transform = inital_transform 
        self.lsq6_protocol     = lsq6_protocol
        
        self.setInitialMinctraccTransform()
        
        self.setHierarchyOptions()
        
        #
        ##
        ### Quick hack to see if things work...
        ##
        #
        
        self.blur        = [1.2,0.6,0.3]
        self.simplex     = [  6,  3,  2]
        self.step        = [1.2,0.6,0.3]
        self.generations = 3
        
    def setInitialMinctraccTransform(self):
        # set option which will by used by the minctracc CmdStage:
        self.linearparam = None
        if(self.initial_transform == "estimate"):
            self.linearparam = "lsq6"
        elif(self.initial_transform == "identity"):
            self.linearparam = "lsq6-identity"
        else:
            print "Error: unknown option used for inital_transform in LSQ6Minctracc: ", self.initial_transform, "\nGoodbye."
            sys.exit()
    
    def setHierarchyOptions(self):
        """
            It is possible to specify an lsq6 protocol.  If none is specified,
            defaults will be used as follows:
            
            ************************* DEFAULTS ************************************
            * all parameters will be set as a factor of the input file's resolution
            
            When using the estimation of centres, 5 stages:
            blur    = [ 90, 35, 17,  9,  4]     # in mm at 56micron files: [5.04,  1.96, 0.952, 0.504, 0.224]
            gradient= [  0,  0,  0,  1,  0]
            simplex = [128, 64, 40, 28, 16]     # in mm at 56mircon files: [7.168, 3.584, 2.24, 1.568, 0.896]
            step    = [ 90, 35, 17,  9,  4] 
        """
    
    def createLSQ6Transformation(self):
        """
            TODO: more info...
            Perform the hierarchical iterations...
        """
        # TODO: check all parameters for consistency in #stages/iterations
        # create all blurs first
        for i in range(self.generations):
            self.p.addStage(ma.blur(self.target, self.blur[i], gradient=True))
            for inputfile in self.inputs:
                # create blur for input
                self.p.addStage(ma.blur(inputfile, self.blur[i], gradient=True))
        
        # now perform the registrations
        for inputfile in self.inputs:
            for i in range(self.generations):
                mt = ma.minctracc(inputfile,
                                  self.target,
                                  blur        = self.blur[i],
                                  simplex     = self.simplex[i],
                                  step        = self.step[i],
                                  linearparam = "lsq6") 
                print mt.cmd
                self.p.addStage(mt)


        

if __name__ == "__main__":
    application = LSQ6Registration()
    application.start()