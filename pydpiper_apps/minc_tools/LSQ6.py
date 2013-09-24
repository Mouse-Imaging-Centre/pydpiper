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
        
        # create a new group to indicate in the output file names that this is the lsq6 stage
        for i in range(len(inputFiles)):
            inputFiles[i].newGroup(groupName="lsq6")
        
        # store the resampled lsq6 files in order to create an average at the end
        filesToAvg = []
        
        """
            Option 1) run a simple lsq6: the input files are assumed to be in the
            same space and roughly in the same orientation.
        """
        
        """
            Option 2) run an lsq6 registration where the centre of the input files
            is estimated.  Orientation is assumed to be similar, space is not.
        """
        
        """
            Option 3) run a brute force rotational minctracc.  Input files can be
            in any random orientation and space.
        """
        if(options.lsq6_method == "lsq6_large_rotations"):
            # We should take care of the appropriate amount of blurring
            # for the input files here 
            parameterList = options.large_rotation_parameters.split(',')
            blurFactor= float(parameterList[0])
            
            blurAtResolution = -1
            # assumption: all files have the same resolution, so we take input file 1
            highestResolution = rf.getFinestResolution(inputFiles[0])
            if(blurFactor != -1):
                blurAtResolution = blurFactor * highestResolution
            
            self.pipeline.addStage(ma.blur(targetPipeFH, fwhm=blurAtResolution))
            
            for inputFH in inputFiles:
                
                self.pipeline.addStage(ma.blur(inputFH, fwhm=blurAtResolution))
            
                self.pipeline.addStage((ma.RotationalMinctracc(inputFH,
                                                               targetPipeFH,
                                                               blur               = blurAtResolution,
                                                               resample_step      = float(parameterList[1]),
                                                               registration_step  = float(parameterList[2]),
                                                               w_translations     = float(parameterList[3]),
                                                               rotational_range   = int(parameterList[4]),
                                                               rotational_interval= int(parameterList[5]) )))
                # if an initial model is used, we might have to concatenate
                # the transformation from the rotational minctracc command
                # with the transformation from the inital model
                if(options.init_model != None):
                    if(initModel[2] != None):
                        prevxfm = inputFH.getLastXfm(targetPipeFH)
                        newxfm = inputFH.registerVolume(initModel[0], "transforms")
                        logFile = fh.logFromFile(inputFH.logDir, newxfm)
                        self.pipeline.addStage(ma.xfmConcat([prevxfm,initModel[2]],
                                                            newxfm,
                                                            logFile))
                likeFileForResample =  targetPipeFH
                targetFHforResample = targetPipeFH
                if(options.init_model != None):
                    if(initModel[1] != None):
                        likeFileForResample = initModel[0]
                        targetFHforResample = initModel[0]
                rs = ma.mincresample(inputFH,targetFHforResample,likeFile=likeFileForResample)
                filesToAvg.append(rs.outputFiles[0])
                self.pipeline.addStage(rs)
            
        # Create the lsq6 average after all has been done...
        if(len(filesToAvg) > 1):
            lsq6AvgOutput = abspath(lsq6Directory) + "/" + "lsq6_average.mnc"
            # TODO: fix mask issue
            lsq6FH = rfh.RegistrationPipeFH(lsq6AvgOutput, mask=None, basedir=lsq6Directory)
            logBase = fh.removeBaseAndExtension(lsq6AvgOutput)
            avgLog = fh.createLogFile(lsq6FH.logDir, logBase)
            avg = ma.mincAverage(filesToAvg, lsq6AvgOutput, logFile=avgLog)
            self.pipeline.addStage(avg)


if __name__ == "__main__":
    application = LSQ6Registration()
    application.start()