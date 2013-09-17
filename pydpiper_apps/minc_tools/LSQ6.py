#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.hierarchical_minctracc as hm
import pydpiper_apps.minc_tools.minc_atoms as ma
from os.path import splitext, abspath
import logging
import Pyro
from optparse import OptionGroup
import sys
import re
import os

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1

class LSQ6Registration(AbstractApplication):
    """ 
        This class handles a 6 parameter (rigid) registration between two files. 
        
        Source:
            Single input file with or without a mask
        
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
        self.parser.set_usage("%prog [options] source.mnc [--target target.mnc or --init-model /init/model/file.mnc]") 

    def setup_appName(self):
        appName = "LSQ6-registration"
        return appName
    
    def run(self):
        options = self.options
        args = self.args
        
        # initial error handling:  verify that only one input file is specified and that it is a MINC file
        if(len(args) < 1):
            print "Error: no source image provided\n"
            sys.exit()
        elif(len(args) > 1):
            print "Error: more than one source image provided: ", args, "\n"
            sys.exit()
        ext = splitext(args[0])[1]
        if(re.match(".mnc", ext) == None):
            print "Error: input file is not a MINC file:, ", args[0], "\n"
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
        
        # not sure yet what the best directory structure for all of this is...
        # create file handles for the source and target:
        inputPipeFH = rf.initializeInputFiles([abspath(args[0])], mainDirectory=mainDirectory)
        targetPipeFH = rf.initializeInputFiles([abspath(options.target)], mainDirectory=mainDirectory)
        
        # create a new group to indicate in the output file names that this is the lsq6 stage
        inputPipeFH[0].newGroup(groupName="lsq6")
        
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
            highestResolution = rf.getHighestResolution(inputPipeFH[0])
            if(blurFactor != -1):
                blurAtResolution = blurFactor * highestResolution
            
            self.pipeline.addStage(ma.blur(inputPipeFH[0], fwhm=blurAtResolution))
            self.pipeline.addStage(ma.blur(targetPipeFH[0], fwhm=blurAtResolution))
            
            self.pipeline.addStage((hm.RotationalMinctracc(inputPipeFH[0],
                                                           targetPipeFH[0],
                                                           blur               = float(parameterList[0]),
                                                           resample_step      = float(parameterList[1]),
                                                           registration_step  = float(parameterList[2]),
                                                           w_translations     = float(parameterList[3]),
                                                           rotational_range   = int(parameterList[4]),
                                                           rotational_interval= int(parameterList[5]) )))

            self.pipeline.addStage(ma.mincresample(inputPipeFH[0],
                                                   targetPipeFH[0],
                                                   likeFile=targetPipeFH[0]))


if __name__ == "__main__":
    application = LSQ6Registration()
    application.start()