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
from optparse import OptionGroup
from datetime import date
import logging
import Pyro
import sys
import re
import csv
import os

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1

def addLSQ6OptionGroup(parser):
    """
        standard options for the LSQ6 module
    """
    group = OptionGroup(parser, "LSQ6-registration options", 
                        "Options for performing a 6 parameter (rigid) registration.")
    group.add_option("--target", dest="target",
                     type="string", default=None,
                     help="File to be used as the target for the 6 parameter alignment.")
    group.add_option("--init-model", dest="init_model",
                     type="string", default=None,
                     help="File in standard space in the initial model. The initial model "
                     "can also have a file in native space and potentially a transformation "
                     "file. See our wiki for detailed information on initial models.")
    parser.set_defaults(lsq6_method="lsq6_large_rotations")
    group.add_option("--lsq6-simple", dest="lsq6_method",
                     action="store_const", const="lsq6_simple",
                     help="Run a 6 parameter alignment assuming that the input files are roughly "
                     "aligned: same space, similar orientation. [default: --lsq6-large-rotations]")
    group.add_option("--lsq6-centre-estimation", dest="lsq6_method",
                     action="store_const", const="lsq6_centre_estimation",
                     help="Run a 6 parameter alignment assuming that the input files have a "
                     "similar orientation, but are scanned in different coils/spaces. [default: --lsq6-large-rotations]")
    group.add_option("--lsq6-protocol", dest="lsq6_protocol",
                     type="string", default=None,
                     help="Specify an lsq6 protocol that overrides the default setting for stages in "
                     "the 6 parameter minctracc call. Specify the levels of blurring, simplex and "
                     "registration step sizes in mm. Use 0 and 1 to indicate whether you want to use the gradient. "
                     "For an example input csv file that can be used, see below")
    group.add_option("--lsq6-large-rotations", dest="lsq6_method",
                     action="store_const", const="lsq6_large_rotations",
                     help="Run a 6 parameter alignment assuming that the input files have a random "
                     "orientation and are scanned in different coils/spaces. A brute force search over "
                     "the x,y,z rotation space is performed to find the best 6 parameter alignment. "
                     "[default: --lsq6-large-rotations]")
    group.add_option("--lsq6-large-rotations-parameters", dest="large_rotation_parameters",
                     type="string", default="10,4,10,8,50,10",
                     help="Settings for the large rotation alignment. factor=factor based on smallest file "
                     "resolution: 1) blur factor, 2) resample step size factor, 3) registration step size "
                     "factor, 4) w_translations factor, 5) rotational range in degrees, 6) rotational "
                     "interval in degrees. [default: %default]")
    parser.add_option_group(group)
    ### sneaky trick to create a readable version of the content of an lsq6 protocol:
    epi = \
"""
Epilogue:
Example content of an lsq6 csv protocol (first three are specified in mm):

"blur";1;0.5;0.3
"simplex";4;2;1
"step";1;0.5;0.3
"gradient";False;True;False

"""
    if(parser.epilog):
        parser.epilog += epi
    else:
        parser.epilog = epi

class LSQ6Registration(AbstractApplication):
    """ 
        This class handles a 6 parameter (rigid) registration between one or more files
        and a single target or an initial model. 
        
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
            1) (--lsq6-simple) single coil, same position for all files.  If for 
            example all samples are scanned in the same coil and the orientation 
            is fixed, e.g., the subjects always enter the scanner in the same 
            position.  This implies that the input files are more or less aligned 
            to begin with.
            
            2) (--lsq6-centre-estimation) multiple coils, same position/orientation 
            for all files.  Think here for example about multiple coil live mice 
            scans.  Even though the files from different coils reside in a different 
            part in space, the orientation is the same.  This means that aligning 
            the centre of gravity for the input files gets them in option 1)
            
            3) (--lsq6-large-rotations) multiple (or single) coil with random scan 
            orientation.  In this case, the input files are in the most random 
            orientation/location.  The procedure here is to do a brute force search 
            in the x,y,z rotation space in order to find the best alignment. 
    """
    def setup_options(self):
        addLSQ6OptionGroup(self.parser)
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
        
        # Setup pipeline name and create directories.
        # TODO: Note duplication from MBM--move to function? 
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
        lsq6Directory = fh.createSubDir(self.outputDir, pipeName + "_lsq6")
        processedDirectory = fh.createSubDir(self.outputDir, pipeName + "_processed")
        
        # create file handles for the input file(s) 
        inputFiles = rf.initializeInputFiles(args, mainDirectory=processedDirectory)

        initModel = None
        if(options.target != None):
            targetPipeFH = rfh.RegistrationPipeFH(abspath(options.target), basedir=lsq6Directory)
        else: # options.init_model != None  
            initModel = rf.setupInitModel(options.init_model, self.outputDir)
            if (initModel[1] != None):
                # we have a target in "native" space 
                targetPipeFH = initModel[1]
            else:
                # we will use the target in "standard" space
                targetPipeFH = initModel[0]
        
        lsq6module = getLSQ6Module(inputFiles,
                                   targetPipeFH,
                                   lsq6Directory,
                                   initialTransform = options.lsq6_method,
                                   initModel        = initModel,
                                   lsq6Protocol     =  options.lsq6_protocol,
                                   largeRotationParameters = options.large_rotation_parameters)       
        # after the correct module has been set, get the transformation and
        # deal with resampling and potential model building
        lsq6module.createLSQ6Transformation()
        lsq6module.finalize()
        self.pipeline.addPipeline(lsq6module.p)
        
        ###########################################################################
        # TESTING
        nucorrection = NonUniformityCorrection(inputFiles, 
                                               initial_model=initModel,
                                               resampleNUCtoLSQ6=False)
        nucorrection.finalize()
        self.pipeline.addPipeline(nucorrection.p)
        
        intensity_normalization = IntensityNormalization(inputFiles,
                                                         initial_model=initModel,
                                                         resampleINORMtoLSQ6=True)
        self.pipeline.addPipeline(intensity_normalization.p)
        
#         intensity_normalization.setINORMasLastBaseVolume()
        # TESTING
        ###########################################################################

def getLSQ6Module(inputFiles,
                  targetPipeFH,
                  lsq6Directory           = None,
                  initialTransform        = None,
                  initModel               = None,
                  lsq6Protocol            = None,
                  largeRotationParameters = None):
    """
        This function serves as a switch that will return the appropriate lsq6 module depending
        on the parameters provided.  The switch is based on the parameter initialTransform.  If
        that parameter is not given, by default a large rotations lsq6 is performed. 
        
        Assumptions:
        * inputFiles are expected to be file handlers
        * targetFile is expected to be a file handler
        * the lsq6Directory is required when at least 2 inputFiles are given
        * initialTransform can have the following string values:
            - "lsq6_simple"
            - "lsq6_centre_estimation"
            - "lsq6_large_rotations"
          if none is provided, lsq6_large_rotations is taken as the default
        * lsq6Protocol: see the LSQ6HierarchicalMinctracc class for more information
          about what this can be
        * largeRotationParameters: see the RotationalMinctracc CmdStage for more information
    """
    lsq6module = None
    if(initialTransform == None):
        initialTransform = "lsq6_large_rotations"
    
    """
        Option 1) run a simple lsq6: the input files are assumed to be in the
        same space and roughly in the same orientation.
    """
    if(initialTransform == "lsq6_simple"):
        lsq6module =  LSQ6HierarchicalMinctracc(inputFiles,
                                                targetPipeFH,
                                                initial_model     = initModel,
                                                lsq6OutputDir     = lsq6Directory,
                                                initial_transform = "identity",
                                                lsq6_protocol     = lsq6Protocol)
    """
        Option 2) run an lsq6 registration where the centre of the input files
        is estimated.  Orientation is assumed to be similar, space is not.
    """
    if(initialTransform == "lsq6_centre_estimation"):
        lsq6module =  LSQ6HierarchicalMinctracc(inputFiles,
                                                targetPipeFH,
                                                initial_model     = initModel,
                                                lsq6OutputDir     = lsq6Directory,
                                                initial_transform = "estimate",
                                                lsq6_protocol     = lsq6Protocol)
    """
        Option 3) run a brute force rotational minctracc.  Input files can be
        in any random orientation and space.
    """
    if(initialTransform == "lsq6_large_rotations"):
        lsq6module = LSQ6RotationalMinctracc(inputFiles,
                                             targetPipeFH,
                                             initial_model = initModel,
                                             lsq6OutputDir = lsq6Directory,
                                             large_rotation_parameters = largeRotationParameters)
    return lsq6module

class NonUniformityCorrection(object):
    """
        
        * inputFiles: a list of either strings or file handlers.  
        
        * multiple input files possible
        
        * group name: if useOriginalInput                      -> nuc-native
                      if also resampleNUCtoLSQ6Space           -> nuc-lsq6
                      if not(useOriginalInput)                 -> nuc
        
        if the input files are file handlers, a "nuc" group will be added
        
        * initial model can be given in order to use its mask 
            - assumption: if an initial model is provided, the input files are assumed
            to be registered towards the standard space file
        
        #TODO: you should use the following boolean only if the file is not in its original
        space anymore 
        
        * useOriginalInput: this is a boolean argument which determines whether we should
                          use the original input file (file_handle.inputFileName).  If this
                          option is set to False, the last base volume will be used.  I.e.
                          there are two spaces allowed to be used:
                          
                          - native/inputfile
                          - last base volume 
        
        * mask... a single mask used for all? not yet determined...
        * if the input files have masks associated with them, the assumption is that all
          input files have an input mask (we only test the first file for the presence
          of an original input mask)
        
        masking options...
            - if an initial model is given, the assumption is that the input files
            have been registered towards the "standard" space.  This space is required
            to have a mask.  This is the mask we will for the non uniformity correction
            if no individual mask is given.  It will need to be resampled to the native
            space of the input file if *useOriginalInput* is set to True
        
    """
    def __init__(self,
                 inputFiles,
                 singlemask = None,
                 useOriginalInput = True,
                 resampleNUCtoLSQ6 = False,
                 initial_model = None):
        # TODO: allow for a single single target instead of using an initial model??
        self.p                 = Pipeline()
        self.inputs            = inputFiles
        self.initial_model     = initial_model
        self.useOriginalInput  = useOriginalInput
        self.resampleNUCtoLSQ6 = resampleNUCtoLSQ6
        self.singlemask        = singlemask
        self.masks             = None
        self.inputFilenames    = []
        self.impFields         = []
        self.NUCorrected       = []
        self.NUCorrectedLSQ6   = []
        
        # check consistency in input files
        if(self.singlemask != None):
            if(rf.isFileHandler(self.inputs[0]) and not(rf.isFileHandler(self.singlemask))):
                print "Error: the input files and single mask file for NonUniformityCorrection should be the same type. Here, inputs are file handlers, but the single mask is not (perhaps you wanted to associate this mask with the input files using the file handler?).\n"
                sys.exit()
            if(not(rf.isFileHandler(self.inputs[0])) and rf.isFileHandler(self.singlemask)):
                print "Error: the input files and single mask file for NonUniformityCorrection should be the same type. Here, inputs are not file handlers, but the single mask is.\n"
                sys.exit()
        
        if(not(self.useOriginalInput) and self.resampleNUCtoLSQ6):
            print "Warning: in NonUniformityCorrection the native files are not used, however resampleNUCtoLSQ6Space is specified. This is not necessary. Disabling this latter option. Only the \"nuc\" group will be added if file handlers are used.\n"
            self.resampleNUCtoLSQ6 = False
        
        # add the first group name (this will either be "nuc" or "nuc-native"
        if(rf.isFileHandler(self.inputs[0])):
            self.addInitialNUCGroupToInputs()
        
        # now that a new group has been added, we can use the LastBasevol in case of file handlers
        filenames = []
        if(rf.isFileHandler(self.inputs[0])):
            for inputFH in self.inputs:
                filenames.append(inputFH.getLastBasevol())
        else:
            # input files are string
            for inputFile in self.inputs:
                filenames.append(inputFile)
        self.inputFilenames = filenames
        
        # deal with masking
        if(self.singlemask != None):
            # a mask is provided that should be used for all input files
            self.masks = [self.singlemask] * len(self.inputs)
        elif(rf.isFileHandler(self.inputs[0])):
            if(self.inputs[0].getMask() != None):
                # the input files have a mask associated with them which we will use
                for inputFH in self.inputs:
                    self.masks.append(inputFH.getMask())
            elif(self.initial_model != None):
                self.setupMaskArrayWithInitialModel()
            else:
                pass
        else:
            pass
        
        self.estimateNonUniformity()
        self.evaluateNonUniformity()
        
        if(self.resampleNUCtoLSQ6):
            self.resampleNUCtoLSQ6Space()
    
    def addInitialNUCGroupToInputs(self):
        """
            adds the initial group to the file handlers.  If useOriginalInput is True, 
            the group name will be "nuc-native", otherwise it will be "nuc".
            
            Be carefule here: if we use the original file as the target for the non uniformity correction,
            the new group needs to be instantiated using that file as well as with its mask.
        """
        # create a new group to indicate in the output file names that this is the lsq6 stage
        for i in range(len(self.inputs)):
            if(self.useOriginalInput):
                self.inputs[i].newGroup(groupName="nuc-native", inputVolume=self.inputs[i].inputFileName, mask=self.inputs[i].mask)
            else:
                self.inputs[i].newGroup(groupName="nuc")
    
    def addLSQ6NUCGroupToInputs(self):
        """
            The assumption is that the input files have been corrected for non uniformity in native space.  This function
            is called just before these corrected files will be resampled to LSQ6 space.  
            
            The group name will be "nuc-lsq6"
        """
        # create a new group to indicate in the output file names that this is the lsq6 stage
        for i in range(len(self.inputs)):
            self.inputs[i].newGroup(groupName="nuc-lsq6", inputVolume=self.NUCorrected[i], mask=self.inputs[i].mask)
    
    
    def setupMaskArrayWithInitialModel(self):
        """
            This function is called when no individual masks are present for the input
            files.  However, there is an initial model available, and thus we are able
            to use the mask that is part of that.
            
            The mask to be used is the "standard space mask".  Whether or not that 
            mask needs to be resampled, depends on whether we are dealing with the
            non uniformity correction in the native image space (useOriginalInput)
        """
        masks = []
        if(self.useOriginalInput):
            # for each input file, we need to resample the standard space mask
            # to its native space
            standardModelFile = self.initial_model[0]
            for inputFH in self.inputs:
                # current assumption is that an lsq6 registration is performed and an lsq6 group is present
                # TODO: allow for differently named 6 paramter groups
                indexLsq6 = None 
                for index, value in inputFH.groupNames.iteritems():
                    if(value == "lsq6"):
                        indexLsq6 = index
                if(indexLsq6 != None):
                    # find the last transform that is associated with the standard space model
                    if(inputFH.groupedFiles[indexLsq6].transforms.has_key(standardModelFile)):
                        transformToStandardModel = inputFH.getLastXfm(standardModelFile, groupIndex=indexLsq6)
                        rs = ma.mincresampleMask(standardModelFile, 
                                                 inputFH, 
                                                 likeFile=inputFH, 
                                                 argArray=["-invert"], 
                                                 transform=transformToStandardModel, 
                                                 outputLocation=inputFH)
                        rs.name = "mincresample mask for NUC" 
                        self.p.addStage(rs)
                        masks.append(rs.outfile)
                        # add this mask to the file handler of the input file (to the original)
                        inputFH.mask = rs.outfile
                else:
                    print "Error: could not determine the transformation towards the standard-space initial model using the lsq6 group. Exiting for now.\n"
                    sys.exit()
        else:
            # if we are not using the original files, we must be using the last input files.  Given that the assumption is 
            # that the input files are aligned to the standard space, we can simply use that mask for all files
            masks = [self.initial_model[0].mask] * len(self.inputs)
        self.masks = masks
            
    def estimateNonUniformity(self):
        """
            more info... create the imp non uniformity estimation
        """
        # TODO: base things on the input file size (and allow for overwritten defaults)
        impFields = []
        for i in range(len(self.inputs)):
            inputName   = self.inputFilenames[i]
            outFileBase = fh.removeBaseAndExtension(inputName) + "_nu_estimate.imp"
            outFileDir  = self.inputs[i].tmpDir
            outFile     = fh.createBaseName(outFileDir, outFileBase)
            impFields.append(outFile)
            cmd  = ["nu_estimate", "-clobber"]
            cmd += ["-distance", "8"]
            cmd += ["-iterations", "100"]
            cmd += ["-stop", "0.0001"]
            cmd += ["-fwhm", "0.15"]
            cmd += ["-shrink", "4"]
            cmd += ["-lambda", "5.0e-02"]
            if(self.masks):
                mask = self.masks[i]
                cmd += ["-mask", InputFile(mask)]
            cmd += [InputFile(inputName)]
            cmd += [OutputFile(outFile)]
            nu_estimate = CmdStage(cmd)
            nu_estimate.colour = "red"
            self.p.addStage(nu_estimate)
        self.impFields = impFields

    def evaluateNonUniformity(self):
        """
            more info... evaluate / apply the imp field from the 
            non uniformity estimation to the input files 
        """
        nuCorrected = []
        for i in range(len(self.inputs)):
            impField = self.impFields[i]
            inputName   = self.inputFilenames[i]
            outFileBase = fh.removeBaseAndExtension(inputName) + "_nu_corrected.mnc"
            outFileDir  = self.inputs[i].resampledDir
            outFile     = fh.createBaseName(outFileDir, outFileBase)
            nuCorrected.append(outFile)
            cmd  = ["nu_evaluate", "-clobber"]
            cmd += ["-mapping", InputFile(impField)]
            cmd += [InputFile(inputName)]
            cmd += [OutputFile(outFile)]
            nu_evaluate = CmdStage(cmd)
            nu_evaluate.colour = "blue"
            self.p.addStage(nu_evaluate)
        self.NUCorrected = nuCorrected


    def resampleNUCtoLSQ6Space(self):
        """
            This function is called when useOriginalInput is True.  That means that the 
            non uniformity correction was applied to the native files.  In order to get
            to the native space, we used the "lsq6" group.  We will use that group again 
            now to resample the NUC file back to lsq6 space.
            
            Can only be called on file handlers 
        """
        if(not(rf.isFileHandler(self.inputs[0]))):
            print "Error: resampleNUCtoLSQ6Space can only be called on file handlers. Goodbye.\n"
            sys.exit()
        
        if(self.initial_model == None):
            print "Error: resampleNUCtoLSQ6Space does not know what to do without an initial model at this moment. Sorry. Goodbye.\n"
            sys.exit()
            
        # create a new group for these files
        self.addLSQ6NUCGroupToInputs()
            
        nuCorrectedLSQ6 = []
        standardModelFile = self.initial_model[0]
        for inputFH in self.inputs:
                # find the lsq6 group again
                indexLsq6 = None 
                for index, value in inputFH.groupNames.iteritems():
                    if(value == "lsq6"):
                        indexLsq6 = index
                if(indexLsq6 != None):
                    # find the last transform that is associated with the standard space model
                    if(inputFH.groupedFiles[indexLsq6].transforms.has_key(standardModelFile)):
                        transformToStandardModel = inputFH.getLastXfm(standardModelFile, groupIndex=indexLsq6)
                        outFileBase = fh.removeBaseAndExtension(inputFH.getLastBasevol()) + "_lsq6.mnc"
                        outFileDir  = inputFH.resampledDir
                        outFile     = fh.createBaseName(outFileDir, outFileBase)
                        nuCorrectedLSQ6.append(outFile)
                        rs = ma.mincresample(inputFH, 
                                             standardModelFile, 
                                             likeFile=standardModelFile,  
                                             transform=transformToStandardModel,
                                             output=outFile,
                                             argArray=["-sinc"])
                        rs.name = "mincresample NUC to LSQ6"
                        print rs.cmd 
                        self.p.addStage(rs)
        self.NUCorrectedLSQ6 = nuCorrectedLSQ6

    def finalize(self):
        """
            Sets the last base volume based on how this object was called/created.  If the files have
            been resampled to LSQ6 space, the last base volume should be NUCorrectedLSQ6, otherwise
            it should be NUCorrected
        """
        if(self.resampleNUCtoLSQ6):
            for i in range(len(self.inputs)):
                self.inputs[i].setLastBasevol(self.NUCorrectedLSQ6[i])
        else:
            for i in range(len(self.inputs)):
                self.inputs[i].setLastBasevol(self.NUCorrected[i])
                if(self.masks != None):
                    self.inputs[i].setMask(self.masks[i])    

class IntensityNormalization(object):
    """
        * mask potential single mask to be used for all input files 
        
        * inputFiles can be provided as a list of strings, or a list
        of file handlers
        
        * if the input files are file handlers, a group inorm will be added
        
        * possible options for the method (for ease of use they are simply the flag 
        used for inormalize):
            - "-ratioOfMeans"
            - "-ratioOfMedians"
            - "-meanOfRatios"
            - "-meanOfLogRatios"
            - "-medianOfRatios"

        * resampleINORMtoLSQ6 - 
        
         === can only be called on file handlers ===
         === needs an initial model in order to work ===
          
        In a way this is a special option for the class.  The intensity normalization
        class can be used on any kind of input at any stage in a pipeline.  But when running an
        image registration pipeline we use the intensity normalization right after the input
        files have been registered using 6 parameters (lsq6).  When the registration is run using
        an initial model, there will be a mask present, and after aligning the input files to that
        model, we can use that mask for the intensity normalization.  To reduce the amount of 
        resampling error in the entire pipeline, the intensity normalization (as well as the non-
        uniformity correction) should be applied in native space.  If that is the situation that
        this class is called in, resampleINORMtoLSQ6 can be set to True, and the normalized file
        will be resampled in lsq6 space, in order to continue there with the lsq12 stage.  Keep in
        mind that after lsq12, the native inormalized (and non-uniformity correction) file should 
        be resampled to lsq12.  That way we avoid one resampling step, i.e., wrong way:
        
        native-normalized -> native-normalized-in-lsq6 -> native-normalized-in-lsq6-in-lsq12 (when starting non linear stages)
        
        right way:
        
        native-normalized -> native-normalized-in-lsq6  (when starting lsq12)
        native-normalized -> native-normlaized-in-lsq12 (when starting non linear stages)
    """
    def __init__(self,
                 inputFiles,
                 mask = None,
                 inorm_const = 1000,
                 method = "-ratioOfMedians",
                 resampleINORMtoLSQ6 = False,
                 initial_model = None):
        self.p                   = Pipeline()
        self.inputs              = inputFiles
        self.masks               = None
        self.inormconst          = inorm_const
        self.method              = method
        self.resampleINORMtoLSQ6 = resampleINORMtoLSQ6
        self.initial_model       = initial_model
        self.INORM               = []
        self.INORMLSQ6           = []
        self.inputFilenames      = []
        
        # deal with input files
        if(rf.isFileHandler(self.inputs[0])):
            for inputFH in self.inputs:
                self.inputFilenames.append(inputFH.getLastBasevol())
            self.setINORMGroupToInputs()
        else:
            for inputName in self.inputs:
                self.inputFilenames.append(inputName)
        
        # once again, sort out the masking business 
        if(mask != None):
            self.masks = [mask] * len(self.inputs)
        else:
            # check whether the input files have masks associated with
            # them already
            if(self.inputs[0].getMask() != None):
                masks = []
                for i in range(len(self.inputs)):
                    masks.append(self.inputs[i].getMask())
                self.masks = masks
                
        # add the "inorm" group name
        if(rf.isFileHandler(self.inputs[0])):
            self.setINORMGroupToInputs()
    
        self.runNormalization()
        
        if(self.resampleINORMtoLSQ6):
            self.resampleINORMtoLSQ6Space()
        
        if(rf.isFileHandler(self.inputs[0])):
            self.setINORMasLastBaseVolume()
    
    def setINORMGroupToInputs(self):
        """
            make sure that by default any input file handler is 
            given the group "inorm"
        """
        # create a new group to indicate in the output file names that this is the lsq6 stage
        for i in range(len(self.inputs)):
            self.inputs[i].newGroup(groupName="inorm")    
        
    def addLSQ6INORMGroupToInputs(self):
        """
            The assumption is that the input files have been intensity normalized in native space.  This function
            is called just before these normalized files will be resampled to LSQ6 space.  
            
            The group name will be "inorm-lsq6"
        """
        # create a new group to indicate in the output file names that this is the lsq6 stage
        for i in range(len(self.inputs)):
            self.inputs[i].newGroup(groupName="inorm-lsq6", inputVolume=self.INORM[i], mask=self.inputs[i].getMask())
        
        
    def runNormalization(self):
        """
        """
        normalized = []
        for i in range(len(self.inputFilenames)):    
            inputName   = self.inputFilenames[i]
            outFileBase = fh.removeBaseAndExtension(inputName) + "_inormalized.mnc"
            outFileDir  = self.inputs[i].resampledDir
            outFile     = fh.createBaseName(outFileDir, outFileBase)
            normalized.append(outFile)
            cmd  = ["inormalize", "-clobber"]
            cmd += ["-const", self.inormconst]
            cmd += [self.method]
            if(self.masks != None):
                cmd += ["-mask", self.masks[i]]
            cmd += [InputFile(inputName)]
            cmd += [OutputFile(outFile)]
            inormalize = CmdStage(cmd)
            inormalize.colour = "yellow"
            print inormalize.cmd
            self.p.addStage(inormalize)
        self.INORM = normalized

    def resampleINORMtoLSQ6Space(self):
        """
            Can only be called on file handlers
            
            The assumption is that an lsq6 registration has been performed, and that there
            is a transformation from the native input file to the standard space in the 
            initial model.  
        """
        if(not(rf.isFileHandler(self.inputs[0]))):
            print "Error: resampleINORMtoLSQ6Space can only be called on file handlers. Goodbye.\n"
            sys.exit()
        
        if(self.initial_model == None):
            print "Error: resampleINORMtoLSQ6Space does not know what to do without an initial model at this moment. Sorry. Goodbye.\n"
            sys.exit()
            
        # create a new group for these files
        self.addLSQ6INORMGroupToInputs()
            
        INORMLSQ6 = []
        standardModelFile = self.initial_model[0]
        for inputFH in self.inputs:
                # find the lsq6 group again
                indexLsq6 = None 
                for index, value in inputFH.groupNames.iteritems():
                    if(value == "lsq6"):
                        indexLsq6 = index
                if(indexLsq6 != None):
                    # find the last transform that is associated with the standard space model
                    if(inputFH.groupedFiles[indexLsq6].transforms.has_key(standardModelFile)):
                        transformToStandardModel = inputFH.getLastXfm(standardModelFile, groupIndex=indexLsq6)
                        outFileBase = fh.removeBaseAndExtension(inputFH.getLastBasevol()) + "_lsq6.mnc"
                        outFileDir  = inputFH.resampledDir
                        outFile     = fh.createBaseName(outFileDir, outFileBase)
                        INORMLSQ6.append(outFile)
                        rs = ma.mincresample(inputFH, 
                                             standardModelFile, 
                                             likeFile=standardModelFile,  
                                             transform=transformToStandardModel,
                                             output=outFile,
                                             argArray=["-sinc"])
                        rs.name = "mincresample INORM to LSQ6"
                        print rs.cmd 
                        self.p.addStage(rs)
        self.INORMLSQ6 = INORMLSQ6

    def setINORMasLastBaseVolume(self):
        if(self.resampleINORMtoLSQ6):
            for i in range(len(self.inputs)):
                self.inputs[i].setLastBasevol(self.INORMLSQ6[i])
        else:
            for i in range(len(self.inputs)):
                self.inputs[i].setLastBasevol(self.INORM[i])
        
        

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
        
        Assumptions:
        * inputFiles are expected to be file handlers
        * targetFile is expected to be a file handler
        * the lsq6OutputDir is required when at least 2 inputFiles are given
        
        Output:
        - (self.lsq6Avg) if at least 2 input files were provided, this class will 
        store the file handler for the average in the variable lsq6Avg 
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
    
    def resampleInputFilesAndSetLastBasevol(self):
        """
            resample input files using the last transformation, and then
            set the last basevolume for the inputfile to be the output
            of the mincresample call
        """
        for inputfile in self.inputs:
            likeFileForResample =  self.target
            targetFHforResample = self.target
            if(self.initial_model != None):
                if(self.initial_model[1] != None):
                    likeFileForResample = self.initial_model[0]
                    targetFHforResample = self.initial_model[0]
            rs = ma.mincresample(inputfile,
                                 targetFHforResample,
                                 likeFile=likeFileForResample,
                                 argArray=["-sinc"])
            self.filesToAvg.append(rs.outputFiles[0])
            self.p.addStage(rs)
            #TODO: The following line might be removed when NUC is sorted out. 
            inputfile.setLastBasevol(rs.outputFiles[0])
                

    def createAverage(self):
        """
            Create the lsq6 average if at least 2 input files have been given
        """
        if(len(self.filesToAvg) > 1):
            lsq6AvgOutput = abspath(self.lsq6OutputDir) + "/" + "lsq6_average.mnc"
            # TODO: fix mask issue
            lsq6FH = rfh.RegistrationPipeFH(lsq6AvgOutput, mask=None, basedir=self.lsq6OutputDir)
            logBase = fh.removeBaseAndExtension(lsq6AvgOutput)
            avgLog = fh.createLogFile(lsq6FH.logDir, logBase)
            # Note: We are calling mincAverage here with filenames rather than file handlers
            avg = ma.mincAverage(self.filesToAvg, lsq6AvgOutput, logFile=avgLog)
            self.p.addStage(avg)
            self.lsq6Avg = lsq6FH

    def finalize(self):
        """
            Within one call, take care of the potential initial model transformation,
            the resampling of input files and create an average 
        """
        self.addNativeToStandardFromInitModel()
        self.resampleInputFilesAndSetLastBasevol()
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
            - large_rotation_parameters is the same as RotationalMinctracc() in minc_atoms,
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
        # initialize all the defaults in parent class
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


class LSQ6HierarchicalMinctracc(LSQ6Base):
    """
        This class performs an lsq6 registration using a series of 6 parameter
        minctracc calls.
        
        * Assumptions/input
            - inputFiles are provided in the form of file handlers
            - targetFile is provided as a file handler
            - if an initial model is provided, the input to the parameter
            initial_model is assumed to be the output of the function 
            setupInitModel()
            - lsq6OutputDir is a string indicating where the the potential average should go
            only needs to be provided if at least 2 input files are given
            
            - initial_transform
            This parameter can be either "estimate" or "identity". When "estimate" is 
            provided, the assumption is that the files are not in the same part in space,
            and the centre of gravity of the files will be used to determine the initial
            alignment.  When "identity" is given, the input files are already assumed to 
            be roughly aligned.
            
            - lsq6_protocol
            By default the minctracc calls that will be run are based on the resolution 
            of the input files. 5 generations are run for "estimate", and 3 generations
            are run in the case of "identity".  If you want to override these default
            settings, you can specify the path to a csv file containing the desired 
            settings using this parameter.  For more information about the defaults,
            see below.  For an example of an lsq6 protocol, see the help for the 
            LSQ6Registration class. 
        
        Output:
            - (self.lsq6Avg) if at least 2 input files were provided, this class will 
            store the file handler for the average in the variable lsq6Avg 
    """
    def __init__(self,
                 inputFiles,
                 targetFile,
                 initial_model     = None,
                 lsq6OutputDir     = None,
                 initial_transform = "estimate",
                 lsq6_protocol     = None):
        # initialize all the defaults in the parent class
        LSQ6Base.__init__(self, inputFiles, targetFile, initial_model, lsq6OutputDir)
        self.initial_transform = initial_transform 
        self.lsq6_protocol     = lsq6_protocol
        
        self.setInitialMinctraccTransform()
        
        self.setHierarchyOptions()
        
    def setInitialMinctraccTransform(self):
        # set option which will by used by the minctracc CmdStage:
        self.linearparam = None
        if(self.initial_transform == "estimate"):
            self.linearparam = "lsq6"
        elif(self.initial_transform == "identity"):
            self.linearparam = "lsq6-identity"
        else:
            print "Error: unknown option used for inital_transform in LSQ6HierarchicalMinctracc: ", self.initial_transform, ". Can be \"estimate\" or \"identity\".\nGoodbye."
            sys.exit()
    
    def setHierarchyOptions(self):
        """
            It is possible to specify an lsq6 protocol (csv file, see the help in the LSQ6Registration class
            for an example.  If none is specified, defaults will be used as follows:
            
            ************************* DEFAULTS ************************************
            * all parameters will be set as a factor of the input file's resolution
            
            *** estimate ***
            When using the estimation of centres, 5 stages:
            blur    = [ 90, 35, 17,  9,  4]     # in mm at 56micron files: [5.04,  1.96, 0.952, 0.504, 0.224]
            gradient= [  0,  0,  0,  1,  0]
            simplex = [128, 64, 40, 28, 16]     # in mm at 56mircon files: [7.168, 3.584, 2.24, 1.568, 0.896]
            step    = [ 90, 35, 17,  9,  4]
            
            *** identity ***
            When using the identity transformation to initialize minctracc, 3 stages:
            blur    = [17,  9,  4]     # in mm at 56micron files: [0.952, 0.504, 0.224]
            gradient= [ 0,  1,  0]
            simplex = [40, 28, 16]     # in mm at 56mircon files: [2.24, 1.568, 0.896]
            step    = [17,  9,  4]
        """
        if(self.lsq6_protocol):
            # read the protocol and set the parameters accordingly
            """Set parameters from specified protocol"""
        
            """Read parameters into array from csv."""
            inputCsv  = open(abspath(self.lsq6_protocol), 'rb')
            csvReader = csv.reader(inputCsv, delimiter=';', skipinitialspace=True)
            params    = []
            for r in csvReader:
                params.append(r)
            """initialize arrays """
            self.blur     = []
            self.gradient = []
            self.step     = []
            self.simplex  = []

            """Parse through rows and assign appropriate values to each parameter array.
               Everything is read in as strings, but in some cases, must be converted to 
               floats, booleans or gradients. 
            """            
            for p in params:
                # first make sure that we skip empty lines in the input csv...
                if(len(p) == 0):
                    pass
                elif p[0]=="blur":
                    """Blurs must be converted to floats."""
                    for i in range(1,len(p)):
                        self.blur.append(float(p[i]))
                elif p[0]=="gradient":
                    """Gradients must be converted to bools."""
                    for i in range(1,len(p)):
                        if p[i]=="True" or p[i]=="TRUE":
                            self.gradient.append(True)
                        elif p[i]=="False" or p[i]=="FALSE":
                            self.gradient.append(False)
                        else:
                            print "Improper parameter specified for the gradient in the lsq6 protocol: ", str(p[i]), " Please speciy True, TRUE, False or FALSE.\nGoodbye"
                            sys.exit()
                elif p[0]=="simplex":
                    """Simplex must be converted to floats."""
                    for i in range(1,len(p)):
                        self.simplex.append(float(p[i]))
                elif p[0]=="step":
                    """Steps must be converted to floats."""
                    for i in range(1,len(p)):
                        self.step.append(float(p[i]))
                else:
                    print "Improper parameter specified for lsq6 protocol: ", str(p[0]), "\nGoodbye"
                    sys.exit()
            
            # now that all the parameters have been set, make sure we have an
            # equal number of inputs for all the settings:
            self.generations = max(len(self.blur),
                                   len(self.gradient),
                                   len(self.simplex),
                                   len(self.step))
            
            if(len(self.blur) < self.generations):
                print "Not all parameters in the lsq6 protocol are the same, \"blur\" has to few.\nGoodbye"
                sys.exit()
            if(len(self.gradient) < self.generations):
                print "Not all parameters in the lsq6 protocol are the same, \"gradient\" has to few.\nGoodbye"
                sys.exit()
            if(len(self.simplex) < self.generations):
                print "Not all parameters in the lsq6 protocol are the same, \"simplex\" has to few.\nGoodbye"
                sys.exit()
            if(len(self.step) < self.generations):
                print "Not all parameters in the lsq6 protocol are the same, \"step\" has to few.\nGoodbye"
                sys.exit()
            
        else:
            # use defaults:
            est_blurfactors      = [   90,   35,   17,   9,    4]
            est_simplexfactors   = [  128,   64,   40,  28,   16]
            est_stepfactors      = [   90,   35,   17,   9,    4]
            est_gradientdefaults = [False,False,False,True,False]
            
            id_blurfactors      = [   17,   9,    4]
            id_simplexfactors   = [   40,  28,   16]
            id_stepfactors      = [   17,   9,    4]
            id_gradientdefaults = [False,True,False]
            
            resolution       = rf.getFinestResolution(self.inputs[0])
            
            if(self.initial_transform == "identity"):
                self.blur        = [i * resolution for i in id_blurfactors]
                self.simplex     = [i * resolution for i in id_simplexfactors]
                self.step        = [i * resolution for i in id_stepfactors]
                self.gradient    = id_gradientdefaults
            else: # assume "estimate"
                self.blur        = [i * resolution for i in est_blurfactors]
                self.simplex     = [i * resolution for i in est_simplexfactors]
                self.step        = [i * resolution for i in est_stepfactors]
                self.gradient    = est_gradientdefaults
            
            self.generations = len(self.blur)
    
    def createLSQ6Transformation(self):
        """
            Assumption: setHierarchyOptions() has been executed prior to calling
            this function
            
            This function ties it all together.  The settings for each of the
            stages for minctracc have been set by setHierarchyOptions().  Here
            all the blurs are created and the minctracc calls are made.
        """
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
                                  gradient    = self.gradient[i],
                                  linearparam = "lsq6") 
                self.p.addStage(mt)


        

if __name__ == "__main__":
    application = LSQ6Registration()
    application.start()
