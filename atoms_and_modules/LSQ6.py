#!/usr/bin/env python

from __future__ import print_function
from pydpiper.application import AbstractApplication
from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.minc_parameters as mp
from os.path import abspath
import logging
import sys

logger = logging.getLogger(__name__)

def verifyCorrectLSQ6TargetOptions(bootstrap, init_model, lsq6_target):
    """
    This function can be called using the parameters that are set using 
    the flags:
    --bootstrap
    --init-model
    --lsq6-target
    
    it will check that exactly one of the options is provided and exits 
    otherwise. 
    """
    # check how many options have been specified that can be used as the initial target
    number_of_target_options = sum((bootstrap   != False,
                                    init_model  != None,
                                    lsq6_target != None))
    if(number_of_target_options == 0):
        print("\nError: please specify a target for the 6 parameter alignmnet. Options are: --lsq6-target, --init-model, --bootstrap.\n")
        sys.exit()
    if(number_of_target_options > 1):
        print("\nError: please specify only one of the following options: --lsq6-target, --init-model, --bootstrap. Don't know which target to use...\n")
        sys.exit()

def addLSQ6ArgumentGroup(parser):
    """
        standard options for the LSQ6 module
    """
    group = parser.add_argument_group("LSQ6-registration options", "Options for performing a 6 parameter (rigid) registration.")
    group.add_argument("--lsq6-target", dest="lsq6_target",
                     type=str, default=None,
                     help="File to be used as the target for the 6 parameter alignment. [Default = %(default)s]")
    group.add_argument("--init-model", dest="init_model",
                     type=str, default=None,
                     help="File in standard space in the initial model. The initial model "
                     "can also have a file in native space and potentially a transformation "
                     "file. See our wiki for detailed information on initial models. [Default = %(default)s]")
    group.add_argument("--bootstrap", dest="bootstrap",
                     action="store_true", default=False,
                     help="Use the first inputfile to the pipeline as the target for the 6 parameter alignment. [Default = %(default)s]")
    parser.set_defaults(lsq6_method="lsq6_large_rotations")
    parser.set_defaults(nuc=True)
    parser.set_defaults(inormalize=True)
    group.add_argument("--lsq6-simple", dest="lsq6_method",
                     action="store_const", const="lsq6_simple",
                     help="Run a 6 parameter alignment assuming that the input files are roughly "
                     "aligned: same space, similar orientation. Keep in mind that if you use an "
                     "initial model with both a standard and a native space, the assumption is "
                     "that the input files are already roughly aligned to the native space. "
                     "Three iterations are run: 1st is 17 times stepsize blur, 2nd is 9 times "
                     "stepsize gradient, 3rd is 4 times stepsize blur. [Default = %(default)s]")
    group.add_argument("--lsq6-centre-estimation", dest="lsq6_method",
                     action="store_const", const="lsq6_centre_estimation",
                     help="Run a 6 parameter alignment assuming that the input files have a "
                     "similar orientation, but are scanned in different coils/spaces. [Default = %(default)s]")
    group.add_argument("--lsq6-large-rotations", dest="lsq6_method",
                     action="store_const", const="lsq6_large_rotations",
                     help="Run a 6 parameter alignment assuming that the input files have a random "
                     "orientation and are scanned in different coils/spaces. A brute force search over "
                     "the x,y,z rotation space is performed to find the best 6 parameter alignment. "
                     "[Default = %(default)s]")
    group.add_argument("--lsq6-large-rotations-parameters", dest="large_rotation_parameters",
                     type=str, default="10,4,10,8",
                     help="Settings for the large rotation alignment. factor=factor based on smallest file "
                     "resolution: 1) blur factor, 2) resample step size factor, 3) registration step size "
                     "factor, 4) w_translations factor  ***** if you are working with mouse brain data "
                     " the defaults do not have to be based on the file resolution; a default set of "
                     " settings works for all mouse brain. In order to use those setting, specify: \"mousebrain\""
                     " as the argument for this option. ***** [default = %(default)s]")
    group.add_argument("--lsq6-rotational-range", dest="large_rotation_range",
                     type=int, default=50,
                     help="Settings for the rotational range in degrees when running the large rotation alignment."
                     " [Default = %(default)s]")
    group.add_argument("--lsq6-rotational-interval", dest="large_rotation_interval",
                     type=int, default=10,
                     help="Settings for the rotational interval in degrees when running the large rotation alignment."
                     " [Default = %(default)s]")
    group.add_argument("--nuc", dest="nuc",
                      action="store_true", 
                      help="Perform non-uniformity correction. [Default = %(default)s]")
    group.add_argument("--no-nuc", dest="nuc",
                      action="store_false", 
                      help="If specified, do not perform non-uniformity correction. Opposite of --nuc.")
    group.add_argument("--inormalize", dest="inormalize",
                      action="store_true", 
                      help="Normalize the intensities after lsq6 alignment and nuc, if done. [Default = %(default)s] ")
    group.add_argument("--no-inormalize", dest="inormalize",
                      action="store_false", 
                      help="If specified, do not perform intensity normalization. Opposite of --inormalize.")
    group.add_argument("--lsq6-protocol", dest="lsq6_protocol",
                      type=str, default=None,
                      help="Specify an lsq6 protocol that overrides the default setting for stages in "
                      "the 6 parameter minctracc call. Parameters must be specified as in the following \n"
                      "example: applications_testing/test_data/minctracc_example_linear_protocol.csv \n"
                      "[Default = %(default)s].")

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
        """Add option groups from specific modules"""
        rf.addGenRegArgumentGroup(self.parser)
        addLSQ6ArgumentGroup(self.parser)

    def setup_appName(self):
        appName = "LSQ6-registration"
        return appName
    
    def run(self):
        options = self.options

        rf.checkInputFiles(options.files)

        verifyCorrectLSQ6TargetOptions(options.bootstrap,
                                       options.init_model,
                                       options.lsq6_target)

        # Setup output directories for LSQ6 registration.        
        dirs = rf.setupDirectories(self.outputDir, options.pipeline_name, module="LSQ6")
        
        # create file handles for the input file(s) 
        inputFiles = rf.initializeInputFiles(options.files, dirs.processedDir, maskDir=options.mask_dir)

        # if we are running a bootstrap or lsq6_target option, pass in the correct target
        target_file_for_lsq6 = None
        target_file_directory = None
        if(options.bootstrap):
            target_file_for_lsq6 = inputFiles[0].inputFileName
            target_file_directory = fh.createSubDir(self.outputDir,options.pipeline_name + "_bootstrap_file")
        if(options.lsq6_target):
            target_file_for_lsq6 = options.lsq6_target
            target_file_directory = fh.createSubDir(self.outputDir,options.pipeline_name + "_target_file")

        #Setup init model and inital target. Function also exists if no target was specified.
        initModel, targetPipeFH = rf.setInitialTarget(options.init_model, 
                                                      target_file_for_lsq6, 
                                                      target_file_directory,
                                                      self.outputDir,
                                                      options.pipeline_name)
        
        # Initialize LSQ6, NonUniformityCorrection and IntensityNormalization classes and
        # construct their pipelines. Note that because we read in the options directly, running
        # NUC and inormalize is still optional 
        runLSQ6NucInorm = LSQ6NUCInorm(inputFiles,
                                       targetPipeFH,
                                       initModel, 
                                       dirs.lsq6Dir, 
                                       options)
        self.pipeline.addPipeline(runLSQ6NucInorm.p)
        

def getLSQ6Module(inputFiles,
                  targetPipeFH,
                  options,  # TODO could narrow this to only queue_type
                  lsq6Directory           = None,
                  initialTransform        = None,
                  initModel               = None,
                  lsq6Protocol            = None,
                  largeRotationParameters = None,
                  largeRotationRange      = None,
                  largeRotationInterval   = None):
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
                                                options,
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
                                                options,
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
                                             options,
                                             initial_model = initModel,
                                             lsq6OutputDir = lsq6Directory,
                                             large_rotation_parameters = largeRotationParameters,
                                             large_rotation_range      = largeRotationRange,
                                             large_rotation_interval   = largeRotationInterval)
    return lsq6module

class LSQ6NUCInorm(object):
    """Because LSQ6 is often called in conjunction with NUC and IntensityNormalization, this class
       enables the calling of both in a single class. This is intended to reduce repeated code and 
       simplify reading/writing at the highest levels."""
    def __init__(self, 
                 inputFiles, 
                 targetPipeFH,
                 initModel, 
                 lsq6Directory,
                 options):
        self.p = Pipeline()
        self.inputFiles = inputFiles
        self.target = targetPipeFH
        self.initModel = initModel
        self.lsq6Dir = lsq6Directory
        self.options = options 
        self.lsq6Avg = None # add the average from the lsq6 module here
        
        self.setupPipeline()
    
    def setupPipeline(self):
        lsq6module = getLSQ6Module(self.inputFiles,
                                   self.target,
                                   options=self.options,
                                   lsq6Directory=self.lsq6Dir,
                                   initialTransform = self.options.lsq6_method,
                                   initModel        = self.initModel,
                                   lsq6Protocol     =  self.options.lsq6_protocol,
                                   largeRotationParameters = self.options.large_rotation_parameters,
                                   largeRotationRange      = self.options.large_rotation_range,
                                   largeRotationInterval   = self.options.large_rotation_interval)       
        # after the correct module has been set, get the transformation and
        # deal with resampling and potential model building
        lsq6module.createLSQ6Transformation()
        lsq6module.finalize()
        self.p.addPipeline(lsq6module.p)
        self.lsq6Avg = lsq6module.lsq6Avg
        
        if self.options.nuc:
            already_resample_to_LSQ6 = False
            if not self.options.inormalize:
                # the non uniformity correction is performed in native space 
                # by default (useOriginalInput=True). If we are not also 
                # performing intensity normalization, we need to already
                # resample the non uniformity corrected file into LSQ6 space
                already_resample_to_LSQ6 = True
            nucorrection = NonUniformityCorrection(self.inputFiles, 
                                                   initial_model=self.initModel,
                                                   resampleNUCtoLSQ6=already_resample_to_LSQ6,
                                                   targetForLSQ6=self.target)
            nucorrection.finalize()
            self.p.addPipeline(nucorrection.p)
        
        if self.options.inormalize:
            need_to_resample_to_LSQ6 = True;
            # Currently when no non-uniformity correction is applied, the input 
            # file is intensity normalized in lsq6 space, not native space. This 
            # means that in that case, we do not need to resample to LSQ6 anymore
            # since the file is already in that space:
            if not self.options.nuc:
                need_to_resample_to_LSQ6 = False
            intensity_normalization = IntensityNormalization(self.inputFiles,
                                                             initial_model=self.initModel,
                                                             resampleINORMtoLSQ6=need_to_resample_to_LSQ6,
                                                             targetForLSQ6=self.target, options=self.options)
            self.p.addPipeline(intensity_normalization.p)

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
            
        * targetForLSQ6: if no initial model is used, but there is a target file (either specified
        through --lsq6-target or --bootstrap), we can get the lsq6 transformation based on that
        file in case we want to resample to LSQ6 space
        
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
                 initial_model = None,
                 targetForLSQ6 = None):
        # TODO: allow for a single single target instead of using an initial model??
        self.p                 = Pipeline()
        self.inputs            = inputFiles
        self.initial_model     = initial_model
        self.targetForLSQ6     = targetForLSQ6
        self.useOriginalInput  = useOriginalInput
        self.resampleNUCtoLSQ6 = resampleNUCtoLSQ6
        self.singlemask        = singlemask
        self.masks             = None
        self.inputFilenames    = []
        self.impFields         = []
        self.NUCorrected       = []
        self.NUCorrectedLSQ6   = []
        self.NUCorrectedLSQ6Masks = []
        
        # check consistency in input files
        if(self.singlemask != None):
            if(rf.isFileHandler(self.inputs[0]) and not(rf.isFileHandler(self.singlemask))):
                print("Error: the input files and single mask file for NonUniformityCorrection should be the same type. Here, inputs are file handlers, but the single mask is not (perhaps you wanted to associate this mask with the input files using the file handler?).\n")
                sys.exit()
            if(not(rf.isFileHandler(self.inputs[0])) and rf.isFileHandler(self.singlemask)):
                print("Error: the input files and single mask file for NonUniformityCorrection should be the same type. Here, inputs are not file handlers, but the single mask is.\n")
                sys.exit()
        
        if(not(self.useOriginalInput) and self.resampleNUCtoLSQ6):
            print("Warning: in NonUniformityCorrection the native files are not used, however resampleNUCtoLSQ6Space is specified. This is not necessary. Disabling this latter option. Only the \"nuc\" group will be added if file handlers are used.\n")
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
                self.masks = []
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
                    print("Error: could not determine the transformation towards the standard-space initial model using the lsq6 group. Exiting for now.\n")
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
            nu_estimate.setLogFile(LogFile(fh.logFromFile(self.inputs[i].logDir, outFile)))
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
            outFileBase = fh.removeBaseAndExtension(inputName) + "_nuc.mnc"
            outFileDir  = self.inputs[i].resampledDir
            outFile     = fh.createBaseName(outFileDir, outFileBase)
            nuCorrected.append(outFile)
            cmd  = ["nu_evaluate", "-clobber"]
            cmd += ["-mapping", InputFile(impField)]
            cmd += [InputFile(inputName)]
            cmd += [OutputFile(outFile)]
            nu_evaluate = CmdStage(cmd)
            nu_evaluate.colour = "blue"
            nu_evaluate.setLogFile(LogFile(fh.logFromFile(self.inputs[i].logDir, outFile)))
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
            print("Error: resampleNUCtoLSQ6Space can only be called on file handlers. Goodbye.\n")
            sys.exit()        

        if(self.initial_model == None and self.targetForLSQ6 == None):
            print("Error: resampleNUCtoLSQ6Space does not know what to do without an initial model and without a target file for the LSQ6 stage. Sorry. Goodbye.\n")
            sys.exit()
            
        # create a new group for these files
        self.addLSQ6NUCGroupToInputs()
            
        nuCorrectedLSQ6 = []
        nuCorrectedLSQ6Masks = []
        lsq6SpaceTarget = None
        if self.initial_model:
            lsq6SpaceTarget = self.initial_model[0]
        else:
            lsq6SpaceTarget = self.targetForLSQ6
        for inputFH in self.inputs:
                # find the lsq6 group again
                indexLsq6 = None 
                for index, value in inputFH.groupNames.iteritems():
                    if(value == "lsq6"):
                        indexLsq6 = index
                if(indexLsq6 != None):
                    # find the last transform that is associated with the standard space model
                    if(inputFH.groupedFiles[indexLsq6].transforms.has_key(lsq6SpaceTarget)):
                        transformToLSQ6 = inputFH.getLastXfm(lsq6SpaceTarget, groupIndex=indexLsq6)
                        outFileBase = fh.removeBaseAndExtension(inputFH.getLastBasevol()) + "_lsq6.mnc"
                        outFileDir  = inputFH.resampledDir
                        outFile     = fh.createBaseName(outFileDir, outFileBase)
                        nuCorrectedLSQ6.append(outFile)
                        resamplings = ma.mincresampleFileAndMask(inputFH,
                                                                 lsq6SpaceTarget,
                                                                 nameForStage="mincresample NUC to LSQ6",
                                                                 likeFile=lsq6SpaceTarget,  
                                                                 transform=transformToLSQ6,
                                                                 output=outFile,
                                                                 argArray=["-sinc"])
                        nuCorrectedLSQ6Masks.append(resamplings.outputFilesMask[0])
                        self.p.addPipeline(resamplings.p)
                    else:
                        # not good either...
                        print("\nError: can not find the 6 parameter transformation between: %s and %s (in the function: resampleNUCtoLSQ6Space). Exiting now...\n" % (inputFH.inputFileName, lsq6SpaceTarget.inputFileName))
                        sys.exit()
                else:
                    #oops...
                    print("\nError: we were not able to find the \"lsq6\" class of files during the non uniformity correction stage. Were trying to determine the transformation that resamples the non uniformity corrected file in native space to LSQ6 space. Exiting now...\n\n")
                    sys.exit()
        self.NUCorrectedLSQ6 = nuCorrectedLSQ6
        self.NUCorrectedLSQ6Masks = nuCorrectedLSQ6Masks

    def finalize(self):
        """
            Sets the last base volume based on how this object was called/created.  If the files have
            been resampled to LSQ6 space, the last base volume should be NUCorrectedLSQ6, otherwise
            it should be NUCorrected
        """
        if(self.resampleNUCtoLSQ6):
            for i in range(len(self.inputs)):
                # the NUC (and potentially the mask) files have been
                # resampled to LSQ6 space.  That means that we can set
                # the last basevol using the last resampled files (i.e.,
                # no arguments to setLastBasevol)
                self.inputs[i].setLastBasevol()
        else:
            for i in range(len(self.inputs)):
                # specify the last basevol explicitly, because applying
                # the non uniformity correction does not resampled the file 
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

        * similarly to what is done in the non uniformity correction class, a target file for 
        the LSQ6 stage can be specified. That way we still know which transformation to use 
        when resampleINORMtoLSQ6 is provided. 

        * resampleINORMtoLSQ6 - 
        
         === can only be called on file handlers ===
          
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
                 options,
                 mask = None,
                 inorm_const = 1000,
                 method = "-ratioOfMedians",
                 resampleINORMtoLSQ6 = False,
                 initial_model = None,
                 targetForLSQ6 = None):
        self.p                   = Pipeline()
        self.options             = options
        self.inputs              = inputFiles
        self.masks               = None
        self.inormconst          = inorm_const
        self.method              = method
        self.resampleINORMtoLSQ6 = resampleINORMtoLSQ6
        self.initial_model       = initial_model
        self.targetForLSQ6       = targetForLSQ6
        self.INORM               = []
        self.INORMLSQ6           = []
        self.INORMLSQ6Masks      = []
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
            outFileBase = fh.removeBaseAndExtension(inputName) + "_inorm.mnc"
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
            inormalize.setLogFile(LogFile(fh.logFromFile(self.inputs[i].logDir, outFile)))
            self.p.addStage(inormalize)
        self.INORM = normalized

    def resampleINORMtoLSQ6Space(self):
        """
            Can only be called on file handlers
            
            The assumption is that an lsq6 registration has been performed, and that there
            is a transformation from the native input file to the standard space in the 
            initial model /// or that we have the 6 parameter transform from native to
            the pipeline target (possibly given by --lsq6-target or --bootstrap)
        """
        if(not(rf.isFileHandler(self.inputs[0]))):
            print("Error: resampleINORMtoLSQ6Space can only be called on file handlers. Goodbye.\n")
            sys.exit()
        
        if(self.initial_model == None and self.targetForLSQ6 == None):
            print("Error: resampleINORMtoLSQ6Space does not know what to do without an initial model and without a target file for the LSQ6 stage. Sorry. Goodbye.\n")
            sys.exit()
            
        # create a new group for these files
        self.addLSQ6INORMGroupToInputs()
            
        INORMLSQ6 = []
        INORMLSQ6Masks = []
        if self.initial_model:
            lsq6SpaceTarget = self.initial_model[0]
        else:
            lsq6SpaceTarget = self.targetForLSQ6
        for inputFH in self.inputs:
                # find the lsq6 group again
                indexLsq6 = None 
                for index, value in inputFH.groupNames.iteritems():
                    if(value == "lsq6"):
                        indexLsq6 = index
                if(indexLsq6 != None):
                    # find the last transform that is associated with the standard space model
                    if(inputFH.groupedFiles[indexLsq6].transforms.has_key(lsq6SpaceTarget)):
                        transformToLSQ6 = inputFH.getLastXfm(lsq6SpaceTarget, groupIndex=indexLsq6)
                        outFileBase = fh.removeBaseAndExtension(inputFH.getLastBasevol()) + "_lsq6.mnc"
                        outFileDir  = inputFH.resampledDir
                        outFile     = fh.createBaseName(outFileDir, outFileBase)
                        INORMLSQ6.append(outFile)
                        resamplings = ma.mincresampleFileAndMask(inputFH, 
                                                                 lsq6SpaceTarget, 
                                                                 nameForStage = "mincresample INORM to LSQ6",
                                                                 likeFile=lsq6SpaceTarget,  
                                                                 transform=transformToLSQ6,
                                                                 output=outFile,
                                                                 argArray=["-sinc"])
                        INORMLSQ6Masks.append(resamplings.outputFilesMask[0])
                        self.p.addPipeline(resamplings.p)
                    else:
                        # not good either...
                        print("\nError: can not find the 6 parameter transformation between: %s and %s (in the function: resampleINORMtoLSQ6Space). Exiting now...\n" % (inputFH.inputFileName, lsq6SpaceTarget.inputFileName))
                        sys.exit()
                else:
                    # oops...
                    print("\nError: we were not able to find the \"lsq6\" class of files during the inormalize stage. Were trying to determine the transformation that resamples the intensity normalized file in native space to LSQ6 space. Exiting now...\n\n")
                    sys.exit()
        self.INORMLSQ6 = INORMLSQ6
        self.INORMLSQ6Masks = INORMLSQ6Masks

    def setINORMasLastBaseVolume(self):
        if(self.resampleINORMtoLSQ6):
            for i in range(len(self.inputs)):
                # the INORM (and potentially the mask) files have been
                # resampled to LSQ6 space.  That means that we can set
                # the last basevol using the last resampled files (i.e.,
                # no arguments to setLastBasevol)
                self.inputs[i].setLastBasevol()                
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
                 options,
                 initial_model = None,
                 lsq6OutputDir = None):
        self.p              = Pipeline()
        self.options        = options
        self.inputs         = inputFiles
        self.target         = targetFile
        self.initial_model  = initial_model
        self.lsq6OutputDir  = lsq6OutputDir 
        self.lsq6Avg        = None # will contain file handler to lsq6 average
        self.filesToAvg     = [] # store the resampled lsq6 files in order to create an average at the end
        
        self.check_inputs()
        self.check_lsq6_folder()
        self.setLSQ6GroupToInputs()
        
        # Get file resolution for each file: will need for subclasses. 
        # We base a number of parameters on the resolution of the target files / input files.
        # This is mostly for minctracc calls which needs input files to be blurred and we set
        # the -step parameter for minctracc based on the resolution of the target for the 
        # registration. As such, we should simply use the targetFile for the registration 
        # to base these parameters on
        self.fileRes = rf.returnFinestResolution(self.target)
        
    def check_inputs(self):
        """
            Currently only file handlers are covered in these classes.  This function
            check whether both input and target files are given as file handlers 
        """
        # input file checking...    
        if(not(type(self.inputs) is list)):
            if(not rf.isFileHandler(self.inputs)):
                print("My apologies... the LSQ6 modules currently only work with file handlers (input file)...\nGoodbye")
                sys.exit()
            else:
                # for ease of use, turn into a list
                self.inputs = [self.inputs]
        else:
            for inputfile in self.inputs:
                if(not rf.isFileHandler(inputfile)):
                    print("My apologies... the LSQ6 modules currently only work with file handlers (input files)...\nGoodbye")
                    sys.exit()
        if(not rf.isFileHandler(self.target)):
            print("My apologies... the LSQ6 modules currently only work with file handlers (target file)...\nGoodbye")
            sys.exit()
    
    def check_lsq6_folder(self):
        """
            Make sure that the output directory for the lsq6 average is 
            defined is at least 2 input files are provided
        """
        if(len(self.inputs) > 1 and not(self.lsq6OutputDir)):
            print("Error: ", len(self.inputs), " input files were provided to the LSQ6 module but no output directory for the average was given. Don't know where to put it...\nGoodbye.")
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
                    if not isinstance(inputFH, rfh.RegistrationPipeFH):
                        raise TypeError("only works with file handlers")
                    else:
                        prevxfm = inputFH.getLastXfm(self.target)                            # pylint: disable=E1101
                        newxfm = inputFH.registerVolume(self.initial_model[0], "transforms") # pylint: disable=E1101
                        logFile = fh.logFromFile(inputFH.logDir, newxfm)                     # pylint: disable=E1101
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
            resamplings = ma.mincresampleFileAndMask(inputfile,
                                                     targetFHforResample,
                                                     likeFile=likeFileForResample,
                                                     argArray=["-sinc"])
            self.filesToAvg.append(resamplings.outputFiles[0])
            self.p.addPipeline(resamplings.p)
            # no arguments need to be provided to setLastBasevol: the last
            # resampled file will be used. If a mask was also resampled
            # during the last stage, this will be updated as well.
            inputfile.setLastBasevol() # pylint: disable=E1101
                

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
            avg = ma.average(self.filesToAvg, outputAvg=lsq6AvgOutput, logFile=avgLog,
                             queue_type=self.options.queue_type)
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
                 options,
                 initial_model = None,
                 lsq6OutputDir = None,
                 large_rotation_parameters="10,4,10,8",
                 large_rotation_range     = 50,
                 large_rotation_interval  = 10):
        # initialize all the defaults in parent class
        LSQ6Base.__init__(self, inputFiles, targetFile, options,
                          initial_model=initial_model,
                          lsq6OutputDir=lsq6OutputDir)
        
        self.parameters        = large_rotation_parameters
        self.rotation_range    = large_rotation_range
        self.rotation_interval = large_rotation_interval
        
    def createLSQ6Transformation(self):    
        # We should take care of the appropriate amount of blurring
        # for the input files here
        
        # assumption: all files have the same resolution, so we take input file 1
        highestResolution = self.fileRes
        blurAtResolution = -1
        resampleStepFactor = -1
        registrationStepFactor = -1
        wTranslationsFactor = -1
        
        # first... if we are dealing with mouse brains, we should set the defaults
        # not on the actual resolution of the files, but on the best parameter set
        # for mouse brains. Still we have to feed those in as factors, so we have
        # to do a bit of a silly conversion from resolution to factors now 
        if(self.parameters == "mousebrain"):
            # the amount of blurring should simply be 0.56 mm
            blurAtResolution = 0.56
            # the resample stepsize should be 0.224
            resampleStepFactor = 0.224 /  highestResolution
            # the registration stepsize should be 0.56
            registrationStepFactor = 0.56 / highestResolution
            # w_translations should be 0.448
            wTranslationsFactor = 0.448 / highestResolution
        else:
            parameterList = self.parameters.split(',')
            blurFactor= float(parameterList[0])
            if(blurFactor != -1):
                blurAtResolution = blurFactor * highestResolution
            resampleStepFactor     = float(parameterList[1])
            registrationStepFactor = float(parameterList[2])
            wTranslationsFactor    = float(parameterList[3])
        
        self.p.addStage(ma.blur(self.target, fwhm=blurAtResolution))
        
        for inputFH in self.inputs:    
            self.p.addStage(ma.blur(inputFH, fwhm=blurAtResolution))
            self.p.addStage((ma.RotationalMinctracc(inputFH,
                                                    self.target,
                                                    blur               = blurAtResolution,
                                                    resample_step      = resampleStepFactor,
                                                    registration_step  = registrationStepFactor,
                                                    w_translations     = wTranslationsFactor,
                                                    rotational_range   = self.rotation_range,
                                                    rotational_interval= self.rotation_interval )))


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
                 options,
                 initial_model     = None,
                 lsq6OutputDir     = None,
                 initial_transform = "estimate",
                 lsq6_protocol     = None):
        # initialize all the defaults in the parent class
        LSQ6Base.__init__(self, inputFiles, targetFile, options,
                          initial_model=initial_model,
                          lsq6OutputDir=lsq6OutputDir)
        self.initial_transform = initial_transform 
        self.lsq6_protocol     = lsq6_protocol
        
        #Read parameters from input file or setup defaults. 
        self.lsq6Params = mp.setLSQ6MinctraccParams(self.fileRes, 
                                            initial_transform=initial_transform, 
                                            reg_protocol=lsq6_protocol)
        self.blurs = self.lsq6Params.blurs
        self.stepSize = self.lsq6Params.stepSize
        self.useGradient = self.lsq6Params.useGradient
        self.simplex = self.lsq6Params.simplex
        self.w_translations = self.lsq6Params.w_translations
        self.generations = self.lsq6Params.generations
        
        # Setup linearparams for minctracc atom
        self.setupLinearParams()
    
    def setupLinearParams(self):
        #All linearparams should be "lsq6" except first, which is based on initial_transform. 
        self.linearParams=[]
        for i in range(self.generations - 1):
            self.linearParams.append("lsq6")
        if self.initial_transform == "estimate":
            self.linearParams.insert(0,"lsq6")
        elif self.initial_transform == "identity":
            self.linearParams.insert(0, "lsq6-identity")
        else:
            print("Error: unknown option used for initial_transform in LSQ6HierarchicalMinctracc: ", self.initial_transform)
            sys.exit()
    
    def createLSQ6Transformation(self):
        """
            Assumption: setLSQ6MinctraccParams has been called prior to running. 
        """
        # create all blurs first
        for i in range(self.generations):
            self.p.addStage(ma.blur(self.target, self.blurs[i], gradient=True))
            for inputfile in self.inputs:
                # create blur for input
                self.p.addStage(ma.blur(inputfile, self.blurs[i], gradient=True))
        
        # now perform the registrations
        for inputfile in self.inputs:
            for i in range(self.generations):
                mt = ma.minctracc(inputfile,
                                  self.target,
                                  blur = self.blurs[i],
                                  simplex = self.simplex[i],
                                  step = self.stepSize[i],
                                  gradient = self.useGradient[i],
                                  w_translations = self.w_translations[i],
                                  linearparam = self.linearParams[i]) 
                self.p.addStage(mt)


        

if __name__ == "__main__":
    application = LSQ6Registration()
    application.start()
