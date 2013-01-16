#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
from pydpiper_apps.minc_tools.registration_file_handling import RegistrationPipeFH
from pydpiper_apps.minc_tools.registration_functions import initializeInputFiles
from pydpiper_apps.MAGeT.MAGeT_modules import MAGeTMask, MAGeTRegister, voxelVote
import Pyro
from os.path import abspath, join
import logging
import glob
import fnmatch
import re
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

class MAGeTApplication(AbstractApplication):
    def setup_options(self):
        self.parser.add_option("--atlas-library", dest="atlas_lib",
                      type="string", default="atlas_label_pairs",
                      help="Directory of existing atlas/label pairs")
        self.parser.add_option("--no-pairwise", dest="pairwise",
                      action="store_false", default=True,
                      help="""Pairwise crossing of templates. Default is true. 
                          If specified, only register inputs to atlases in library""")
        self.parser.add_option("--output-dir", dest="output_directory",
                      type="string", default=".",
                      help="Directory where output data will be saved.")
        self.parser.add_option("--mask", dest="mask",
                      action="store_true", default=False,
                      help="Create a mask for all images prior to handling labels")
        self.parser.add_option("--mask-only", dest="mask_only",
                      action="store_true", default=False,
                      help="Create a mask for all images only, do not run full algorithm")
        self.parser.add_option("--max-templates", dest="max_templates",
                      default=25, type="int",
                      help="Maximum number of templates to generate")
        self.parser.add_option("--registration-method", dest="reg_method",
                      default="minctracc", type="string",
                      help="Specify whether to use minctracc or mincANTS")
        self.parser.add_option("--masking-method", dest="mask_method",
                      default="minctracc", type="string",
                      help="Specify whether to use minctracc or mincANTS for masking")
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_backupDir(self):
        """Output directory set here as well. backups subdirectory automatically
        placed here so we don't need to set via the command line"""
        backup_dir = fh.makedirsIgnoreExisting(self.options.output_directory)    
        self.pipeline.setBackupFileLocation(backup_dir)

    def setup_appName(self):
        appName = "MAGeT"
        return appName

    def run(self):
        options = self.options
        args = self.args
        self.reconstructCommand()
        
        if options.reg_method != "minctracc" and options.reg_method != "mincANTS":
            logger.error("Incorrect registration method specified: " + options.reg_method)
            sys.exit()
        
        outputDir = fh.makedirsIgnoreExisting(options.output_directory)
        atlasDir = fh.createSubDir(outputDir, "input_atlases")
        
        """Read in atlases from directory specified in --atlas-library and 
            create fileHandling classes. Assumes atlas/label/mask groups have one 
            of the following naming schemes:
                name_average.mnc/name_labels.mnc/name_mask.mnc
                name.mnc/name_labels.mnc/name_mask.mnc
            Note that all three files are required even if masking is not used."""
        average = [] #array of input atlas averages
        labels = [] #array of input atlas labels, one for each average
        masks = [] # array of masks, one for each average
        atlases = [] #array of RegistrationPipeFH classes for each atlas/label pair
        numAtlases = 0
        for inFile in glob.glob(join(options.atlas_lib, "*.mnc")):
            if fnmatch.fnmatch(inFile, "*labels.mnc"):
                labels.append(abspath(inFile))
            elif fnmatch.fnmatch(inFile, "*mask.mnc"):
                masks.append(abspath(inFile))
            else:
                average.append(abspath(inFile))
        # check to make sure len(average)==len(labels)
        if not len(average) == len(labels):
            logger.error("Number of input atlas labels does not match averages.")
            logger.error("Check " + str(options.atlas_lib) + " and try again.")
            sys.exit() 
        elif not len(average) == len(masks):
            logger.error("Number of input atlas masks does not match averages.")
            logger.error("Check " + str(options.atlas_lib) + " and try again.")
            sys.exit()
        else:
        # match labels with averages
            numAtlases = len(labels)
            for iLabel in labels: 
                atlasStart = iLabel.split("_labels.mnc")
                for iAvg in average:
                    if re.search(atlasStart[0], iAvg):
                        for iMask in masks:
                            if re.search(atlasStart[0], iMask):
                                atlasPipeFH = RegistrationPipeFH(abspath(iAvg), 
                                                                 mask=abspath(iMask),
                                                                 basedir=atlasDir)
                                break
                        break
                atlasPipeFH.addLabels(abspath(iLabel), inputLabel=True) 
                atlases.append(atlasPipeFH)
        
        #MF TODO: add some checking to make sure that atlas/labels/naming all worked correctly
        # eg if we have A4_mask.mnc "matching" with A3_labels, we wont get right thing.
        
        # Create fileHandling classes for images
        inputs = initializeInputFiles(args, outputDir)
        
        templates = []
        numTemplates = len(args)
        
        """ If --mask is specified and we are masking brains, do it here."""
        if options.mask or options.mask_only:
            mp = MAGeTMask(atlases, inputs, numAtlases, options.mask_method)
            self.pipeline.addPipeline(mp)
        
        if not options.mask_only:
            if numTemplates > options.max_templates:
                numTemplates = options.max_templates
            # Register each atlas to each input image up to numTemplates
            for nfile in range(numTemplates):
                for afile in range(numAtlases):
                    sp = MAGeTRegister(inputs[nfile], 
                                       atlases[afile],
                                       options.reg_method,
                                       name="initial",
                                       createMask=False)
                    self.pipeline.addPipeline(sp)
                # each template needs to be added only once, but will have multiple 
                # input labels
                templates.append(inputs[nfile])
            
            # once the initial templates have been created, go and register each
            # inputFile to the templates. If --pairwise=False, do voxel voting on 
            # input-atlas registrations only
            if options.pairwise:
                for inputFH in inputs:
                    for tmplFH in templates:
                        if tmplFH.getLastBasevol() != inputFH.getLastBasevol():
                            sp = MAGeTRegister(inputFH, 
                                               tmplFH, 
                                               options.reg_method,
                                               name="templates", 
                                               createMask=False)
                            self.pipeline.addPipeline(sp)
                    voxel = voxelVote(inputFH, options.pairwise, False)
                    self.pipeline.addStage(voxel)
            else:
                # only do voxel voting in this case if there was more than one input atlas
                if numAtlases > 1:
                    for inputFH in inputs:
                        voxel = voxelVote(inputFH, options.pairwise, False)
                        self.pipeline.addStage(voxel)   
        
            logger.info("Number of input atlas/label pairs: " + str(numAtlases))   
            logger.info("Number of templates: " + str(numTemplates))     
    
      
if __name__ == "__main__":
    
    application = MAGeTApplication()
    application.start()
               
