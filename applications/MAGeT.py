#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
from atoms_and_modules.registration_file_handling import RegistrationPipeFH
from atoms_and_modules.registration_functions import initializeInputFiles, addGenRegOptionGroup
from atoms_and_modules.MAGeT_modules import MAGeTMask, MAGeTRegister, voxelVote, addMAGeTOptionGroup
from atoms_and_modules.LSQ12 import addLSQ12OptionGroup
from atoms_and_modules.NLIN import addNlinRegOptionGroup
from os.path import abspath, join
import logging
import glob
import fnmatch
import sys

logger = logging.getLogger(__name__)

class MAGeTApplication(AbstractApplication):
    def setup_options(self):
        addGenRegOptionGroup(self.parser)
        addMAGeTOptionGroup(self.parser)
        addLSQ12OptionGroup(self.parser)
        addNlinRegOptionGroup(self.parser)
        self.parser.set_usage("%prog [options] input files") 

    def setup_appName(self):
        appName = "MAGeT"
        return appName

    def run(self):
        
        if self.options.reg_method != "minctracc" and self.options.reg_method != "mincANTS":
            logger.error("Incorrect registration method specified: " + self.options.reg_method)
            sys.exit()
        
        atlasDir = fh.createSubDir(self.outputDir, "input_atlases")
        
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
        for inFile in glob.glob(join(self.options.atlas_lib, "*.mnc")):
            if fnmatch.fnmatch(inFile, "*labels.mnc"):
                labels.append(abspath(inFile))
            elif fnmatch.fnmatch(inFile, "*mask.mnc"):
                masks.append(abspath(inFile))
            else:
                average.append(abspath(inFile))
        # check to make sure len(average)==len(labels)
        if not len(average) == len(labels):
            logger.error("Number of input atlas labels does not match averages.")
            logger.error("Check " + str(self.options.atlas_lib) + " and try again.")
            sys.exit() 
        elif not len(average) == len(masks):
            logger.error("Number of input atlas masks does not match averages.")
            logger.error("Check " + str(self.options.atlas_lib) + " and try again.")
            sys.exit()
        else:
        # match labels with averages
            numAtlases = len(labels)
            for iLabel in labels: 
                atlasStart = iLabel.split("_labels.mnc")
                for iAvg in average:
                    avgStart = iAvg.split("_average.mnc")
                    if fnmatch.fnmatch(atlasStart[0], avgStart[0]):
                        for iMask in masks:
                            maskStart = iMask.split("_mask.mnc")
                            if fnmatch.fnmatch(atlasStart[0], maskStart[0]):
                                atlasPipeFH = RegistrationPipeFH(abspath(iAvg), 
                                                                 mask=abspath(iMask),
                                                                 basedir=atlasDir)
                                break
                        break
                atlasPipeFH.addLabels(abspath(iLabel), inputLabel=True) 
                atlases.append(atlasPipeFH)
        
        #MF TODO: add some checking to make sure that atlas/labels/naming all worked correctly
        # eg if we have A4_mask.mnc "matching" with A3_labels, we wont get right thing.
        
        """ Create fileHandling classes for images. If a directory of masks is specified, 
            they will be assigned to the appropriate input file handlers."""
        inputs = initializeInputFiles(self.args, self.outputDir, maskDir=self.options.mask_dir)
        
        templates = []
        numTemplates = len(self.args)
        
        """ If --mask is specified and we are masking brains, do it here."""
        if self.options.mask or self.options.mask_only:
            mp = MAGeTMask(atlases, 
                           inputs, 
                           numAtlases, 
                           self.options.mask_method,
                           lsq12_protocol=self.options.lsq12_protocol,
                           nlin_protocol=self.options.nlin_protocol)
            self.pipeline.addPipeline(mp)
        
        if not self.options.mask_only:
            if numTemplates > self.options.max_templates:
                numTemplates = self.options.max_templates
            # Register each atlas to each input image up to numTemplates
            for nfile in range(numTemplates):
                for afile in range(numAtlases):
                    sp = MAGeTRegister(inputs[nfile], 
                                       atlases[afile],
                                       self.options.reg_method,
                                       name="initial",
                                       createMask=False,
                                       lsq12_protocol=self.options.lsq12_protocol,
                                       nlin_protocol=self.options.nlin_protocol)
                    self.pipeline.addPipeline(sp)
                # each template needs to be added only once, but will have multiple 
                # input labels
                templates.append(inputs[nfile])
            
            # once the initial templates have been created, go and register each
            # inputFile to the templates. If --pairwise=False, do voxel voting on 
            # input-atlas registrations only
            if self.options.pairwise:
                for inputFH in inputs:
                    for tmplFH in templates:
                        if tmplFH.getLastBasevol() != inputFH.getLastBasevol():
                            sp = MAGeTRegister(inputFH, 
                                               tmplFH, 
                                               self.options.reg_method,
                                               name="templates", 
                                               createMask=False,
                                               lsq12_protocol=self.options.lsq12_protocol,
                                               nlin_protocol=self.options.nlin_protocol)
                            self.pipeline.addPipeline(sp)
                    voxel = voxelVote(inputFH, self.options.pairwise, False)
                    self.pipeline.addStage(voxel)
            else:
                # only do voxel voting in this case if there was more than one input atlas
                if numAtlases > 1:
                    for inputFH in inputs:
                        voxel = voxelVote(inputFH, self.options.pairwise, False)
                        self.pipeline.addStage(voxel)   
        
            logger.info("Number of input atlas/label pairs: " + str(numAtlases))   
            logger.info("Number of templates: " + str(numTemplates))     
    
      
if __name__ == "__main__":
    
    application = MAGeTApplication()
    application.start()
               
