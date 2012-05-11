#!/usr/bin/env python

from pydpiper.pipeline import *
from pydpiper.queueing import *
from pydpiper.application import AbstractApplication
from minctracc import *
import pydpiper.file_handling as fh
from os.path import abspath, join
from multiprocessing import Event
import logging
import glob
import fnmatch
import re

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

# JPL: why is this class in MAGeT - this, except for the final label resampling, is generic hierarchical minctracc, right?
class SMATregister:
    def __init__(self, inputPipeFH, 
                 templatePipeFH,
                 steps=[1,0.5,0.5,0.2,0.2,0.1],
                 blurs=[0.25,0.25,0.25,0.25,0.25, -1], 
                 gradients=[False, False, True, False, True, False],
                 iterations=[60,60,60,10,10,4],
                 simplexes=[3,3,3,1.5,1.5,1],
                 w_translations=0.2,
                 linearparams = {'type' : "lsq12", 'simplex' : 1, 'step' : 1},
                 name="initial"):
        self.p = Pipeline()
        self.name = name
        
        for b in blurs:
            #MF TODO: -1 case is also handled in blur. Need here for addStage.
            #Fix this redundancy and/or better design?
            if b != -1:
                tblur = blur(templatePipeFH, b, gradient=True)
                iblur = blur(inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
            
        # Two lsq12 stages: one using 0.25 blur, one using 0.25 gradient
        for g in [False, True]:    
            linearStage = minctracc(inputPipeFH, 
                                      templatePipeFH, 
                                      blur=blurs[0], 
                                      gradient=g,                                     
                                      linearparam=linearparams["type"],
                                      step=linearparams["step"],
                                      simplex=linearparams["simplex"],
                                      w_translations=w_translations,
                                      similarity=0.5)
            self.p.addStage(linearStage)

        # create the nonlinear registrations
        for i in range(len(steps)):
            nlinStage = minctracc(inputPipeFH, 
                                  templatePipeFH, 
                                  blur=blurs[i],
                                  gradient=gradients[i],
                                  iterations=iterations[i],
                                  step=steps[i],
                                  similarity=0.8,
                                  w_translations=w_translations,
                                  simplex=simplexes[i])
            self.p.addStage(nlinStage)
        
        # Resample all inputLabels 
        inputLabelArray = templatePipeFH.returnLabels(True)
        if len(inputLabelArray) > 0:
            """ for the initial registration, resulting labels should be added
                to inputLabels array for subsequent pairwise registration
                otherwise labels should be added to labels array for voting """
            if self.name == "initial":
                addOutputToInputLabels = True
            else:
                addOutputToInputLabels = False
            for i in range(len(inputLabelArray)):
                resampleStage = mincresampleLabels(templatePipeFH, 
                                                   likeFile=inputPipeFH,
                                                   labelIndex=i,
                                                   setInputLabels=addOutputToInputLabels)
                self.p.addStage(resampleStage)
        # resample files
        resampleStage = mincresample(templatePipeFH, likeFile=inputPipeFH)
        self.p.addStage(resampleStage)

def voxelVote(inputFH, pairwise):
    # if we do pairwise crossing, use output labels for voting (Default)
    # otherwise, return inputLabels from initial atlas-input crossing
    useInputLabels = False
    if not pairwise:
        useInputLabels = True
    labels = inputFH.returnLabels(useInputLabels)
    out = fh.createBaseName(inputFH.labelsDir, inputFH.basename) 
    out += "_votedlabels.mnc"
    logFile = inputFH.logFromFile(out)
    cmd = ["voxel_vote.py"] + [InputFile(l) for l in labels] + [OutputFile(out)]
    voxel = CmdStage(cmd)
    voxel.setLogFile(LogFile(logFile))
    return(voxel)

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
                      type="string",
                      help="Mask to use for all images")
        self.parser.add_option("--max-templates", dest="max_templates",
                      default=25, type="int",
                      help="Maximum number of templates to generate")
        
        self.parser.set_usage("%prog [options] input files") 

    def run(self):
        options = self.options
        args = self.args
        reconstruct = ""
        for i in range(len(sys.argv)):
            reconstruct += sys.argv[i] + " "
        logger.info("Command is: " + reconstruct)
        
        outputDir = fh.makedirsIgnoreExisting(options.output_directory)
        atlasDir = fh.createSubDir(outputDir, "input_atlases")
        
        """Read in atlases from directory specified in --atlas-library and 
            create fileHandling classes. Assumes atlas/label pairs have one 
            of the following naming schemes:
                name_average.mnc/name_labels.mnc
                name.mnc/name_labels.mnc"""
        average = [] #array of input atlas averages
        labels = [] #array of input atlas labels, one for each average
        atlases = [] #array of RegistrationPipeFH classes for each atlas/label pair
        numAtlases = 0
        for inFile in glob.glob(os.path.join(options.atlas_lib, "*.mnc")):
            if not fnmatch.fnmatch(inFile, "*labels.mnc"):
                average.append(abspath(inFile))
            else:
                labels.append(abspath(inFile))
        # check to make sure len(average)==len(labels)
        if not len(average) == len(labels):
            logger.error("Number of input atlas labels does not match averages.")
            logger.error("Check " + str(options.atlas_lib) + " and try again.")
            sys.exit()
        else:
        # match labels with averages
            numAtlases = len(labels)
            for iLabel in labels: 
                atlasStart = iLabel.split("_labels.mnc")
                for iAvg in average:
                    if re.search(atlasStart[0], iAvg):
                        atlasPipeFH = RegistrationPipeFH(abspath(iAvg), atlasDir)
                        break
                atlasPipeFH.addLabels(abspath(iLabel), inputLabel=True)
                if options.mask:
                    atlasPipeFH.setMask(abspath(options.mask))  
                atlases.append(atlasPipeFH)
        
        # Create fileHandling classes for images
        inputs = []
        for iFile in range(len(args)):
            inputPipeFH = RegistrationPipeFH(abspath(args[iFile]), outputDir)
            if options.mask:
                inputPipeFH.setMask(abspath(options.mask))
            inputs.append(inputPipeFH)
        
        templates = []
        numTemplates = len(args)
        if numTemplates > options.max_templates:
            numTemplates = options.max_templates
        # Register each atlas to each input image up to numTemplates
        for nfile in range(numTemplates):
            for afile in range(numAtlases):
                sp = SMATregister(inputs[nfile], atlases[afile])
                self.pipeline.addPipeline(sp.p)
            # each template needs to be added only once, but will have multiple 
            # input labels
            templates.append(inputs[nfile])
            
        # once the initial templates have been created, go and register each
        # inputFile to the templates. If --pairwise=False, do voxel voting on 
        # input-atlas registrations only
        if options.pairwise:
            for inputFH in inputs:
                for tmplFH in templates:
                    sp = SMATregister(inputFH, tmplFH, name="templates")
                    self.pipeline.addPipeline(sp.p)
                voxel = voxelVote(inputFH, options.pairwise)
                self.pipeline.addStage(voxel)
        else:
            # only do voxel voting in this case if there was more than one input atlas
            if numAtlases > 1:
                for inputFH in inputs:
                    voxel = voxelVote(inputFH, options.pairwise)
                    self.pipeline.addStage(voxel)   
        
        logger.info("Number of input atlas/label pairs: " + str(numAtlases))   
        logger.info("Number of templates: " + str(numTemplates))     
    
      
if __name__ == "__main__":
    FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
    now = datetime.now()  
    FILENAME = "MAGeT.py-" + now.strftime("%Y%m%d-%H%M%S") + ".log"
    logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)
    
    application = MAGeTApplication()
    application.start()
               
