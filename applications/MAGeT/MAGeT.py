#!/usr/bin/env python

from pydpiper.pipeline import *
from pydpiper.queueing import *
from pydpiper.application import AbstractApplication
from minctracc import *
import pydpiper.file_handling as fh
from os.path import abspath
from multiprocessing import Event
import logging

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
        
        for b in blurs:
            #MF TODO: -1 case is also handled in blur. Need here for addStage.
            #Fix this redundancy and/or better design?
            if b != -1:
                iblur = blur(inputPipeFH, b, gradient=True)
                tblur = blur(templatePipeFH, b, gradient=True)
                self.p.addStage(iblur)
                self.p.addStage(tblur)
            
        # Two lsq12 stages: one using 0.25 blur, one using 0.25 gradient
        for g in [False, True]:    
            linearStage = minctracc(templatePipeFH, 
                                      inputPipeFH, 
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
            nlinStage = minctracc(templatePipeFH, 
                                  inputPipeFH, 
                                  blur=blurs[i],
                                  gradient=gradients[i],
                                  iterations=iterations[i],
                                  step=steps[i],
                                  similarity=0.8,
                                  w_translations=w_translations,
                                  simplex=simplexes[i])
            self.p.addStage(nlinStage)
        
        # resample labels with final registration
        if templatePipeFH.getInputLabels():
            resampleStage = mincresampleLabels(templatePipeFH, likeFile=inputPipeFH)
            self.p.addStage(resampleStage)
        # resample files
        resampleStage = mincresample(templatePipeFH, likeFile=inputPipeFH)
        self.p.addStage(resampleStage)


def voxelVote(inputFH):
    labels = inputFH.returnLabels()
    out = fh.createBaseName(inputFH.labelsDir, inputFH.basename) 
    out += "_votedlabels.mnc"
    logFile = inputFH.logFromFile(out)
    cmd = ["voxel_vote.py"] + [InputFile(l) for l in labels] + [OutputFile(out)]
    voxel = CmdStage(cmd)
    voxel.setLogFile(LogFile(logFile))
    return(voxel)

class MAGeTApplication(AbstractApplication):
    def setup_options(self):
        self.parser.add_option("--atlas-labels", "-a", dest="atlas_labels",
                      type="string", 
                      help="MINC volume containing labelled structures")
        self.parser.add_option("--atlas-image", "-i", dest="atlas_image",
                      type="string",
                      help="MINC volume of image corresponding to labels")
        self.parser.add_option("--template-lib", dest="template_library",
                      type="string", default=".",
                      help="Directory where every image is an atlas pair")
        self.parser.add_option("--mask", dest="mask",
                      type="string",
                      help="Mask to use for all images")
        self.parser.add_option("--create-template-lib", dest="create_templates",
                      action="store_true", 
                      help="Create the template library from scratch")
        self.parser.add_option("--use-existing-template-lib", dest="create_templates",
                      action="store_false",
                      help="Use existing template library, location defined by the --template-lib option")
        self.parser.add_option("--max-templates", dest="max_templates",
                      default=25, type="int",
                      help="Maximum number of templates to generate")
        
        self.parser.set_usage("%prog [options] input files") 

    def run(self):
        options = self.options
        args = self.args
        
        outputDir = fh.makedirsIgnoreExisting(options.template_library)
        tmplDir = fh.createSubDir(outputDir, "atlas")
        
        # create the initial templates - either total number of files
        # or the maximum number of templates, whichever is lesser
        templates = []
        inputs = []
        numTemplates = len(args)
        if numTemplates > options.max_templates:
            numTemplates = options.max_templates
                
        # Create fileHandling classes for initial template
        tmplPipeFH = RegistrationPipeFH(abspath(options.atlas_image), tmplDir)
        tmplPipeFH.setInputLabels(abspath(options.atlas_labels))
        if options.mask:
            tmplPipeFH.setMask(abspath(options.mask))    
    
        # Create fileHandling classes for images
        for iFile in range(len(args)):
            inputPipeFH = RegistrationPipeFH(abspath(args[iFile]), outputDir)
            if options.mask:
                inputPipeFH.setMask(abspath(options.mask))
            inputs.append(inputPipeFH)
    
        for nfile in range(numTemplates):
            sp = SMATregister(inputs[nfile], tmplPipeFH)
            templates.append(inputs[nfile])
            self.pipeline.addPipeline(sp.p)

        # once the initial templates have been created, go and register each
        # inputFile to the templates
        for inputFH in inputs:
            for tmplFH in templates:
                sp = SMATregister(inputFH, tmplFH, name="templates")
                self.pipeline.addPipeline(sp.p)
            voxel = voxelVote(inputFH)
            self.pipeline.addStage(voxel)
            
        print "templates: " + str(numTemplates)    
    
      
if __name__ == "__main__":
    FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
    now = datetime.now()  
    FILENAME = "MAGeT.py-" + now.strftime("%Y%m%d-%H%M%S") + ".log"
    logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)
    
    application = MAGeTApplication()
    application.start()
               
