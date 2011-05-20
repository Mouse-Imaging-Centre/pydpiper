#!/usr/bin/env python

from pipeline import *
from optparse import OptionParser
from os.path import basename,dirname,isdir,abspath
from os import mkdir
import time
import networkx as nx
#rom networkx import DiGraph

Pyro.config.PYRO_MOBILE_CODE=1 

class minctracc(CmdStage):
    def __init__(self, source, target, output, 
                 source_mask=None, 
                 target_mask=None,
                 iterations=40,
                 step=0.5,
                 transform=None,
                 weight=0.8,
                 stiffness=0.98,
                 similarity=0.3,
                 w_translations=0.2):
        CmdStage.__init__(self, None) #don't do any arg processing in superclass
        self.source = source
        self.target = target
        self.output = output
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.iterations = str(iterations)
        self.lattice_diameter = str(step*3)
        self.step = str(step)
        self.transform = transform
        self.weight = str(weight)
        self.stiffness = str(stiffness)
        self.similarity = str(similarity)
        self.w_translations = str(w_translations)

        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        self.colour = "red"

    def setName(self):
        self.name = "minctracc nlin step: " + self.step 
    def addDefaults(self):
        self.cmd = ["minctracc",
                    "-clobber",
                    "-similarity", self.similarity,
                    "-weight", self.weight,
                    "-stiffness", self.stiffness,
                    "-w_translations", self.w_translations,self.w_translations,self.w_translations,
                    "-step", self.step, self.step, self.step,

                    self.source,
                    self.target,
                    self.output]
        
        # assing inputs and outputs
        # assing inputs and outputs
        self.inputFiles = [self.source, self.target]
        if self.source_mask:
            self.inputFiles += [self.source_mask]
            self.cmd += ["-source_mask", self.source_mask]
        if self.target_mask:
            self.inputFiles += [self.target_mask]
            self.cmd += ["-model_mask", self.target_mask]
        if self.transform:
            self.inputFiles += [self.transform]
            self.cmd += ["-transform", self.transform]
        self.outputFiles = [self.output]

    def finalizeCommand(self):
        """add the options for non-linear registration"""
        # build the command itself
        self.cmd += ["-iterations", self.iterations,
                    "-nonlinear", "corrcoeff", "-sub_lattice", "6",
                    "-lattice_diameter", self.lattice_diameter,
                     self.lattice_diameter, self.lattice_diameter]

class lsq12minctracc(minctracc):
    def __init__(self, source, target, output,
                 source_mask=None, target_mask=None):
        minctracc.__init__(self,source,target,output,
                           source_mask=source_mask,
                           target_mask=target_mask)
    def finalizeCommand(self):
        """add the options for a 12 parameter fit"""
        self.cmd += ["-xcorr", "-lsq12"]
    def setName(self):
        self.name = "minctracc lsq12 "

        

class blur(CmdStage):
    def __init__(self, input, output, fwhm):
        # note - output should end with _blur.mnc
        CmdStage.__init__(self, None)
        self.base = output.replace("_blur.mnc", "")
        self.inputFiles = [input]
        self.outputFiles = [output]
        self.cmd = ["mincblur", "-clobber", "-fwhm", str(fwhm),
                    input, self.base]
        self.name = "mincblur " + str(fwhm) + " " + basename(input)
        self.colour="blue"

class Template:
    def __init__(self, image, labels, mask=None, outputdir=None):
        self.image = image
        self.labels = labels
        self.mask = mask
        if outputdir == None:
            self.outputdir = dirname(image)
        else:
            self.outputdir = outputdir

class SMATregister:
    def __init__(self, input, template, outDir="", inputMask=None,
                 steps=[0.5,0.2],
                 blurs=[0.5,0.2], iterations=[80,20],
                 name="initial"):
        self.p = Pipeline()
        self.input = input
        input_base = basename(input).replace(".mnc", "")
        outDir = abspath(outDir)
        input_dir = outDir + "/" + input_base
        if not isdir(input_dir):
            mkdir(input_dir)
        self.outDir = input_dir
        self.inputMask = inputMask
        tbase = basename(template.labels).replace(".mnc","")


        input_base = input_dir + "/" + input_base
        template_base = template.outputdir + "/" + basename(template.image).replace(".mnc", "")

        input_blurs = []
        template_blurs = []

        for b in blurs:
            iblur = input_base + "_fwhm" + str(b) + "_blur.mnc" 
            tblur = template_base + "_fwhm" + str(b) + "_blur.mnc" 
            self.p.addStage(blur(input, iblur, b))
            self.p.addStage(blur(template.image, tblur, b))
            input_blurs += [iblur]
            template_blurs += [tblur]

        # lsq12 alignment
        linxfm = input_base + "_" + tbase + "lsq12.xfm"
        self.p.addStage(lsq12minctracc(template_blurs[0],
                                       input_blurs[0], linxfm,
                                       inputMask, template.mask))

        # create the nonlinear registrations
        xfms = []
        for i in range(len(steps)):
            cxfm = input_base + tbase + "step" + str(i) + ".xfm"
            if i == 0:
                pxfm = linxfm
            else:
                pxfm = xfms[i - 1]
            xfms += [cxfm]
            #print("grunkle: " + str(i) + " " + xfms)
            self.p.addStage(minctracc(template_blurs[i],
                                      input_blurs[i], cxfm,
                                      inputMask, template.mask,
                                      iterations=iterations[i],
                                      step=steps[i],
                                      transform=pxfm))
        # resample labels with final registration
        
        self.output = input_base + "resampled_" + tbase + ".mnc"
        self.p.addStage(CmdStage(["mincresample", "-2", "-clobber", 
                                  "-like", input,
                                  "-keep_real_range", "-nearest_neighbour",
                                  "-transform", InputFile(cxfm),
                                  InputFile(template.labels), 
                                  OutputFile(self.output)]))

    def getTemplate(self):
        return(Template(self.input, self.output, self.inputMask,
                        self.outDir))

def test(npipes):
    starttime = time.time()
    atlas = Template("atlas-image.mnc", "atlas-labels.mnc")
    p = Pipeline()

    for i in range(npipes):
        file = "input_" + str(i) + ".mnc"
        sp = SMATregister(file, atlas)
        p.addPipeline(sp.p)
    p.initialize()
    endtime = time.time()
    print("test time: " + str(endtime-starttime))
    return(p)


if __name__ == "__main__":
    usage = "%prog [options] input1.mnc ... inputn.mnc"
    description = "description needed"

    parser = OptionParser(usage=usage, description=description)

    parser.add_option("--atlas-labels", "-a", dest="atlas_labels",
                      type="string", 
                      help="MINC volume containing labelled structures")
    parser.add_option("--atlas-image", "-i", dest="atlas_image",
                      type="string",
                      help="MINC volume of image corresponding to labels")
    parser.add_option("--template-lib", dest="template_library",
                      type="string", default=".",
                      help="Directory where every image is an atlas pair")
    parser.add_option("--mask", dest="mask",
                      type="string",
                      help="Mask to use for all images")
    parser.add_option("--create-template-lib", dest="create_templates",
                      action="store_true", 
                      help="Create the template library from scratch")
    parser.add_option("--use-existing-template-lib", dest="create_templates",
                      action="store_false",
                      help="Use existing template library, location defined by the --template-lib option")
    parser.add_option("--max-templates", dest="max_templates",
                      default=25, type="int",
                      help="Maximum number of templates to generate")

    (options,args) = parser.parse_args()

    outputDir = abspath(options.template_library)
    tmplDir = outputDir + "/atlas"
    if not isdir(tmplDir):
        mkdir(tmplDir)
    tmpl = Template(options.atlas_image, options.atlas_labels,
                    mask=options.mask, outputdir=tmplDir)
    p = Pipeline()

    # create the initial templates - either total number of files
    # or the maximum number of templates, whichever is lesser
    templates = []
    numTemplates = len(args)
    if numTemplates > options.max_templates:
        numTemplates = options.max_templates
    
    for nfile in range(numTemplates):
        sp = SMATregister(args[nfile], tmpl, outDir=outputDir, inputMask=options.mask)
        templates.append(sp.getTemplate())
        p.addPipeline(sp.p)

    # once the initial templates have been created, go and register each
    # file to the templates
    for file in args:
        labels = []
        for t in templates:
            sp = SMATregister(file, t, outDir=outputDir,
                              inputMask=options.mask, name="templates")
            labels.append(InputFile(sp.output))
            p.addPipeline(sp.p)
        bname = basename(file).replace(".mnc", "")
        of = OutputFile(outputDir + "/" + bname + "_votedlabels.mnc")
        cmd = ["voxel_vote.py"] + labels + [of]
        p.addStage(CmdStage(cmd))

    p.initialize()
    p.printStages()
    nx.write_dot(p.G, "labeled-tree.dot")
    
    pipelineNoNSDaemon(p)
    print "templates: " + str(numTemplates)
    #print("CHORDAL: " + nx.algorithms.chordal.chordal_alg.is_chordal(p.G))

    
