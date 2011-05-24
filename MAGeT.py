#!/usr/bin/env python

from pipeline import *
from minctracc import *
from optparse import OptionParser
from os.path import basename,dirname,isdir,abspath
from os import mkdir
import time
import networkx as nx
#rom networkx import DiGraph

Pyro.config.PYRO_MOBILE_CODE=1 

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
        input_base_fname = basename(input).replace(".mnc", "")
        outDir = abspath(outDir)
        input_dir = outDir + "/" + input_base_fname
        if not isdir(input_dir):
            mkdir(input_dir)
        input_log_dir = input_dir + "/log"
        if not isdir(input_log_dir):
            mkdir(input_log_dir)
        templ_log_dir = template.outputdir + "/log"
        if not isdir(templ_log_dir):
            mkdir(templ_log_dir)
        self.outDir = input_dir
        self.inputMask = inputMask
        tbase = basename(template.labels).replace(".mnc","")

        input_base = input_dir + "/" + input_base_fname
        template_base = template.outputdir + "/" + basename(template.image).replace(".mnc", "")
        input_log_base = input_log_dir + "/" + input_base_fname
        templ_log_base = templ_log_dir + "/" + basename(template.image).replace(".mnc", "")

        input_blurs = []
        template_blurs = []

        for b in blurs:
            iblur = input_base + "_fwhm" + str(b) + "_blur.mnc" 
            tblur = template_base + "_fwhm" + str(b) + "_blur.mnc" 
            ilog = input_log_base + "_fwhm" + str(b) + "_blur.log"
            tlog = templ_log_base + "_fwhm" + str(b) + "_blur.log"
            self.p.addStage(blur(input, iblur, ilog, b))
            self.p.addStage(blur(template.image, tblur, tlog, b))
            input_blurs += [iblur]
            template_blurs += [tblur]

        # lsq12 alignment
        linxfm = input_base + "_" + tbase + "lsq12.xfm"
        logfile = input_log_base + "_" + tbase + "lsq12.log"
        linearparam = "lsq12"
        self.p.addStage(linearminctracc(template_blurs[0],
                                       input_blurs[0], linxfm,
				       logfile, linearparam,
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
            logfile = input_log_base + tbase + "step" + str(i) + ".log"
            linearparam = "nlin"
            self.p.addStage(minctracc(template_blurs[i],
                                      input_blurs[i], cxfm,
                                      logfile, linearparam,
                                      inputMask, template.mask,
                                      iterations=iterations[i],
                                      step=steps[i],
                                      transform=pxfm))
        # resample labels with final registration
        
        self.output = input_base + "resampled_" + tbase + ".mnc"
        logfile = input_log_base + "resampled_" + tbase + ".log"
	
	resargs = ["-keep_real_range", "-nearest_neighbour"]
	self.p.addStage(mincresample(template.labels, self.output, logfile, resargs, input, cxfm))
	
    def getTemplate(self):
        return(Template(self.input, self.output, self.inputMask,
                        self.outDir))

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
    parser.add_option("--uri-file", dest="urifile",
		      type="string", default=None,
                      help="Location for uri file if NameServer is not used.")
    parser.add_option("--use-ns", dest="use_ns",
		      action="store_true",
		      help="Use the Pyro NameServer to store object locations")
    parser.add_option("--create-graph", dest="create_graph",
		      action="store_true",
		      help="Create a .dot file with graphical representation of pipeline relationships")
    

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
        lf = LogFile(outputDir + "/" + bname + "_votedlabels.log")
        cmd = ["voxel_vote.py"] + labels + [of]
        voxel = CmdStage(cmd)
        voxel.setLogFile(lf)
        p.addStage(voxel)

    p.initialize()
    p.printStages()
    
    if options.create_graph:
    	nx.write_dot(p.G, "labeled-tree.dot")
    
    #If Pyro NameServer option was specified, use it.
    
    if options.use_ns:
	pipelineDaemon(p)
    else:
        pipelineNoNSDaemon(p, options.urifile)
    
    print "templates: " + str(numTemplates)

    
