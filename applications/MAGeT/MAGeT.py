#!/usr/bin/env python

from pipeline.pipeline import *
from minctracc import *
from optparse import OptionParser
from os.path import basename,dirname,isdir,abspath
from os import mkdir
import time
import networkx as nx

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
        
        fh = mincFileHandling()
        
        input_base_fname = fh.removeFileExt(input)
        input_dir, input_base = fh.createSubDirSubBase(abspath(outDir), input_base_fname, input_base_fname)
        input_log_dir, input_log_base = fh.createLogDirLogBase(input_dir, input_base_fname)
        
        templ_base_fname = fh.removeFileExt(template.image)
        template_base = fh.createBaseName(template.outputdir, templ_base_fname)
        templ_log_dir, templ_log_base = fh.createLogDirLogBase(template.outputdir, templ_base_fname)
        
        self.outDir = input_dir
        self.inputMask = inputMask
        tbase = fh.removeFileExt(template.labels)

        input_blurs = []
        template_blurs = []

        for b in blurs:
            iblur, ilog = fh.createBlurOutputAndLogFiles(input_base, input_log_base, str(b))
            tblur, tlog = fh.createBlurOutputAndLogFiles(template_base, templ_log_base, str(b))
            self.p.addStage(blur(input, iblur, ilog, b))
            self.p.addStage(blur(template.image, tblur, tlog, b))
            input_blurs += [iblur]
            template_blurs += [tblur]

        # lsq12 alignment
        linearparam = "lsq12"
        linxfm, logfile = fh.createXfmAndLogFiles(input_base, input_log_base, [tbase, linearparam])
        self.p.addStage(linearminctracc(template_blurs[0],
                                       input_blurs[0], linxfm,
                                       logfile, linearparam,
                                       inputMask, template.mask))

        # create the nonlinear registrations
        xfms = []
        linearparam = "nlin"
        for i in range(len(steps)):
            cxfm, logfile = fh.createXfmAndLogFiles(input_base, input_log_base, [tbase, "step", str(i)])
            if i == 0:
                pxfm = linxfm
            else:
                pxfm = xfms[i - 1]
            xfms += [cxfm]
            self.p.addStage(minctracc(template_blurs[i],
                                      input_blurs[i], cxfm,
                                      logfile, linearparam,
                                      inputMask, template.mask,
                                      iterations=iterations[i],
                                      step=steps[i],
                                      transform=pxfm))
        # resample labels with final registration
        
        self.output, logfile = fh.createResampledAndLogFiles(input_base, input_log_base, [tbase])
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

    fh = FileHandling()
    outputDir = abspath(options.template_library)
    tmplDir = fh.createSubDir(outputDir, "atlas")
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
        bname = fh.removeFileExt(file)
        base = fh.createBaseName(outputDir, bname + "_votedlabels")
        out, log = fh.createOutputAndLogFiles(base, base, ".mnc")
        cmd = ["voxel_vote.py"] + labels + [OutputFile(out)]
        voxel = CmdStage(cmd)
        voxel.setLogFile(LogFile(log))
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

    
