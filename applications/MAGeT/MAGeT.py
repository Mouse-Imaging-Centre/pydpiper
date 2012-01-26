#!/usr/bin/env python

from pydpiper.pipeline import *
from pydpiper.queueing import *
from minctracc import *
import pydpiper.file_handling as fh
from optparse import OptionParser
from os.path import dirname,isdir,abspath
from os import mkdir
import networkx as nx
from multiprocessing import Event

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
    def __init__(self, inputFile, template, outDir=None, inputMask=None,
                 steps=[0.5,0.2],
                 blurs=[0.5,0.2], iterations=[80,20],
                 name="initial"):
        self.p = Pipeline()
        self.input = inputFile
        
        if not outDir:
            outDir = abspath(os.curdir)
            
        inputPipeFH = RegistrationPipeFH(inputFile, abspath(outDir))
        templatePipeFH = RegistrationPipeFH(template.image, template.outputdir)
        
        self.outDir = inputPipeFH.subjDir
        self.inputMask = inputMask
        
        # Note that after testing, these arrays will go away. 
        input_blurs = []
        template_blurs = []

        for b in blurs:
            iblur = blur(inputPipeFH, b)
            tblur = blur(templatePipeFH, b)
            self.p.addStage(iblur)
            self.p.addStage(tblur)
            #MF TODO: Make sure array concatenation syntax is correct
            # Last blurs for both input and template should be the ones just added
            input_blurs += inputPipeFH.getBlur()
            template_blurs += templatePipeFH.getBlur()
            print input_blurs #debug
            print template_blurs #debug

        # lsq12 alignment
        linearparam = "lsq12"
        linearStage = linearminctracc(templatePipeFH, 
                                      inputPipeFH, 
                                      blur=blurs[0], 
                                      linearparam,
                                      source_mask=template.mask,
                                      target_Mask=inputMask)
        self.p.addStage(linearStage)

        # create the nonlinear registrations
        for i in range(len(steps)):
            nlinStage = minctracc(templatePipeFH, 
                                  inputPipeFH, 
                                  blur=blurs[i],
                                  source_mask=template.mask,
                                  target_Mask=inputMask,
                                  iterations=iterations[i],
                                  step=steps[i])
            self.p.addStage(nlinStage)
        
        # resample labels with final registration
        resargs = ["-keep_real_range", "-nearest_neighbour"]
        resampleStage = mincresample(template.labels, 
                                     argarray=resargs, 
                                     likeFile=inputFile)
        self.p.addStage(resampleStage)
        self.output = resampleStage.outputFiles[0] # ok to assume this? 
    
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
                      help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
    parser.add_option("--use-ns", dest="use_ns",
                      action="store_true",
                      help="Use the Pyro NameServer to store object locations")
    parser.add_option("--create-graph", dest="create_graph",
                      action="store_true",
                      help="Create a .dot file with graphical representation of pipeline relationships")
    parser.add_option("--num-executors", dest="num_exec", 
                      type="int", default=0, 
                      help="Launch executors automatically without having to run pipeline_excutor.py independently.")
    parser.add_option("--proc", dest="proc", 
                      type="int", default=4,
                      help="Number of processes per executor. Default is 4. Also sets max value for processor use per executor. Overridden if --num-executors not specified.")
    parser.add_option("--mem", dest="mem", 
                      type="float", default=8,
                      help="Total amount of requested memory. Default is 8G. Overridden if --num-executors not specified.")
    parser.add_option("--queue", dest="queue", 
                      type="string", default=None,
                      help="Use specified queueing system to submit jobs. Default is None.")
    parser.add_option("--restart", dest="restart", 
                      action="store_true",
                      help="Restart pipeline using backup files.")
    
    (options,args) = parser.parse_args()
    
    if options.queue=="pbs":
        ppn = 8
        time = "2:00:00:00"
        roq = runOnQueueingSystem(options, ppn, time, sys.argv)
        roq.createPbsScripts()
    else:
        outputDir = fh.makedirsIgnoreExisting(options.template_library)
        tmplDir = fh.createSubDir(outputDir, "atlas")
        tmpl = Template(options.atlas_image, options.atlas_labels,
                        mask=options.mask, outputdir=tmplDir)
        p = Pipeline()
        p.setBackupFileLocation(outputDir)

        if options.restart:
            p.restart()
        else:
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
            # inputFile to the templates
            for inputFile in args:
                labels = []
                for t in templates:
                    sp = SMATregister(inputFile, t, outDir=outputDir, inputMask=options.mask, name="templates")
                    labels.append(InputFile(sp.output))
                    p.addPipeline(sp.p)
                bname = fh.removeFileExt(inputFile)
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
    
        #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
        # if use_ns is specified, Pyro NameServer must be started. 
        returnEvent = Event()
        pipelineDaemon(p, returnEvent, options, sys.argv[0])
        returnEvent.wait()
        print "templates: " + str(numTemplates)

    
