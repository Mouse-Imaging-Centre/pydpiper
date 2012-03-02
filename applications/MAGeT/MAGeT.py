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

class SMATregister:
    def __init__(self, inputPipeFH, 
                 templatePipeFH,
                 steps=[0.5,0.2],
                 blurs=[0.5,0.2], 
                 gradients=[True, False],
                 iterations=[80,20],
                 name="initial"):
        self.p = Pipeline()
        
        for b in blurs:
            iblur = blur(inputPipeFH, b, gradient=True)
            tblur = blur(templatePipeFH, b, gradient=True)
            self.p.addStage(iblur)
            self.p.addStage(tblur)
            
        # lsq12 alignment
        linearparam = "lsq12"
        linearStage = linearminctracc(templatePipeFH, 
                                      inputPipeFH, 
                                      blur=blurs[0], 
                                      linearparam=linearparam)
        self.p.addStage(linearStage)

        # create the nonlinear registrations
        for i in range(len(steps)):
            nlinStage = minctracc(templatePipeFH, 
                                  inputPipeFH, 
                                  blur=blurs[i],
                                  gradient=gradients[i],
                                  iterations=iterations[i],
                                  step=steps[i])
            self.p.addStage(nlinStage)
        
        # resample labels with final registration
        resampleStage = mincresampleLabels(templatePipeFH, likeFile=inputPipeFH)
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
        #MF TODO: move this into else:
        tmplDir = fh.createSubDir(outputDir, "atlas")
        
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
                
            # Create fileHandling classes for initial template
            tmplPipeFH = RegistrationPipeFH(abspath(options.atlas_image), tmplDir)
            tmplPipeFH.setInputLabels(abspath(options.atlas_labels))
            tmplPipeFH.setMask(abspath(options.mask))    
    
            for nfile in range(numTemplates):
                inputPipeFH = RegistrationPipeFH(abspath(args[nfile]), outputDir)
                inputPipeFH.setMask(abspath(options.mask))
                sp = SMATregister(inputPipeFH, tmplPipeFH)
                templates.append(inputPipeFH)
                p.addPipeline(sp.p)

            # once the initial templates have been created, go and register each
            # inputFile to the templates
            for inputFH in templates:
                labels = []
                for tmplFH in templates:
                        if tmplFH.getLastBasevol() != inputFH.getLastBasevol():
                            sp = SMATregister(inputFH, tmplFH, name="templates")
                            p.addPipeline(sp.p)
                voxel = voxelVote(inputFH)
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

    
