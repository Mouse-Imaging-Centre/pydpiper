#!/usr/bin/env python

from pydpiper.pipeline import *
from pydpiper.queueing import *
from minctracc import *
from optparse import OptionParser
from os.path import dirname,isdir,abspath
from os import mkdir
import networkx as nx
from multiprocessing import Event

Pyro.config.PYRO_MOBILE_CODE=1 

#Nomenclature: 
#    Atlas    - a MINC volume and corresponding label volume
#    Template - Refers to the application of atlas labels applied to a subject brain 
#               for use in the MAGeT pipeline.
#               Ambiguously, this may also be used as a relative term to mean any 
#               labeled brain (i.e. an atlas).
#    Model    - Relative term.  An atlas, or a template.  May also refer to 
#               ancillary files related to an Atlas (e.g. blurs, masks, etc.).  
#    Subject  - Target brain of the MAGeT pipeline that is to be labeled.  
 
class Template:
    def __init__(self, image, labels, mask=None, model_dir=None, outputdir=None):
        """image     - template image file
           labels    - list of label files
           mask      - label mask for 'image'
           model_dir - directory of blurred images used for linear resampling
           outputdir - (deprecated)"""
           
        if type(labels) == str:
            labels = [labels]
            
        assert type(labels) == list, "labels must be a string or a list of strings"
        
        self.image     = image
        self.labels    = labels
        self.mask      = mask
        self.model_dir = model_dir
        if outputdir == None:
            self.outputdir = dirname(image)
        else:
            self.outputdir = outputdir

class SMATregister:
    """Register input image to a labelled template"""
    def __init__(self, target, template, outDir="",
                 steps=[4,2,1],
                 name="initial"):
        
        self.p = Pipeline()
        self.input = target
      
        
        fh = mincFileHandling()
            
        input_base_fname = fh.removeFileExt(target)
        input_dir, input_base = fh.createSubDirSubBase(abspath(outDir), input_base_fname, input_base_fname)
        log_dir, log_base = fh.createLogDirLogBase(input_dir, input_base_fname)
        
        self.outDir = input_dir
        
        # nu_correct the like
        inuc, inuc_log = fh.createOutputAndLogFiles(input_base, log_base, "_nuc.mnc" )
        p.addStage(nu_correct(target, inuc, inuc_log))
        
        # linear registration to TAL space, using the template as a model      
        linxfm, tal_log = fh.createXfmAndLogFiles(input_base, log_base, ".xfm")
        assert template.model_dir != "", "Expected model directory supplied for mritotal linear registration step"        
        p.addStage(minctotal(inuc, linxfm, template.model_dir, template.image, tal_log))

        #p.addStage(linearminctracc(template_blurs[0],
        #                               input_blurs[0], linxfm,
        #                               logfile, linearparam,
        #                               inputMask, template.mask))

        # apply linear registration
        linres, linres_log = fh.createResampledAndLogFiles(input_base, log_base, ["linres"])
        p.addStage(mincresample(inuc, linres, linres_log, argarray=["-sinc", "-width"], like=template.image, cxfm=linxfm))
        
        # create the nonlinear registrations
        # by default three iterations: 
        #    1. step = '4', '4', '4'; lattice_diam = '12', '12', '12'; sub_lattice = 8; ident
        #    2. step = '2', '2', '2'; lattice_diam = '6', '6', '6'; sub_lattice = 8; ident
        #    3. step = '1', '1', '1'; lattice_diam = '3', '3', '3'; sub_lattice = 6
        self.output = linres
        return 
    
        xfms = []
        for i in range(len(steps)):
            cxfm, logfile = fh.createXfmAndLogFiles(input_base, log_base, ["step", str(i)])
            if i == 0:
                pxfm = linxfm
            else:
                pxfm = xfms[i - 1]
            xfms += [cxfm]
            p.addStage(minctracc(linres, subcortical, cxfm, logfile,
                                      step=steps[i],
                                      transform=pxfm))
        
        # resample image with final registration
        
        # resample labels with final registration
        self.output, logfile = fh.createResampledAndLogFiles(input_base, log_base, [tbase])
        resargs = ["-keep_real_range", "-nearest_neighbour"]
        self.p.addStage(mincresample(template.labels, self.output, logfile, resargs, target, cxfm))
    
    def getTemplate(self):
        return(Template(self.input, self.output, outputdir=self.outDir))

if __name__ == "__main__":
    usage = "%prog [options] input1.mnc ... inputn.mnc"
    description = "description needed"

    parser = OptionParser(usage=usage, description=description)

    parser.add_option("--atlas-labels", "-a", dest="atlas_labels",
                      type="string", 
                      action="append",
                      help="MINC volume containing labelled structures")
    parser.add_option("--atlas-image", "-i", dest="atlas_image",
                      type="string",
                      help="MINC volume of image corresponding to labels")
    parser.add_option("--atlas-models", dest="atlas_models", 
                      type="string",
                      help="Directory containing blurred and masked MINC volumes of atlas image")
    
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
    
    # PydPiper options
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
                      type="int", default=1, 
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
        fh = FileHandling()
        outputDir = abspath(options.template_library)
        if not isdir(outputDir):
            mkdir(outputDir)
        tmplDir = fh.createSubDir(outputDir, "atlas")
        tmpl = Template(options.atlas_image, 
                        options.atlas_labels,
                        mask=options.mask, 
                        outputdir=tmplDir, 
                        model_dir=options.atlas_models)
        
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
                sp = SMATregister(args[nfile], tmpl, outDir=outputDir)
                templates.append(sp.getTemplate())
                p.addPipeline(sp.p)

            # once the initial templates have been created, go and register each
            # file to the templates
#            for file in args:
#                labels = []
#                for t in templates:
#                    sp = SMATregister(file, t, outDir=outputDir, inputMask=options.mask, name="templates")
#                    labels.append(InputFile(sp.output))
#                    p.addPipeline(sp.p)
#                bname = fh.removeFileExt(file)
#                base = fh.createBaseName(outputDir, bname + "_votedlabels")
#                out, log = fh.createOutputAndLogFiles(base, base, ".mnc")
#                cmd = ["voxel_vote.py"] + labels + [OutputFile(out)]
#                voxel = CmdStage(cmd)
#                voxel.setLogFile(LogFile(log))
#                p.addStage(voxel)

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

    
