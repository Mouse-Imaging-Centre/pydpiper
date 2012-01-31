#!/usr/bin/env python

import logging
from pydpiper.pipeline import *
from pydpiper.queueing import *
from minctools import minctracc, nu_correct, bestlinreg, mincresample, xfmconcat, mincFileHandling
from optparse import OptionParser
from os.path import dirname, abspath
from os import mkdir
import networkx as nx
from multiprocessing import Event
import copy 
import glob 

Pyro.config.PYRO_MOBILE_CODE=1 
fh = mincFileHandling()
test_mode = False
logger = logging.getLogger("MAGeT")
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
    def __init__(self, image, labels, mask=None, roi=None, directory = None):
        """
           Represents an atlas or a generated template.  
           
           image     - brain image file
           labels    - label image file
           roi       - MINC volume containing a region of the atlas image to do non-linear registration on.
           directory - Directory where this generated template exists. None if not known.
        """
           
        assert type(labels) == str, "labels must be a pointer to a label file (as a string)"
        
        self.image     = image
        self.labels    = labels
        self.roi       = roi or image
        self.mask      = mask
        self.directory = directory   


class SMATregister:
    def __init__(self, target, template, root_dir=""):
        """Register input image (target) to a labelled template (template).
        
            root_dir is the directory under which to place the output directories and files.
            The directory structure produced is <root_dir>/<template>/<target>/. All registration files are placed
            in that folder  
        """
        
        
        self.target = target
        self.template = template
        self.root_dir = root_dir
        
    
    def build_pipeline(self):
        p = Pipeline()
        
        # We create the output directories, in the following pattern:
        #
        # <root_dir>/<template>/<target>/
        #
        # where root_dir is given when this object was initialised 
        #       <target> is the name of the target image
        #       <template> is the name of the template image
        #
        # This folder is called the output_dir.
        #
        # log files will appear in:
        # <output_dir>/log/
        #
        # 
        target_base_fname   = fh.removeFileExt(self.target)
        template_base_fname = fh.removeFileExt(self.template.image)
        output_template_dir = fh.createSubDir(abspath(self.root_dir), template_base_fname)
        
        output_dir = fh.createSubDir(output_template_dir, target_base_fname)    
        log_dir = fh.createLogDir(output_dir)
    
        log_base = log_dir + "/"
        output_base_fname = output_dir + "/"
        
        inuc, inuc_log = fh.createOutputAndLogFiles(output_base_fname, log_base, "nuc.mnc" )
        p.addStage(nu_correct(self.target, inuc, inuc_log))
        
        # linear registration to TAL space, using the template as a model      
        linxfm, linreg_log = fh.createXfmAndLogFiles(output_base_fname, log_base, ["lin"])
        p.addStage(bestlinreg(inuc, self.template.image, linxfm, linreg_log))
        
        #assert template.model_dir != "", "Expected model directory supplied for mritotal linear registration step"        
        #p.addStage(mritotal(inuc, linxfm, template.model_dir, fh.removeFileExt(template.image), linreg_log))
        
        if not test_mode:
            linres, linres_log = fh.createMincAndLogFiles(output_base_fname, log_base, ["linres"])
            p.addStage(mincresample(inuc, linres, linres_log, argarray=["-sinc", "-width", "2"], like=self.template.image, cxfm=linxfm))
    
            # non-linear registration        
            iterations = 15  
            nl0, logfile0 = fh.createXfmAndLogFiles(output_base_fname, log_base, ["nl_0"])
            nl1, logfile1 = fh.createXfmAndLogFiles(output_base_fname, log_base, ["nl_1"])
            nl2, logfile2 = fh.createXfmAndLogFiles(output_base_fname, log_base, ["nl_2"])
        

            p.addStage(minctracc(linres, self.template.roi, nl0, logfile0, 
                                step=4,
                                sub_lattice=8, 
                                iterations=iterations, 
                                ident=True))
            
        
            p.addStage(minctracc(linres, self.template.roi, nl1, logfile1, 
                                 step=2,
                                 sub_lattice=8, 
                                 transform = nl0,
                                 iterations=iterations,
                                 ident=True))
            p.addStage(minctracc(linres, self.template.roi, nl2, logfile2, 
                                 step=1,
                                 sub_lattice=6, 
                                 transform = nl1,
                                 iterations=iterations))
        
            nlxfm, logfile_nl = fh.createXfmAndLogFiles(output_base_fname, log_base, ["reg"])
            p.addStage(xfmconcat([linxfm, nl2], nlxfm, logfile_nl))
            
        else:
            nlxfm = linxfm
            
        # resample labels with final registration
        resargs = ["-nearest_neighbour", "-invert", "-byte"]
        labels, logfile = fh.createMincAndLogFiles(output_base_fname, log_base, ["labels"])
        p.addStage(mincresample(self.template.labels, labels, logfile, resargs, cxfm=nlxfm, like=inuc))
        
        outputs = []
        outputs.append(labels)        
        
        output_template = Template(self.target, labels, roi=self.target)
        
        return (p, output_template)
    

if __name__ == "__main__":
    usage = "%prog [options] subjects_dir"
    description = "subjects_dir holds the subject brain images in .mnc format"

    parser = OptionParser(usage=usage, description=description)

    # atlas options
    parser.add_option("--atlas-labels", "-a", dest="atlas_labels",
                      type="string", 
                      help="MINC volume containing labelled structures")
    parser.add_option("--atlas-image", "-i", dest="atlas_image",
                      type="string",
                      help="MINC volume of image corresponding to labels")
    parser.add_option("--atlas-roi", "-r", dest="atlas_roi",
                      type="string",
                      help="MINC volume containing a region of the atlas image to do non-linear registration on")
    parser.add_option("--mask", dest="mask",
                      type="string",
                      help="Mask to use for all images")
    parser.add_option("--max-templates", dest="max_templates",
                      default=25, type="int",
                      help="Maximum number of templates to generate")
    parser.add_option("--validation-labels", dest="validation_labels",
                      type="string", 
                      help="Directory containing label files corresponding to input templates (form: <input>_labels.mnc) used for validation")
    
    # output options
    parser.add_option("--output-dir", dest="output_directory",
                      type="string", default="output",
                      help="Directory where output (template library and segmentations are stored.")    
    
    parser.add_option("--test-mode", dest="test_mode", 
                      action="store_true", 
                      default=False, 
                      help="Run this code in a testing mode.  Registration is simplified to make for much shorter process.")
    
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
                      type="int", default=0, 
                      help="Launch executors automatically without having to run pipeline_excutor.py independently.")
    parser.add_option("--proc", dest="proc", 
                      type="int", default=4,
                      help="Number of processes per executor. Default is 4. Also sets max value for processor use per executor. Overridden if --num-executors not specified.")
    parser.add_option("--mem", dest="mem", 
                      type="float", default=4,
                      help="Total amount of requested memory. Default is 4G. Overridden if --num-executors not specified.")
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
        test_mode = options.test_mode
        
        outputDir = abspath(options.output_directory)
        if not isdir(outputDir):
            mkdir(outputDir)
        
        # create the output directories 
        # template_dir holds all of the generated templates
        # segmentation_dir holds all of the participant segmentations, including the final voted on labels
        template_dir = fh.createSubDir(outputDir, "template-lib")
        segmentation_dir= fh.createSubDir(outputDir, "segmentations")
        
        atlas = Template(options.atlas_image, 
                         options.atlas_labels,
                         mask=options.mask, 
                         roi = options.atlas_roi )
        
        p = Pipeline()
        p.setBackupFileLocation(outputDir)

        if options.restart:
            print "Restarting pipeline"
            p.restart()
        else:
            subjects_dir = args[0]
            
            # get a list of the subjects
            subject_files = glob.glob(os.path.join(subjects_dir,"*.mnc"))
            
            # create the initial templates - either total number of files
            # or the maximum number of templates, whichever is lesser
            templates = []
            numTemplates = min(len(subject_files), options.max_templates)
            
    
            for nfile in range(numTemplates):
                sp = SMATregister(subject_files[nfile], atlas, root_dir=template_dir)
                pipeline, output_template = sp.build_pipeline()
                templates.append(output_template)
                p.addPipeline(pipeline)

            # once the initial templates have been created, go and register each
            # subject to the templates
            for subject in subject_files:
                labels = []
                for t in templates:
                    sp = SMATregister(subject, t, root_dir=segmentation_dir)
                    pipeline, output_template = sp.build_pipeline()
                    labels.append(InputFile(output_template.labels))
                    p.addPipeline(pipeline)
                    
                subject_base_fname = fh.removeFileExt(subject)
                subject_dir = os.path.join(segmentation_dir,subject_base_fname)
                
                # vote!
                final_dir, base_name = fh.createSubDirSubBase(subject_dir, "final", subject_base_fname + "_labels")
                voted_labels, log = fh.createOutputAndLogFiles(base_name, base_name, ".mnc")
                cmd = ["voxel_vote.py"] + labels + [OutputFile(voted_labels)]
                vote = CmdStage(cmd)
                vote.setLogFile(LogFile(log))
                p.addStage(vote)
                    
                if not options.validation_labels:
                    continue
                
                # check if there is a validation label set for this subject
                validation_labels = os.path.join(options.validation_labels, fh.removeFileExt(subject) + "_labels.mnc")
                if not os.path.exists(validation_labels):
                    continue
                    
                validation_output_file = os.path.join(final_dir, "validation.csv")
                validate = CmdStage(["volume_similarity.sh", InputFile(voted_labels), validation_labels, OutputFile(validation_output_file)])
                validate.setLogFile(LogFile(os.path.join(final_dir, "validation.log")))
                p.addStage(validate)
                    
            p.initialize()
            p.printStages()
    
        if options.create_graph:
            print "Writing dot file..."
            nx.write_dot(p.G, "labeled-tree.dot")
            print "Done."
        #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
        # if use_ns is specified, Pyro NameServer must be started. 
        returnEvent = Event()
        pipelineDaemon(p, returnEvent, options, sys.argv[0])
        returnEvent.wait()
        print "templates: " + str(numTemplates)

    
