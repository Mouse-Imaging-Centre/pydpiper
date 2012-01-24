#!/usr/bin/env python

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
    def __init__(self, image, labels, mask=None, model_dir=None, roi=None):
        """image     - template image file
           labels    - list of label files
           mask      - label mask for 'image'
           model_dir - directory of blurred images used for linear resampling by mritotal
           roi       - MINC volume containing a region of the atlas image to do non-linear registration on."
        """
           
        if type(labels) == str:
            labels = [labels]
            
        assert type(labels) == list, "labels must be a string or a list of strings"
        
        self.image     = image
        self.labels    = labels
        self.mask      = mask
        self.model_dir = model_dir
        self.roi       = roi


class SMATregister:
    def __init__(self, target, template, root_dir=""):
        """Register input image (target) to a labelled template (template).
        
            root_dir is the directory under which to place the output directories and files. 
        """
        
        
        self.target = target
        self.template = template
        self.root_dir = root_dir
        
    
    def build_pipeline(self):
        p = Pipeline()
        
        # file handling stuffs
        
        # Create the output directories, in the following pattern:
        #
        # <root_dir>/<target>/<template>/
        #
        # where root_dir is given when this object was initialised
        #       <target> is the name of the target image 
        #       <template> is the name of the template image
        #
        # log files will appear in:
        # <root_dir>/<target>/<template>/log/
        #
        # in these folders will be prefixed with <target>.  e.g. H00234_nuc.mnc 
        # (if <target> is H00234)
        # 
        target_base_fname   = fh.removeFileExt(self.target)
        template_base_fname = fh.removeFileExt(self.template.image)
        output_target_dir = fh.createSubDir(abspath(self.root_dir), target_base_fname)            
        output_dir, output_base_fname = fh.createSubDirSubBase(output_target_dir, template_base_fname, target_base_fname)
        log_dir, log_base = fh.createLogDirLogBase(output_dir, target_base_fname)
    
        inuc, inuc_log = fh.createOutputAndLogFiles(output_base_fname, log_base, "_nuc.mnc" )
        p.addStage(nu_correct(self.target, inuc, inuc_log))
        
        # linear registration to TAL space, using the template as a model      
        linxfm, linreg_log = fh.createXfmAndLogFiles(output_base_fname, log_base, ["lin"])
        p.addStage(bestlinreg(inuc, self.template.image, linxfm, linreg_log))
        #assert template.model_dir != "", "Expected model directory supplied for mritotal linear registration step"        
        #p.addStage(mritotal(inuc, linxfm, template.model_dir, fh.removeFileExt(template.image), linreg_log))
        

        linres, linres_log = fh.createResampledAndLogFiles(output_base_fname, log_base, ["linres"])
        p.addStage(mincresample(inuc, linres, linres_log, argarray=["-sinc", "-width", "2"], like=self.template.image, cxfm=linxfm))

        # non-linear registration        
        iterations = 15  
        nl0, logfile0 = fh.createXfmAndLogFiles(output_base_fname, log_base, ["step_0"])
        nl1, logfile1 = fh.createXfmAndLogFiles(output_base_fname, log_base, ["step_1"])
        nl2, logfile2 = fh.createXfmAndLogFiles(output_base_fname, log_base, ["step_2"])
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
        
        nlxfm, logfile_nl = fh.createXfmAndLogFiles(output_base_fname, log_base, ["nl"])
        p.addStage(xfmconcat([linxfm, nl2], nlxfm, logfile_nl))
        
        # resample labels with final registration
        resargs = ["-keep_real_range", "-nearest_neighbour", "-invert"]
        
        outputs = []
        
        self.template.labels.sort()  #TODO: hack: sort the labels, so that the indices correlate across subjects (works for up to 10 labels!)
        for label, index in zip(self.template.labels, range(len(self.template.labels))):
            output, logfile = fh.createResampledAndLogFiles(output_base_fname, log_base, ["label", str(index)])
            outputs.append(output)        
            p.addStage(mincresample(label, output, logfile, resargs, cxfm=nlxfm, like=inuc))
    
        output_template = Template(self.target, outputs, roi=self.target)
        
        return (p, output_template)
    

if __name__ == "__main__":
    usage = "%prog [options] subjects_dir"
    description = "subjects_dir holds the subject brain images in .mnc format"

    parser = OptionParser(usage=usage, description=description)

    # atlas options
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
    parser.add_option("--atlas-roi", "-r", dest="atlas_roi",
                      type="string",
                      help="MINC volume containing a region of the atlas image to do non-linear registration on")
    parser.add_option("--mask", dest="mask",
                      type="string",
                      help="Mask to use for all images")
    parser.add_option("--max-templates", dest="max_templates",
                      default=25, type="int",
                      help="Maximum number of templates to generate")
    
    # output options
    parser.add_option("--output-dir", dest="output_directory",
                      type="string", default="output",
                      help="Directory where output (template library and segmentations are stored.")    
    
    
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
                         model_dir=options.atlas_models,
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
                labels = {}
                for t in templates:
                    sp = SMATregister(subject, t, root_dir=segmentation_dir)
                    pipeline, output_template = sp.build_pipeline()
                    for index in range(len(output_template.labels)):
                        labels[index] = labels.get(index, []) + [InputFile(output_template.labels[index])]
                    p.addPipeline(pipeline)
                    
                subject_base_fname = fh.removeFileExt(subject)
                subject_dir = os.path.join(segmentation_dir,subject_base_fname)
                
                for index, label_set in labels.items():
                    voted_dir, base_name = fh.createSubDirSubBase(subject_dir, "final", subject_base_fname + "_votedlabels_" + str(index))
                    voted_labels, log = fh.createOutputAndLogFiles(base_name, base_name, ".mnc")
                    cmd = ["voxel_vote.py"] + label_set + [OutputFile(voted_labels)]
                    voxel = CmdStage(cmd)
                    voxel.setLogFile(LogFile(log))
                    p.addStage(voxel)

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

    
