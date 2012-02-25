#!/usr/bin/env python

import logging
from pydpiper.pipeline import Pipeline, InputFile, OutputFile, LogFile, CmdStage
from pydpiper.application import AbstractApplication
from minctools import minctracc, nu_correct, bestlinreg, mincresample, xfmconcat, mincFileHandling
import os.path
import glob 


fh = mincFileHandling()
logger = logging.getLogger(__name__)

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
    def __init__(self, image, labels, mask=None, roi=None, directory = None, labels_dir = None, reg_dir = None):
        """
           Represents an atlas or a generated template.  
           
           image     - brain image file
           labels    - label image file
           roi       - MINC volume containing a region of the atlas image to do non-linear registration on.
           directory - Directory where this generated template exists. None if not known.
           labels_dir - Directory where the generated label file exists for this template.
           reg_dir   - Directory where the generated registration xfms exist for this template.
        """
           
        assert type(labels) == str, "labels must be a pointer to a label file (as a string)"
        
        self.image     = image
        self.labels    = labels
        self.roi       = roi or image
        self.mask      = mask
        self.directory = directory
        self.labels_dir = labels_dir
        self.reg_dir   = reg_dir 


class BasicLabelPropogationStrategy:
    def __init__(self, target, template, root_dir, labels_dir):
        """Register input image (target) to a labelled template (template) and propogate its labels.
        
            root_dir is the directory under which to place the output pairwise registrations, 
            labels_dir is the directory under which to place the output labels.
            
            The directory structure for registrations are <root_dir>/<template>/<target>/. 
            All transform registration files are placed in that folder.
            
            Labels appear in the <labels_dir>/<template>/<target>/ folder
        """
        
        
        self.target = target
        self.template = template
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        
    def do_linear_registration(self, p, log_base, output_base_fname, inuc):
        linxfm, linreg_log = fh.createOutputAndLogFiles(output_base_fname, log_base, "lin.xfm")
        p.addStage(bestlinreg(inuc, self.template.image, linxfm, linreg_log))
        return linxfm


    def do_non_linear_registration(self, p, log_base, output_base_fname, inuc, linxfm):
        linres, linres_log = fh.createOutputAndLogFiles(output_base_fname, log_base, "linres.mnc")
        p.addStage(mincresample(inuc, linres, linres_log, argarray=["-sinc", "-width", "2"], like=self.template.image, cxfm=linxfm))
        
        # non-linear registration
        iterations = 15
        nl0, logfile0 = fh.createOutputAndLogFiles(output_base_fname, log_base, "nl_0.xfm")
        nl1, logfile1 = fh.createOutputAndLogFiles(output_base_fname, log_base, "nl_1.xfm")
        nl2, logfile2 = fh.createOutputAndLogFiles(output_base_fname, log_base, "nl_2.xfm")
        p.addStage(minctracc(linres, self.template.roi, nl0, logfile0, step=4, sub_lattice=8, iterations=iterations, ident=True))
        p.addStage(minctracc(linres, self.template.roi, nl1, logfile1, step=2, sub_lattice=8, transform=nl0, iterations=iterations, ident=True))
        p.addStage(minctracc(linres, self.template.roi, nl2, logfile2, step=1, sub_lattice=6, transform=nl1, iterations=iterations))
        nlxfm, logfile_nl = fh.createOutputAndLogFiles(output_base_fname, log_base, "reg.xfm")
        p.addStage(xfmconcat([linxfm, nl2], nlxfm, logfile_nl))
        return nlxfm


    def create_base_names(self, output_dir):
        target_base_fname = fh.removeFileExt(self.target)
        template_base_fname = fh.removeFileExt(self.template.image)
        output_template_dir = fh.createSubDir(os.path.abspath(output_dir), template_base_fname)
        output_dir = fh.createSubDir(output_template_dir, target_base_fname)
        log_dir = fh.createLogDir(output_dir)
        log_base = log_dir + "/"
        output_base_fname = output_dir + "/"
        return output_base_fname, log_base, output_dir

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
        output_base_fname, log_base, output_dir = self.create_base_names(self.root_dir)
        
        inuc, inuc_log = fh.createOutputAndLogFiles(output_base_fname, log_base, "nuc.mnc" )
        p.addStage(nu_correct(self.target, inuc, inuc_log))
        
        linxfm = self.do_linear_registration(p, log_base, output_base_fname, inuc)
        nlxfm = self.do_non_linear_registration(p, log_base, output_base_fname, inuc, linxfm)
        
        # resample labels with final registration
        labels_base_fname, labels_log_base, labels_output_dir = self.create_base_names(self.labels_dir)
        labels, logfile = fh.createOutputAndLogFiles(labels_base_fname, labels_log_base, "labels.mnc" )
        p.addStage(mincresample(self.template.labels, labels, logfile, ["-nearest_neighbour", "-invert", "-byte"], cxfm=nlxfm, like=inuc))      
        
        output_template = Template(self.target, labels, roi=self.target, labels_dir = labels_output_dir, reg_dir = output_dir)
        
        return (p, output_template)

class TestLabelPropogationStrategy(BasicLabelPropogationStrategy):
    """A testing version of this strategy which skips the non-linear registration step."""
    def __init__(self, target, template, root_dir, labels_dir):
        BasicLabelPropogationStrategy.__init__(self, target, template, root_dir, labels_dir)
    def do_non_linear_registration(self, p, log_base, output_base_fname, inuc, linxfm):
        linres, linres_log = fh.createOutputAndLogFiles(output_base_fname, log_base, "linres.mnc")
        p.addStage(mincresample(inuc, linres, linres_log, argarray=["-sinc", "-width", "2"], like=self.template.image, cxfm=linxfm))
        return linxfm
        
class BasicMAGeT():
    def __init__(self, atlases, subject_images):
        self.atlases = atlases
        self.subject_images = subject_images
        self.max_templates = len(subject_images)
        self.label_propagation_method = BasicLabelPropogationStrategy
        
    def set_label_propagation_method(self, method):
        """Sets the label propagation method to use.
        
           Method should be a SMATRegistration or subclass
        """
        self.label_propagation_method = method
    
    def set_max_templates(self, max_templates):
        """Sets the maximum number of subjects to use in the template library.""" 
        self.max_templates = max_templates
        
    def get_templates(self):
        """Returns the subject images to be used to build the template library.
        
           Return value is simply a list of paths to images."""
        return self.subject_images[:min(len(self.subject_images), self.max_templates)]
        
    def build_pipeline(self, pipeline, registrations_dir, labels_dir):
        self.templates = []
        
        # for each atlas, register to all of the templates in order to build the template library
        for atlas in self.atlases: 
            for subject_image in self.get_templates():
                sp = self.label_propagation_method(subject_image, atlas, root_dir=registrations_dir, labels_dir = labels_dir)
                p, output_template = sp.build_pipeline()
                self.templates.append(output_template)
                pipeline.addPipeline(p)

        # once the initial templates have been created, go and register each
        # subject to the templates
        subjects_labels = {}
        for subject_image in self.subject_images:
            labels = []
            for t in self.templates:
                tmpl_labels_dir = os.path.dirname(t.labels_dir) 
                sp = self.label_propagation_method(subject_image, t, root_dir=registrations_dir,  labels_dir = tmpl_labels_dir)
                p, output_template = sp.build_pipeline()
                labels.append(output_template.labels)
                pipeline.addPipeline(p)
            subjects_labels[subject_image] = labels
        return subjects_labels


def get_labels_for_image(image_file, labels_dir):
    """Get the labels file for the given image.
    
       Look in the labels_dir for a file named <image_file>_labels.mnc
    """
    return os.path.join(labels_dir, fh.removeFileExt(image_file) + "_labels.mnc")

def xcorr_vote_all_subjects(subject_files, templates, output_dir, pipeline):
    #
    # for each subject, calculate the cross-correlation between it and each of the templates.
    #  
    # cross-correlation is calculated on a mask created from the dialated average of all the labels
    # across all of the templates for the subject
    #
    # Output for this step is in the following form:
    #  <output_dir>/<subject>/xcorr_mask.mnc
    #  <output_dir>/<subject>/labels.mnc    
    #  <template>/<subject>/template_xcorr.txt
    
    for subject_file in subject_files:
        xcorr_vote(subject_file, templates, output_dir, pipeline)
        
def xcorr_vote(subject_file, templates, output_dir, pipeline, reg_dir):
        # STEP 1: calculate the correlation masks for each subject
        subject_name = fh.removeFileExt(subject_file)
        subject_xcorr_base = fh.createSubDir(output_dir, subject_name) + "/"
        
        # first, gather all of the label files ...
        template_dirs = [template.labels_dir for template in templates]
        subject_label_files = map(lambda x: os.path.join(x, subject_name, "labels.mnc"), template_dirs)
        
        # ... and average them
        cmd, merged_labels_file = single_output_command_helper("mincaverage", subject_xcorr_base, "merged_labels.mnc", subject_label_files)
        pipeline.addStage(cmd)
        
        # ... threshold so we only get labels
        cmd, thesholded_labels_file = single_output_command_helper("minccalc", subject_xcorr_base, "thresholded_labels.mnc", [merged_labels_file], args=["-expression", "A[0]>0"])
        pipeline.addStage(cmd)
        
        # .. dialate to create the mask
        cmd, label_mask_file = single_output_command_helper("mincmorph", subject_xcorr_base, "label_mask.mnc", [thesholded_labels_file], args=["-successive", "DDD"])
        pipeline.addStage(cmd)
        
        # STEP 2: using this mask, calculate the correlation between this subject, and all of the templates
        
        # to parallel lists holding xcorr between a subject and template, and the corresponding propagated labels
        subject_xcorrs_list = []
        subject_labels_list = []
         
        for template in templates: 
            subject_dir = os.path.join(reg_dir, fh.removeFileExt(template.image), subject_name) + "/"
            subject_template_linreg = os.path.join(subject_dir, "linres.mnc")
            subject_labels = os.path.join(template.labels_dir, subject_name, "labels.mnc")
            
            cmd, subject_xcorr = single_output_command_helper("xcorr_vol.sh", subject_dir, "template_xcorr.txt", [subject_template_linreg, template.image, label_mask_file])
            pipeline.addStage(cmd)
            
            subject_xcorrs_list.append(subject_xcorr)
            subject_labels_list.append(subject_labels)
            
        # STEP 3: vote!
        cmd, xcorr_voted_labels = single_output_command_helper("xcorr_vote.py", subject_xcorr_base, "labels.mnc", subject_labels_list + subject_xcorrs_list)
        pipeline.addStage(cmd)
        return xcorr_voted_labels
        

def majority_vote_all_subjects(subject_files, templates, output_dir, pipeline):
    for subject_file in subject_files:
        subject_name = fh.removeFileExt(subject_file)
        
        # first, gather all of the label files ...
        template_dirs = [template.labels_dir for template in templates]
        subject_label_files = map(lambda x: os.path.join(x, subject_name, "labels.mnc"), template_dirs)
        
        majority_vote(subject_file, subject_label_files, output_dir, pipeline)
        
def majority_vote(subject, labels, output_dir, pipeline):
    subject_base_fname = fh.removeFileExt(subject)
    vote_dir = fh.createSubDir(output_dir, subject_base_fname)     
    cmd, voted_labels = single_output_command_helper("voxel_vote.py", vote_dir, "labels.mnc", labels)
    pipeline.addStage(cmd)
    return voted_labels

def compare_similarity(image_path, expected_labels_path, computed_labels_path, output_dir, pipeline):       
    compare_base = fh.createSubDir(output_dir, fh.removeFileExt(image_path)) 
    cmd, validation_output_file = single_output_command_helper("volume_similarity.sh", compare_base, "validation.csv", [expected_labels_path, computed_labels_path])
    pipeline.addStage(cmd)
    return validation_output_file
        
def single_output_command_helper(command_name,  output_base, output, input_files = [], args = []):
    """Returns a properly configured CmdStage from the given input."""
    output_base = output_base[-1] == "/" and output_base or output_base + "/"  # nasty way of verifying the trailing slash
    log_base = fh.createLogDir(output_base) + "/"
    
    output_file, log_file = fh.createOutputAndLogFiles(output_base, log_base, output)
    cmd = CmdStage([command_name] + args + map(lambda x: InputFile(x), input_files) + [OutputFile(output_file)]) 
    cmd.setLogFile(LogFile(log_file))
    return cmd, output_file
    
class MAGeTApplication(AbstractApplication):
    def setup_options(self):
        self.parser.add_option("--atlas-labels", "-a", dest="atlas_labels",
                          type="string", 
                          help="MINC volume containing labelled structures")
        self.parser.add_option("--atlas-images", "-i", dest="atlas_images",
                          type="string",
                          help="MINC volume of image corresponding to labels")
        self.parser.add_option("--atlas-roi", "-r", dest="atlas_roi",
                          type="string",
                          help="MINC volume containing a region of the atlas image to do non-linear registration on")
        self.parser.add_option("--mask", dest="mask",
                          type="string",
                          help="Mask to use for all images")
        self.parser.add_option("--max-templates", dest="max_templates",
                          default=25, type="int",
                          help="Maximum number of templates to generate")
        self.parser.add_option("--output-dir", dest="output_directory",
                          type="string", default="output",
                          help="Directory where output (template library and segmentations are stored.")
        self.parser.add_option("--test-mode", dest="test_mode",
                          action="store_true", default=False,
                          help="Turns on a testing mode in which the pipeline is simplified and produces output of reduced quality.")
        
        self.parser.set_usage("%prog [options] subjects_dir")        

    def run(self):
        options = self.options
        args = self.args
        
        test_mode = options.test_mode
        if test_mode: 
            logging.info("Test mode is on. Pipeline is simplified. Don't expect great results. ")
            
        outputDir = os.path.abspath(options.output_directory)
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)
        
        # create the output directories 
        # template_dir holds all of the generated templates
        # segmentation_dir holds all of the participant segmentations, including the final voted on labels
        registrations_dir = fh.createSubDir(outputDir, "registrations")
        labels_dir = fh.createSubDir(outputDir, "labels")
        
        subjects_dir = args[0]
        atlas_images_dir = options.atlas_images
        atlas_labels_dir = options.atlas_labels
        
        atlases = []
        for atlas_image in glob.glob(os.path.join(atlas_images_dir, "*.mnc")):
            atlases.append(Template(atlas_image, get_labels_for_image(atlas_image, atlas_labels_dir)))
        
        subject_files = glob.glob(os.path.join(subjects_dir,"*.mnc"))
        
        maget = BasicMAGeT(atlases, subject_files)
        if test_mode: 
            maget.set_label_propagation_method(TestLabelPropogationStrategy)
        maget.set_max_templates(options.max_templates)
        maget.build_pipeline(self.pipeline, registrations_dir, labels_dir)
        
        # fuse labels!     
        majority_vote_dir = fh.createSubDir(outputDir, "labels_majority_vote")
        majority_vote_all_subjects(subject_files, maget.templates, majority_vote_dir, self.pipeline)
        
        #xcorr_dir = fh.createSubDir(outputDir, "labels_xcorr_vote")
        #xcorr_vote_vote_all_subjects(subject_files, maget.templates, xcorr_dir, self.pipeline)
        
        
            
        
if __name__ == "__main__":
    application = MAGeTApplication()
    application.start()
    
