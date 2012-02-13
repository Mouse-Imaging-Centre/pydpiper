'''
Created on Feb 8, 2012

@author: jp
'''
from MAGeT import *
import logging
import os
import glob
import random
from pydpiper.application import AbstractApplication


class ValidationMAGeT(BasicMAGeT):
    """This variation of the MAGeT algorithm allows you to configure a disjoint set of subjects and templates."""
    def __init__(self, atlas_templates, template_images, subject_images):
        BasicMAGeT.__init__(self, atlas_templates, subject_images)
        self.template_images = template_images
        
    def get_templates(self):
        """Returns the subject images to be used to build the template library.
        
           Return value is simply a list of paths to images."""
        return self.template_images

class MAGeTSubSamplingCrossValidationApp(AbstractApplication):
    """Performs Leave One Out Cross Validation on a given test set.
    
       A number of validation runs are performed.  In each run, the test set is randomly permuted, and 
       the permutation is partitioned into atlases, the template images, and the validation image.  
       Traditional MAGeT is performed wherein the atlases are registered to each template image to 
       form the template library.  Then, the validation image is labeled by merging labels predicted 
       from the template library.  
    """
    def setup_options(self):
        self.parser.add_option("--num-atlases", dest="num_atlases",
                          default=3, type="int",
                          help="Maximum number of atlases to use in each validation")

        self.parser.add_option("--num-validations", dest="num_validations",
                          default=10, type="int",
                          help="Maximum number of validation iterations to do")
                
        self.parser.add_option("--output-dir", dest="output_directory",
                          type="string", default="output",
                          help="Directory where output (template library and segmentations are stored.")
        
        self.parser.add_option("--test-mode", dest="test_mode",
                          action="store_true", default=False,
                          help="Turns on a testing mode in which the pipeline is simplified and produces output of reduced quality.")

        self.parser.add_option("--random-seed", dest="seed",
                          type="int", default=1,
                          help="Sets the random number generator seed value.")
        
        
        self.parser.set_usage("%prog [options] images_dir labels_dir")        

    def run(self):
        options = self.options
        args = self.args
        
        test_mode = options.test_mode
        if test_mode: 
            logging.info("Test mode is on. Pipeline is simplified. Don't expect great results. ")
            
        outputDir = os.path.abspath(options.output_directory)
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)
        
        random.seed(options.seed)
        
        # create the output directories 
        # template_dir holds all of the generated templates
        # segmentation_dir holds all of the participant segmentations, including the final voted on labels
        registrations_dir = fh.createSubDir(outputDir, "registrations")
        
        input_images_dir = args[0]
        input_labels_dir = args[1]

        brains = []
        for image in glob.glob(os.path.join(input_images_dir, "*.mnc")):
            brains.append(Template(image, get_labels_for_image(image, input_labels_dir)))
        
        num_validations = options.num_validations
        
        # begin the validation
        for i in range(num_validations):
            print "Validation #", i
            subjects = brains[:]
            random.shuffle(subjects)
            
            atlases = subjects[:options.num_atlases]
            templates = subjects[options.num_atlases:]
            validators = templates 
            
            assert len(templates) > 0, "Not enough inputs to create atlas, template and validation set."
            
            template_files = [template.image for template in templates]
            validation_files = [validator.image for validator in validators]
            
            print "Running MAGeT with the following configuration:"
            print "\t Atlases: "
            print "\t\t", "\n\t\t".join([atlas.image for atlas in atlases])
            print "\t Templates: "
            print "\t\t", "\n\t\t".join(template_files)
            print "\t Validation Images: "
            print "\t\t", "\n\t\t".join(validation_files)
            
            maget = ValidationMAGeT(atlases, template_files, validation_files)
            
            if test_mode: 
               maget.set_label_propagation_method(TestLabelPropogationStrategy)
               
            subjects_labels = maget.build_pipeline(self.pipeline, registrations_dir)
        
            # fuse labels!
            fusion_dir = fh.createSubDir(outputDir, "fusion")
            majority_vote_dir = fh.createSubDir(fusion_dir, "majority_vote_%i" %i)
            for validator in validators:
                image = validator.image
                expected_labels = validator.labels
                
                majority_vote_labels = majority_vote(image, subjects_labels[image], majority_vote_dir, self.pipeline)
                compare_similarity(image, expected_labels, majority_vote_labels, majority_vote_dir, self.pipeline)
            
                #xcorr_vote_labels = xcorr_vote(image, maget.templates, xcorr_dir, self.pipeline, registrations_dir)
                #compare_similarity(image, expected_labels, xcorr_vote_labels, xcorr_dir, self.pipeline)
            
            #     
if __name__ == "__main__":
    application = MAGeTSubSamplingCrossValidationApp()
    application.start()
    