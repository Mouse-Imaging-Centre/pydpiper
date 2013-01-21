#!/usr/bin/env python

from optparse import OptionGroup

def addMBMGroup(parser):
    group = OptionGroup(parser, "MBM options", 
                        "Options for MICe-build-model and related programs.")
    group.add_option("--pipeline-name", dest="pipeline_name",
                    type="string", default=None,
                    help="Name of pipeline and prefix for models.")
    parser.add_option_group(group)
    
def tmpLongitudinalOptionGroup(parser):
    group = OptionGroup(parser, "Temporary options for longitudinal registrations", 
                        "Currently we must run the original version of build model to get a common space for "
                        "longitudinal registrations. Ultimately, we will do the necessary alignments using pydpiper modules.")
    group.add_option("--MBM-directory", dest="mbm_dir",
                      type="string", default=None, 
                      help="_processed directory from MBM used to average specified time point.")
    group.add_option("--nlin-average", dest="nlin_avg",
                      type="string", default=None, 
                      help="Final nlin average from MBM run.")
    parser.add_option_group(group)