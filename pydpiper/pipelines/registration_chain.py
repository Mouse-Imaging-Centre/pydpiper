#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import
import csv
from collections import defaultdict
from atom.api import Atom, Int, Str, Dict, Enum, Instance
from pydpiper.minc.analysis import determinants_at_fwhms, invert
from pydpiper.core.containers import Result
from pydpiper.minc.registration import (Stages, mincANTS_NLIN_build_model, mincANTS_default_conf,
                                        MincANTSConf, mincANTS, intrasubject_registrations, mincaverage, concat)
from pydpiper.minc.files import MincAtom
from pydpiper.execution.application import execute
#from pydpiper.pipelines.LSQ6 import lsq6
from configargparse import ArgParser
from pydpiper.core.arguments import addApplicationArgumentGroup, \
    addGeneralRegistrationArgumentGroup, addExecutorArgumentGroup, \
    addRegistrationChainArgumentGroup, addStatsArgumentGroup

import os

# TODO (general for all option records, not just for the registration chain):
# namedtuples are better than Argparse Namespaces for specification 
# (more descriptive name -- other benefits?) and
# for being able to get the data back out again (used in PP for filtering & re-sending
# arguments via submit scripts, which is the wrong approach, but ...),
# but they don't have the built-in type/arity checking or monoid-ish operations
# of Argparse, so we should create our own type which can be sent to both ??

# ALSO it's unclear whether our attempt at 'nominal' typing is a good idea, since, e.g.,
# a chain calls LSQ6, so will have to have lsq6 options as a subset of its own options
# ... so we could have a nested record, but this might not be best if both the chain
# and lsq6 need a particular option ... (also executor options, ...) ??

class ChainConf(Atom):
    input_space            = Enum('native', 'lsq6', 'lsq12')
    common_time_point      = Instance(type(None),int )
    # could make this a Fraction or a Decimal to represent, e.g., day 18.5, etc.
    # (float would be dangerous since we want to use these values as indices, etc.)
    common_time_point_name = Str("common")
    csv_file               = Instance(type(None), str)
    # perhaps the following belongs in a different class...
    stats_kernels          = Str("0.5,0.2,0.1") 

class Subject(Atom):
    """
    A Subject contains the intersubject_registration_time_pt and a dictionary
    that maps timepoints to scans/data related to this Subject. (these can be
    stored for instance as string, FileAtoms/MincAtoms or XfmHandler) 
    """
    intersubject_registration_time_pt = Instance(int)
    time_pt_dict   = Dict()    # validation (key=Int, value=Str) doesn't work? ...

    def __eq__(self, other):
        return (self is other or
                (self.intersubject_registration_time_pt == other.intersubject_registration_time_pt
                 and self.time_pt_dict == other.time_pt_dict
                 and self.__class__ == other.__class__))
    # ugh; also, should this be type(self) == ... ?

    def get_intersubject_registration_image(self):
        return self.time_pt_dict[self.intersubject_registration_time_pt]

    intersubject_registration_image = property(get_intersubject_registration_image,
                                               'intersubject_registration_image property')
    
    def __repr__(self):
        return "Subject(inter_sub_time_pt: %s, time_pt_dict keys: %s ... (values not shown))" % (self.intersubject_registration_time_pt,
                                                                 self.time_pt_dict.keys())
    
class TimePointError(Exception):
    pass
                 
def map_over_time_pt_dict_in_Subject(f, d):
    """Map `f` non-destructively (if `f` is) over (the values of)
    the inner time_pt_dict of a { subject : Subject }
    
    >>> (map_over_time_pt_dict_in_Subject(lambda x: x[3],
    ...          { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict={3:'s1_3.mnc', 4:'s1_4.mnc'}),
    ...            's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict={4:'s2_4.mnc', 5:'s2_5.mnc'})} )
    ...   == { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {3:'3',4:'4'}),
    ...        's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {4:'4',5:'5'}) })
    True
    """
    new_d = {}
    for s_id, subj in d.iteritems():
        new_time_pt_dict = {}
        for t,x in subj.time_pt_dict.iteritems():
            new_time_pt_dict[t] = f(x)
        new_subj = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                           time_pt_dict   = new_time_pt_dict)
        new_d[s_id] = new_subj
    return new_d

def parse_common(string):
    truthy_strings = ['1','True','true','T','t']
    falsy_strings  = ['','0','False','false','F','f']
    def fmt(strs):
        return "'" + "','".join(strs) + "'"
    string = string.strip()
    if string in truthy_strings:
        return True
    elif string in falsy_strings:
        return False
    else:
        raise ValueError('Unrecognized value %s; ' % string
                         + 'Please use one of ' + fmt(truthy_strings)
                         + ' in the "is_common" field of your csv file ' 
                         + 'to use this file for intersubject registration, or '
                         + 'one of ' + fmt(falsy_strings) + 'to specify otherwise')

# TODO standardize on pt/point
# TODO write some longer (non-doc)tests
def parse_csv(rows, common_time_pt): # row iterator, int -> { subject_id(str) : Subject }
    """
    Read subject information from a csv file containing at least the columns
    'subject_id', 'timepoint', and 'filename', and optionally a 'bitfield' column
    'is_common' containing one 1 per subject and 0 or empty fields for the other scans.
    
    Return a dictionary from subject IDs to `Subject`s.

    >>> csv_data = "subject_id,timepoint,filename,genotype\\ns1,1,s1_1.mnc,1\\n".split('\\n')
    >>> (parse_csv(csv_data, common_time_pt=1)
    ...   == { 's1' : Subject(intersubject_registration_time_pt=1, time_pt_dict={ 1 : 's1_1.mnc'})})
    True
    """
    subject_info = defaultdict(Subject)
    # Populate the subject -> Subject dictionary from the rows"""
    for row in csv.DictReader(rows):
        try:
            subj_id   = row['subject_id']
            timepoint = int(row['timepoint'])
            filename  = row['filename']
        except KeyError as e:
            raise KeyError("csv file must contain at least "
                           "'subject_id', 'timepoint', 'filename' fields; "
                           "missing: %s" % e.message)
        else:
            subject_info[subj_id].time_pt_dict[timepoint] = filename
            if parse_common(row.get('is_common', '')):
                if subject_info[subj_id].intersubject_registration_time_pt is not None:
                    raise TimePointError(
                        "multiple common time points specified for subject '%s'"
                        % subj_id)
                else:
                    subject_info[subj_id].intersubject_registration_time_pt = timepoint
    # could make this part into a separate fn that copies input, returns updated version:
    # Iterate through subjects, filling in intersubject-registration time points with the common
    # time point if unspecified for a given subject, and raising an error if there's no timepoint
    # available or no scan for the specified timepoint
    for s_id, s in subject_info.iteritems():
        if s.intersubject_registration_time_pt is None:
            if common_time_pt is None:
                raise TimePointError("no subject-specific or default inter-subject "
                                     "time point provided for subject '%s'" % s_id)
            elif common_time_pt in s.time_pt_dict.keys():
                s.intersubject_registration_time_pt = common_time_pt
            elif common_time_pt == -1 or common_time_pt == "-1":
                s.intersubject_registration_time_pt = max(s.time_pt_dict.keys())
            else:
                raise TimePointError("subject '%s' didn't have a scan for "
                                     "the common time point specified (%s); "
                                     "fix this or specify a different timepoint "
                                     "for this subject by putting a value in an "
                                     "'is_common' column of your table"
                                     % (s_id, str(common_time_pt)))
        else:
            if common_time_pt != s.intersubject_registration_time_pt:
                print('note: overriding common_time_pt %d with time point %d for subject %s'
                      % (common_time_pt, s.intersubject_registration_time_pt, s_id))
                    
    return subject_info


# NOTE I've moved the optional lsq6 stuff outside this function to promote re-use;
# actual call could look something like this:
#def chain_with_optional_lsq6(inputs, options):
#    def native():
#        pass
#    def lsq6():
#        pass
#
#    fns = { 'native' : native, 'lsq6' : lsq6 }
#
#    try:
#        f = fns[options.input_space]
#    except KeyError:
#        raise ValueError("illegal input space: %s; allowed options: %s" % \
#                         (options.input_space, ','.join(map(str,fns.keys()))))
#    # call f...

def final_transforms(pipeline_subject_info, intersubj_xfms_dict, chain_xfms_dict):
    """
    This function takes a subject mapping (with timepoints to MincAtoms) and returns a
    subject mapping of timepoints to `XfmHandler`s. Those transformations for
    each subject will contain the non-rigid transformation to the common time point average
    
    chain_xfms_dict maps subject_ids to a tuple containing a list of tuples (time_point, transformation) 
    and the index to the common time point in that list
    """
    s = Stages()
    new_d = {}
    for s_id, subj in pipeline_subject_info.iteritems():
        # this time point dictionary will hold a mapping
        # of time points to transformations to the final
        # common average (and it will have the inter_sub_time_pt)
        new_time_pt_dict = {}
        # the transformation for the common time point is easy:
        new_time_pt_dict[subj.intersubject_registration_time_pt] = \
            intersubj_xfms_dict[subj.intersubject_registration_image]        
        chain_transforms, index_of_common_time_pt = chain_xfms_dict[s_id]
        
        # will hold the XfmHandler from current to average of common time pt
        current_xfm_to_common = intersubj_xfms_dict[subj.intersubject_registration_image]
        # we start at the common time point and are going forward at this point
        # so we will assign the concatenated transform to the target of each 
        # transform we are adding 
        for time_pt, transform in chain_transforms[index_of_common_time_pt:]:
            current_xfm_to_common = s.defer(concat([s.defer(invert(transform)),current_xfm_to_common], name="%s%s_to_common" % (s_id, time_pt))) # TODO: naming
            new_time_pt_dict[time_pt] = current_xfm_to_common
        # we need to do something similar moving backwards: make sure to reset
        # the current_xfm_to_common here!
        current_xfm_to_common = intersubj_xfms_dict[subj.intersubject_registration_image]
        for time_pt, transform in chain_transforms[index_of_common_time_pt-1::-1]:
            current_xfm_to_common = s.defer(concat([transform,current_xfm_to_common], name="%s%s_to_common" % (s_id, time_pt)))
            new_time_pt_dict[time_pt] = current_xfm_to_common
        
        new_subj = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                           time_pt_dict   = new_time_pt_dict)
        new_d[s_id] = new_subj
    return Result(stages=s, output=new_d)

def chain(options):

    # TODO:
    # one overall question for this entire piece of code is how
    # we are going to make sure that we can concatenate/add all
    # the transformations together. Many of the sub-registrations
    # that are performed (inter-subject registration, lsq6 using
    # multiple initial models) are applied to subsets of the entire 
    # data, making it harder to keep the mapping simple/straightforward

    s = Stages()
    
    with open(options.csv_file, 'r') as f:
        subject_info = parse_csv(f, options.common_time_point)
    
    pipeline_processed_dir = os.path.join(options.output_directory, options.pipeline_name + "_processed")
    pipeline_lsq12_common_dir = os.path.join(options.output_directory, options.pipeline_name + "_lsq12_" + options.common_name)
    pipeline_nlin_common_dir = os.path.join(options.output_directory, options.pipeline_name + "_nlin_" + options.common_name)
    
    pipeline_subject_info = map_over_time_pt_dict_in_Subject(
                                     lambda subj_str:  MincAtom(name=subj_str, pipeline_sub_dir=pipeline_processed_dir),
                                     subject_info)
    
    if options.input_space not in ChainConf.input_space.items:
        raise ValueError('unrecognized input space: %s; choices: %s' % (options.input_space, ChainConf.input_space.items))
    
    if options.input_space == 'native':
        raise NotImplementedError("We currently have not implemented the code for 'input space': %s" % options.input_space)
        
        # TODO:
        # what should be done here is the LSQ6 registration, options:
        # - bootstrap
        # - lsq6-target
        # - initial model
        # - "pride" of initial models (different targets for different time points)
        
        
    some_temp_target = None

    # NB currently LSQ6 expects an array of files, but we have a map.
    # possibilities:
    # - note that pairwise is enough (except for efficiency -- redundant blurring, etc.)
    #   and just use the map fn above with an LSQ6 fn taking only a single source
    # - rewrite LSQ6 to use such a (nested) map
    # - write conversion which creates a tagged array from the map, performs LSQ6,
    #   and converts back
    # - write 'over' which takes a registration, a data structure, and 'get/set' fns ...?
    

    # Intersubject registration: LSQ12/NLIN registration of common-timepoint images
    # The assumption here is that all these files are roughly aligned. Here is a toy
    # schematic of what happens. In this example, the common timepoint is set timepoint 2: 
    #
    #                            ------------
    # subject A    A_time_1   -> | A_time_2 | ->   A_time_3
    # subject B    B_time_1   -> | B_time_2 | ->   B_time_3
    # subject C    C_time_1   -> | C_time_2 | ->   C_time_3
    #                            ------------
    #                                 |
    #                            group_wise registration on time point 2
    #
    dict_intersubj_atom_to_xfm = {}
    if options.input_space == 'lsq6' or options.input_space == 'lsq12':
        intersubj_imgs = { s_id : subj.intersubject_registration_image
                          for s_id, subj in pipeline_subject_info.iteritems() }
    else:
        # TODO: the intersubj_imgs should now be extracted from the
        # lsq6 aligned images
        raise NotImplementedError("We currently have not implemented the code for 'input space': %s" % options.input_space)
    
    if options.verbose:
        print("\nImages that are used for the inter-subject registration:")
        print("ID\tsubject")
        for subject in intersubj_imgs:
            print(subject + '\t' + intersubj_imgs[subject].orig_path)  
    
    # input files that started off in native space have been aligned rigidly 
    # by this point in the code (i.e., lsq6)
    if options.input_space == 'lsq6' or options.input_space == 'native':
        raise NotImplementedError("We currently have not implemented the code for 'input space': %s" % options.input_space)
    #intersubj_xfms = lsq12_NLIN_build_model_on_dictionaries(imgs=intersubj_imgs,
        #                                                        conf=conf,
        #                                                        lsq12_dir=lsq12_directory
                                                                #, like={atlas_from_init_model_at_this_tp}
        #                                                        )
    elif options.input_space == 'lsq12':
        #TODO: write reader that creates a mincANTS configuration out of an input protocol 
        if not some_temp_target:
            some_temp_target = s.defer(mincaverage(imgs=intersubj_imgs.values(),
                                           name_wo_ext="avg_of_input_files",
                                           output_dir=pipeline_nlin_common_dir))
        test_conf = mincANTS_default_conf
        first_level = MincANTSConf(iterations="100x100x100x0")
        full_hierarchy = [first_level, test_conf]
        intersubj_xfms = s.defer(mincANTS_NLIN_build_model(imgs=intersubj_imgs.values(),
                                                   initial_target=some_temp_target, # this doesn't make sense yet
                                                   nlin_dir=pipeline_nlin_common_dir,
                                                   confs=full_hierarchy))
        dict_intersubj_atom_to_xfm = {xfm.source: xfm for xfm in intersubj_xfms.xfms}

    ## within-subject registration
    # In the toy scenario below: 
    # subject A    A_time_1   ->   A_time_2   ->   A_time_3
    # subject B    B_time_1   ->   B_time_2   ->   B_time_3
    # subject C    C_time_1   ->   C_time_2   ->   C_time_3
    # 
    # The following registrations are run:
    # 1) A_time_1   ->   A_time_2
    # 2) A_time_2   ->   A_time_3
    #
    # 3) B_time_1   ->   B_time_2
    # 4) B_time_2   ->   B_time_3
    #
    # 5) C_time_1   ->   C_time_2
    # 6) C_time_2   ->   C_time_3    
    
    chain_xfms = { s_id : s.defer(intrasubject_registrations(subj, MincANTSConf()))
                   for s_id, subj in pipeline_subject_info.iteritems() }

    # create transformation from each subject to the final common time point average
    final_non_rigid_xfms = s.defer(final_transforms(pipeline_subject_info,
                                                          dict_intersubj_atom_to_xfm,
                                                          chain_xfms))

    subject_determinants = map_over_time_pt_dict_in_Subject(
        lambda xfm: s.defer(determinants_at_fwhms(xfm=s.defer(invert(xfm)),
                                                  inv_xfm=xfm,
                                                  blur_fwhms=options.stats_kernels)),
        final_non_rigid_xfms)
    
    return Result(stages=s, output=(final_non_rigid_xfms, subject_determinants))



if __name__ == "__main__":
    # TODO: the following can be captured in some sort of initialization 
    # function to make it easier to write/create a new application
    
    # *** *** *** *** *** *** *** *** ***
    # use an environment variable to look for a default config file
    default_config_file = os.getenv("PYDPIPER_CONFIG_FILE")
    if default_config_file is not None:
        config_files = [default_config_file]
    else:
        config_files = []
    parser = ArgParser(default_config_files=config_files)

    addExecutorArgumentGroup(parser)
    addApplicationArgumentGroup(parser)
    addGeneralRegistrationArgumentGroup(parser)
    addRegistrationChainArgumentGroup(parser)
    addStatsArgumentGroup(parser)
    
    options = parser.parse_args()    
    # *** *** *** *** *** *** *** *** ***

    chain_stages, _ = chain(options)

    #[print(s.render() + '\n') for s in chain_stages]

    execute(chain_stages, options)
