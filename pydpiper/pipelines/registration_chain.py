#!/usr/bin/env python3

import csv
from collections import defaultdict
import os
import sys

from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple, TypeVar

from pydpiper.minc.analysis import determinants_at_fwhms, invert
from pydpiper.core.stages import Result
from pydpiper.minc.registration import (Subject, Stages, mincANTS_NLIN_build_model, mincANTS_default_conf,
                                        MincANTSConf, mincANTS, intrasubject_registrations, mincaverage, 
                                        concat, check_MINC_input_files, get_registration_targets,
                                        lsq6_nuc_inorm, get_resolution_from_file, XfmHandler)
from pydpiper.minc.files import MincAtom
from pydpiper.execution.application import execute  # type: ignore
#from pydpiper.pipelines.LSQ6 import lsq6
#from configargparse import ArgParser
from pydpiper.core.arguments import (application_parser,
                                     chain_parser,
                                     execution_parser,
                                     lsq6_parser, lsq12_parser,
                                     registration_parser,
                                     stats_parser,
                                     parse,
                                     AnnotatedParser, BaseParser, CompoundParser,
                                     RegistrationConf,
                                     InputSpace, LSQ6Method)


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

# some data structures used in this pipeline:

class ChainConf(object):
    def __init__(self, common_time_point : int,
                 csv_file                : str,
                 pride_of_models         : str,
                 common_time_point_name  : str = "common") -> None:
        # could make this a Fraction or a Decimal to represent, e.g., day 18.5, etc.
        # (float would be dangerous since we want to use these values as indices, etc.)
        self.common_time_point = common_time_point
        self.csv_file = csv_file
        self.common_time_point_name = common_time_point_name
        self.pride_of_models = pride_of_models


    
class TimePointError(Exception):
    pass

def chain(options):
    """Create a registration chain pipeline from the given options."""

    # TODO:
    # one overall question for this entire piece of code is how
    # we are going to make sure that we can concatenate/add all
    # the transformations together. Many of the sub-registrations
    # that are performed (inter-subject registration, lsq6 using
    # multiple initial models) are applied to subsets of the entire 
    # data, making it harder to keep the mapping simple/straightforward

    chain_opts = options.chain  # type : ChainConf

    s = Stages()
    
    with open(options.chain.csv_file, 'r') as f:
        subject_info = parse_csv(rows=f, common_time_pt=options.chain.common_time_point)

    output_dir    = options.application.pipeline_name
    pipeline_name = options.application.pipeline_name

    pipeline_processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    pipeline_lsq12_common_dir = os.path.join(output_dir, pipeline_name + "_lsq12_" + options.chain.common_time_point_name)
    pipeline_nlin_common_dir = os.path.join(output_dir, pipeline_name + "_nlin_" + options.chain.common_time_point_name)
    
    
    pipeline_subject_info = map_over_time_pt_dict_in_Subject(
                                     lambda subj_str:  MincAtom(name=subj_str, pipeline_sub_dir=pipeline_processed_dir),
                                     subject_info)  # type: Dict[str, Subject[MincAtom]]
    
    # verify that in input files are proper MINC files, and that there 
    # are no duplicates in the filenames
    all_Minc_atoms = []  # type: List[MincAtom]
    for s_id, subj in pipeline_subject_info.items():
        for subj_time_pt, subj_filename in subj.time_pt_dict.items():
            all_Minc_atoms.append(subj_filename)
    # check_MINC_input_files takes strings, so pass along those instead of the actual MincAtoms
    check_MINC_input_files([minc_atom.path for minc_atom in all_Minc_atoms])
    
    if options.registration.input_space not in InputSpace.__members__.values():
        raise ValueError('unrecognized input space: %s; choices: %s' %
                         (options.registration.input_space, ','.join(InputSpace.__members__)))
    
    if options.registration.input_space == InputSpace.native:
        if options.lsq6.bootstrap:
            raise ValueError("\nA bootstrap model is ill-defined for the registration chain. "
                             "(Which file is the 'first' input file?). Please use the --lsq6-target "
                             "flag to specify a target for the lsq6 stage, or use an initial model.")
        if options.chain.pride_of_models:
            raise NotImplementedError(
                "We currently have not implemented the code that handles the pride of initial models...")
        else:
            # if we are not dealing with a pride of models, we can retrieve a fixed
            # registration target for all input files:
            registration_targets = get_registration_targets(init_model=options.lsq6.init_model,
                                                            lsq6_target=options.lsq6.lsq6_target,
                                                            bootstrap=options.lsq6.bootstrap,
                                                            output_dir=output_dir,
                                                            pipeline_name=pipeline_name)
            
            # we want to store the xfm handlers in the same shape as pipeline_subject_info,
            # as such we will call lsq6_nuc_inorm for each file individually. This returns
            # a list of XfmHandlers, however, given that we pass the MincAtoms one by one,
            # we can simply extract the first (and only) element from the list.
            # s.defer(...)[0]
            xfm_handlers_dict_lsq6 = map_over_time_pt_dict_in_Subject(
                                         lambda subj_atom:  s.defer(lsq6_nuc_inorm([subj_atom],
                                                                                   registration_targets,
                                                                                   options.registration.resolution,
                                                                                   options.lsq6,
                                                                                   subject_matter=options.registration.subject_matter))[0],
                                         pipeline_subject_info)  # type: Dict[str, Subject[XfmHandler]]
        
        
    some_temp_target = None # type: MincAtom  # FIXME just set this right away

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
    dict_intersubj_atom_to_xfm = {}  # type: Dict[MincAtom, XfmHandler]
    if options.registration.input_space in (InputSpace.lsq6, InputSpace.lsq12):
        intersubj_imgs = { s_id : subj.intersubject_registration_image
                          for s_id, subj in pipeline_subject_info.items() }
    else:
        # TODO: the intersubj_imgs should now be extracted from the
        # lsq6 aligned images
        raise NotImplementedError("We currently have not implemented the code for 'input space': %s" % options.registration.input_space)
    
    if options.application.verbose:
        print("\nImages that are used for the inter-subject registration:")
        print("ID\timage")
        for subject in intersubj_imgs:
            print(subject + '\t' + intersubj_imgs[subject].orig_path)  
    
    # input files that started off in native space have been aligned rigidly 
    # by this point in the code (i.e., lsq6)
    if options.registration.input_space in [InputSpace.lsq6, InputSpace.native]:
        raise NotImplementedError("We currently have not implemented the code for 'input space': %s" %
                                  options.registration.input_space)
    #intersubj_xfms = lsq12_NLIN_build_model_on_dictionaries(imgs=intersubj_imgs,
        #                                                        conf=conf,
        #                                                        lsq12_dir=lsq12_directory
                                                                #, like={atlas_from_init_model_at_this_tp}
        #                                                        )
    elif options.registration.input_space == InputSpace.lsq12:
        #TODO: write reader that creates a mincANTS configuration out of an input protocol 
        if not some_temp_target:
            some_temp_target = s.defer(mincaverage(imgs=list(intersubj_imgs.values()),
                                           name_wo_ext="avg_of_input_files",
                                           output_dir=pipeline_nlin_common_dir))
        conf1 = mincANTS_default_conf.replace(default_resolution=options.registration.resolution,
                                              iterations="100x100x100x0")
        conf2 = mincANTS_default_conf.replace(default_resolution=options.registration.resolution)
        full_hierarchy = [conf1, conf2]
        intersubj_xfms = s.defer(mincANTS_NLIN_build_model(imgs=list(intersubj_imgs.values()),
                                                   initial_target=some_temp_target, # this doesn't make sense yet
                                                   nlin_dir=pipeline_nlin_common_dir,
                                                   confs=full_hierarchy))
        dict_intersubj_atom_to_xfm = { xfm.source : xfm for xfm in intersubj_xfms.output }

    #return Result(stages=s, output=())
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
    
    chain_xfms = { s_id : s.defer(intrasubject_registrations(
                                    subj,
                                    mincANTS_default_conf._replace(default_resolution=options.registration.resolution)))
                   for s_id, subj in pipeline_subject_info.items() }

    # create transformation from each subject to the final common time point average
    final_non_rigid_xfms = s.defer(final_transforms(pipeline_subject_info,
                                                          dict_intersubj_atom_to_xfm,
                                                          chain_xfms))

    subject_determinants = map_over_time_pt_dict_in_Subject(
        lambda xfm: s.defer(determinants_at_fwhms(xfm=s.defer(invert(xfm)),
                                                  inv_xfm=xfm,
                                                  blur_fwhms=options.stats.stats_kernels)),
        final_non_rigid_xfms)
    
    return Result(stages=s, output=(final_non_rigid_xfms, subject_determinants))
                 
K = TypeVar('K')
T = TypeVar('T')
U = TypeVar('U')

def map_over_time_pt_dict_in_Subject(f : Callable[[T], U],
                                     d : Dict[K, Subject[T]]) -> Dict[K, Subject[U]]:
    """Map `f` non-destructively (if `f` is) over (the values of)
    the inner time_pt_dict of a { subject : Subject }
    
    >>> (map_over_time_pt_dict_in_Subject(lambda x: x[3],
    ...          { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict={3:'s1_3.mnc', 4:'s1_4.mnc'}),
    ...            's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict={4:'s2_4.mnc', 5:'s2_5.mnc'})} )
    ...   == { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {3:'3',4:'4'}),
    ...        's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {4:'4',5:'5'}) })
    True
    """
    new_d = {}  # type: Dict[K, Subject[T]]
    for s_id, subj in d.items():
        new_time_pt_dict = {}  # type: Dict[int, U]
        for t, x in subj.time_pt_dict.items():
            new_time_pt_dict[t] = f(x)
        new_subj = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                           time_pt_dict = new_time_pt_dict)  # type: Subject[U]
        new_d[s_id] = new_subj
    return new_d # type: Dict[K, Subject[U]]

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
    'is_common' containing up to one '1' per subject and '0' or empty fields for the other scans.
    
    Return a dictionary from subject IDs to `Subject`s.

    >>> csv_data = "subject_id,timepoint,filename,genotype\\ns1,1,s1_1.mnc,1\\n".split('\\n')
    >>> (parse_csv(csv_data, common_time_pt=1)
    ...   == { 's1' : Subject(intersubject_registration_time_pt=1, time_pt_dict={ 1 : 's1_1.mnc'})})
    True
    """
    subject_info = defaultdict(lambda: Subject(intersubject_registration_time_pt=None))
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
    for s_id, s in subject_info.items():
        if s.intersubject_registration_time_pt is None:
            if common_time_pt is None:
                raise TimePointError("no subject-specific or default inter-subject "
                                     "time point provided for subject '%s' " 
                                     "You can use either the --common-time-point flag "
                                     "to specify which time point you want to use, or "
                                     "have a column in your csv called \'is_common\' "
                                     "indicating with 0s and 1s which files to use "
                                     "for the common time point." % s_id)
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
    for s_id, subj in pipeline_subject_info.items():
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


if __name__ == "__main__":

    p = CompoundParser(
          [AnnotatedParser(parser=execution_parser, namespace='execution'),
           AnnotatedParser(parser=application_parser, namespace='application'),
           AnnotatedParser(parser=registration_parser, namespace='registration', cast=RegistrationConf),
           AnnotatedParser(parser=chain_parser, namespace='chain'), # cast=ChainConf),
           AnnotatedParser(parser=lsq6_parser, namespace='lsq6'),
           AnnotatedParser(parser=lsq12_parser, namespace='lsq12'), # should be MBM or build_model ...
           #AnnotatedParser(parser=BaseParser(addLSQ12ArgumentGroup), namespace='lsq12-inter-subj'),
           #addNLINArgumentGroup,
           AnnotatedParser(parser=stats_parser, namespace='stats')])
    
    # TODO could abstract and then parametrize by prefix/ns ??
    options = parse(p, sys.argv[1:])
    
    # TODO: the registration resolution should be set somewhat outside
    # of any actual function? Maybe the right time to set this, is here
    # when options are gathered? 
    if not options.registration.resolution:
        
        # TODO: enforce existence of resolution as early as possible, e.g.,
        # by creating a mutually exclusive options group
        
        if options.lsq6.init_model:
            # Ha! an initial model always points to the file in standard space, so we 
            # can directly use this file to get the resolution from
            options.registration.resolution = get_resolution_from_file(options.lsq6.init_model)
        if options.lsq6.bootstrap:
            options.registration.resolution = get_resolution_from_file(options.application.files[0])
        if options.lsq6.lsq6_target:
            options.registration.resolution = get_resolution_from_file(options.lsq6.lsq6_target)
            
    if not options.registration.resolution:
        raise ValueError("Crap... couldn't get the registration resolution...")
    
    print("Ha! The registration resolution is: %s\n" % options.registration.resolution)
    # *** *** *** *** *** *** *** *** ***

    chain_stages = chain(options).stages

    execute(chain_stages, options)
