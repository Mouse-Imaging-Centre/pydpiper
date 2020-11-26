#!/usr/bin/env python3

import csv
from argparse import Namespace
from collections import defaultdict
import os
import sys
import bisect
import re

from typing import Callable, Dict, List, TypeVar, Iterator, Generic, Optional, Tuple, Type

from pydpiper.core.util import pairs
from pydpiper.minc.analysis import determinants_at_fwhms, invert_xfmhandler
from pydpiper.core.stages import Result
from pydpiper.minc.nlin import NLIN
from pydpiper.minc.registration import (Stages,
                                        mincaverage,
                                        concat_xfmhandlers, ensure_distinct_basenames, registration_targets,
                                        lsq6_nuc_inorm, get_resolution_from_file, XfmHandler,
                                        InputSpace, lsq12_nlin_build_model, TargetType,
                                        MinctraccConf,
                                        create_quality_control_images,
                                        check_MINC_files_have_equal_dimensions_and_resolution,
                                        default_lsq12_multilevel_minctracc, get_pride_of_models_mapping,
                                        mincresample, lsq12_nlin, get_nonlinear_component)
from pydpiper.minc.files import MincAtom
from pydpiper.execution.application import execute  # type: ignore
from pydpiper.core.arguments import (application_parser,
                                     execution_parser,
                                     lsq6_parser, lsq12_parser,
                                     registration_parser,
                                     stats_parser,
                                     parse,
                                     AnnotatedParser, CompoundParser, nlin_parser,
                                     _chain_parser)
from pydpiper.minc.registration_strategies import get_model_building_procedure


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


V = TypeVar('V')


class Subject(Generic[V]):
    """
    A Subject contains the intersubject_registration_time_pt and a dictionary
    that maps timepoints to scans/data of type `V` related to this Subject.
    (Here V could be - for instance - str, FileAtom/MincAtom or XfmHandler).
    """

    def __init__(self,
                 intersubject_registration_time_pt: int,
                 time_pt_dict: Optional[Dict[int, V]] = None) -> None:
        # TODO: change the time_pt datatype to decimal or rational to allow, e.g., 18.5?
        self.intersubject_registration_time_pt = intersubject_registration_time_pt  # type: int
        self.time_pt_dict = time_pt_dict or dict()  # type: Dict[int, V]

    # compare by fields, not pointer
    def __eq__(self, other) -> bool:
        return (self is other or
                (self.__class__ == other.__class__
                 and self.intersubject_registration_time_pt == other.intersubject_registration_time_pt
                 and self.time_pt_dict == other.time_pt_dict))

    # ugh; also, should this be type(self) == ... ?

    # TODO: change name? This might not be an 'image'
    @property
    def intersubject_registration_image(self) -> V:
        return self.time_pt_dict[self.intersubject_registration_time_pt]

    def __repr__(self) -> str:
        return ("Subject(intersubject_registration_time_pt: %s, time_pt_dict keys: %s ... (values not shown))"
                % (self.intersubject_registration_time_pt, self.time_pt_dict.keys()))


# seems specific enough (at least in its current incarnation) to the registration chain that it lives in this file:
def intrasubject_registrations(subj: Subject,
                               linear_conf: MinctraccConf,
                               nlin_module: Type[NLIN],
                               nlin_options, #: nlin_module.Conf,
                               resolution: float) \
        -> Result[Tuple[List[Tuple[int, int, XfmHandler]], int]]:
    """

    subj -- Subject (has a intersubject_registration_time_pt and a time_pt_dict
            that maps timepoints to individual subjects

    Return:
    ([ (source_time_pt, target_time_pt, XfmHandler),
       (...,...,...),(...,...,...)],
     index_of_common_time_pt)
     note this is one element smaller than the number of time points.
    """
    # TODO: somehow the output of this function should provide us with
    # easy access from a MincAtom to an XfmHandler from time_pt N to N+1
    # either here or in the chain() function

    s = Stages()
    timepts = sorted(subj.time_pt_dict.items())  # type: List[Tuple[int, MincAtom]]
    timepts_indices = [index for index, _subj_atom in timepts]  # type: List[int]
    # we need to find the index of the common time point and for that we
    # should only look at the first element of the tuples stored in timepts
    index_of_common_time_pt = timepts_indices.index(subj.intersubject_registration_time_pt)  # type: int

    time_pt_to_xfms = [(timepts_indices[source_index],
                        timepts_indices[source_index + 1],
                        s.defer(lsq12_nlin(source=src[1],
                                           target=dest[1],
                                           lsq12_conf=linear_conf,
                                           nlin_module=nlin_module,
                                           nlin_options=nlin_options,
                                           resolution=resolution,
                                           #nlin_conf=nlin_conf,
                                           resample_source=True)))
                       for source_index, (src, dest) in enumerate(pairs(timepts))]
    return Result(stages=s, output=(time_pt_to_xfms, index_of_common_time_pt))


def chain(options):
    """Create a registration chain pipeline from the given options."""

    # TODO:
    # one overall question for this entire piece of code is how
    # we are going to make sure that we can concatenate/add all
    # the transformations together. Many of the sub-registrations
    # that are performed (inter-subject registration, lsq6 using
    # multiple initial models) are applied to subsets of the entire 
    # data, making it harder to keep the mapping simple/straightforward


    #chain_opts = options.chain  # type : ChainConf

    s = Stages()
    
    with open(options.chain.csv_file, 'r') as f:
        subject_info = parse_csv(rows=f, common_time_pt=options.chain.common_time_point)

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    resolution = options.registration.resolution

    pipeline_processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    pipeline_lsq12_common_dir = os.path.join(output_dir, pipeline_name + "_lsq12_" + options.chain.common_time_point_name)
    pipeline_nlin_common_dir = os.path.join(output_dir, pipeline_name + "_nlin_" + options.chain.common_time_point_name)
    pipeline_montage_dir = os.path.join(output_dir, pipeline_name + "_montage")
    
    
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
    ensure_distinct_basenames([minc_atom.path for minc_atom in all_Minc_atoms])

    if options.registration.input_space == InputSpace.lsq6 or \
        options.registration.input_space == InputSpace.lsq12:
        # the input files are not going through the lsq6 alignment. This is the place
        # where they will all be resampled using a single like file, and get the same
        # image dimensions/lengths/resolution. So in order for the subsequent stages to
        # finish (mincaverage stages for instance), all files need to have the same
        # image parameters:
        check_MINC_files_have_equal_dimensions_and_resolution([minc_atom.path for minc_atom in all_Minc_atoms],
                                                              additional_msg="Given that the input images are "
                                                                             "already in " + str(options.registration.input_space) +
                                                                             " space, all input files need to have "
                                                                             "the same dimensions/starts/step sizes.")

    if options.registration.input_space not in InputSpace.__members__.values():
        raise ValueError('unrecognized input space: %s; choices: %s' %
                         (options.registration.input_space, ','.join(InputSpace.__members__)))
    
    if options.registration.input_space == InputSpace.native:
        if options.lsq6.target_type == TargetType.bootstrap:
            raise ValueError("\nA bootstrap model is ill-defined for the registration chain. "
                             "(Which file is the 'first' input file?). Please use the --lsq6-target "
                             "flag to specify a target for the lsq6 stage, or use an initial model.")
        if options.lsq6.target_type == TargetType.pride_of_models:
            pride_of_models_dict = get_pride_of_models_mapping(pride_csv=options.lsq6.target_file,
                                                               output_dir=options.application.output_directory,
                                                               pipeline_name=options.application.pipeline_name)
            subj_id_to_subj_with_lsq6_xfm_dict = map_with_index_over_time_pt_dict_in_Subject(
                                    lambda subj_atom, time_point:
                                        s.defer(lsq6_nuc_inorm([subj_atom],
                                                               registration_targets=get_closest_model_from_pride_of_models(
                                                                                        pride_of_models_dict, time_point),
                                                               resolution=options.registration.resolution,
                                                               lsq6_options=options.lsq6,
                                                               lsq6_dir=None,  # never used since no average
                                                               # (could call this "average_dir" with None -> no avg ?)
                                                               subject_matter=options.registration.subject_matter,
                                                               create_qc_images=False,
                                                               create_average=False))[0],
                                        pipeline_subject_info)  # type: Dict[str, Subject[XfmHandler]]
        else:
            # if we are not dealing with a pride of models, we can retrieve a fixed
            # registration target for all input files:
            targets = s.defer(registration_targets(lsq6_conf=options.lsq6,
                                           app_conf=options.application, reg_conf=options.registration))
            
            # we want to store the xfm handlers in the same shape as pipeline_subject_info,
            # as such we will call lsq6_nuc_inorm for each file individually and simply extract
            # the first (and only) element from the resulting list via s.defer(...)[0].
            subj_id_to_subj_with_lsq6_xfm_dict = map_over_time_pt_dict_in_Subject(
                                         lambda subj_atom:
                                           s.defer(lsq6_nuc_inorm([subj_atom],
                                                                  registration_targets=targets,
                                                                  resolution=options.registration.resolution,
                                                                  lsq6_options=options.lsq6,
                                                                  # no average will be created, is just one file...
                                                                  lsq6_dir=None,
                                                                  create_qc_images=False,
                                                                  create_average=False,
                                                                  subject_matter=options.registration.subject_matter)
                                                   )[0],
                                         pipeline_subject_info)  # type: Dict[str, Subject[XfmHandler]]

        # create verification images to show the 6 parameter alignment
        montageLSQ6 = pipeline_montage_dir + "/quality_control_montage_lsq6"
        # TODO, base scaling factor on resolution of initial model or target
        filesToCreateImagesFrom = []
        for subj_id, subj in subj_id_to_subj_with_lsq6_xfm_dict.items():
            for time_pt, subj_time_pt_xfm in subj.time_pt_dict.items():
                filesToCreateImagesFrom.append(subj_time_pt_xfm.resampled)

        # TODO it's strange that create_quality_control_images gets the montage directory twice
        # TODO (in montages=output=montageLSQ6 and in montage_dir), suggesting a weirdness in create_q_c_images
        lsq6VerificationImages = s.defer(create_quality_control_images(filesToCreateImagesFrom,
                                                                       montage_output=montageLSQ6,
                                                                       montage_dir=pipeline_montage_dir,
                                                                       message=" the input images after the lsq6 alignment"))

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

    # dictionary that holds the transformations from the intersubject images
    # to the final common space average
    intersubj_img_to_xfm_to_common_avg_dict = {}  # type: Dict[MincAtom, XfmHandler]
    if options.registration.input_space in (InputSpace.lsq6, InputSpace.lsq12):
        # no registrations have been performed yet, so we can point to the input files
        s_id_to_intersubj_img_dict = { s_id : subj.intersubject_registration_image
                          for s_id, subj in pipeline_subject_info.items() }
    else:
        # lsq6 aligned images
        # When we ran the lsq6 alignment, we stored the XfmHandlers in the Subject dictionary. So when we call
        # xfmhandler.intersubject_registration_image, this returns an XfmHandler. From which
        # we want to extract the resampled file (in order to continue the registration with)
        s_id_to_intersubj_img_dict = { s_id : subj_with_xfmhandler.intersubject_registration_image.resampled
                          for s_id, subj_with_xfmhandler in subj_id_to_subj_with_lsq6_xfm_dict.items() }
    
    if options.application.verbose:
        print("\nImages that are used for the inter-subject registration:")
        print("ID\timage")
        for subject in s_id_to_intersubj_img_dict:
            print(subject + '\t' + s_id_to_intersubj_img_dict[subject].path)

    # determine what configuration to use for the non linear registration
    nlin_module = get_nonlinear_component(reg_method=options.nlin.reg_method)

    build_model_component = get_model_building_procedure(options.nlin.reg_strategy,
                                                         # was: model_building.reg_strategy
                                                         reg_module=nlin_module)

    nlin_conf = (build_model_component.parse_build_model_protocol(
                     options.nlin.nlin_protocol, resolution=resolution)
                 if options.nlin.nlin_protocol is not None
                 else build_model_component.get_default_build_model_conf(resolution=resolution))

    if options.registration.input_space in [InputSpace.lsq6, InputSpace.native]:
        intersubj_xfms = s.defer(lsq12_nlin_build_model(imgs=list(s_id_to_intersubj_img_dict.values()),
                                                lsq12_conf=options.lsq12,
                                                nlin_conf=nlin_conf,
                                                use_robust_averaging=options.nlin.use_robust_averaging,
                                                resolution=options.registration.resolution,
                                                lsq12_dir=pipeline_lsq12_common_dir,
                                                nlin_module=build_model_component,
                                                nlin_dir=pipeline_nlin_common_dir,
                                                nlin_prefix="common"))
                                                #, like={atlas_from_init_model_at_this_tp}
    elif options.registration.input_space == InputSpace.lsq12:
        #TODO: write reader that creates a ANTS configuration out of an input protocol
        # if we're starting with files that are already aligned with an affine transformation
        # (overall scaling is also dealt with), then the target for the non linear registration
        # should be the average of the current input files.
        first_nlin_target = s.defer(mincaverage(imgs=list(s_id_to_intersubj_img_dict.values()),
                                                name_wo_ext="avg_of_input_files",
                                                output_dir=pipeline_nlin_common_dir))
        intersubj_xfms = s.defer(build_model_component.build_model(
                                     imgs=list(s_id_to_intersubj_img_dict.values()),
                                     initial_target=first_nlin_target,
                                     nlin_dir=pipeline_nlin_common_dir,
                                     nlin_module=build_model_component,
                                     conf=nlin_conf))


    intersubj_img_to_xfm_to_common_avg_dict = { xfm.source : xfm for xfm in intersubj_xfms.output }

    # create one more convenience data structure: a mapping from subject_ID to the xfm_handler
    # that contains the transformation from the subject at the common time point to the
    # common time point average.
    subj_ID_to_xfm_handler_to_common_avg = {}
    for s_id, subj_at_common_tp in s_id_to_intersubj_img_dict.items():
        subj_ID_to_xfm_handler_to_common_avg[s_id] = intersubj_img_to_xfm_to_common_avg_dict[subj_at_common_tp]

    # create verification images to show the inter-subject  alignment
    montage_inter_subject = pipeline_montage_dir + "/quality_control_montage_inter_subject_registration.png"
    avg_and_inter_subject_images = []
    avg_and_inter_subject_images.append(intersubj_xfms.avg_img)
    for xfmh in intersubj_xfms.output:
        avg_and_inter_subject_images.append(xfmh.resampled)

    inter_subject_verification_images = s.defer(create_quality_control_images(
                                                  imgs=avg_and_inter_subject_images,
                                                  montage_output=montage_inter_subject,
                                                  montage_dir=pipeline_montage_dir,
                                                  message=" the result of the inter-subject alignment"))

    if options.application.verbose:
        print("\nTransformations for intersubject images to final nlin common space:")
        print("MincAtom\ttransformation")
        for subj_atom, xfm_handler in intersubj_img_to_xfm_to_common_avg_dict.items():
            print(subj_atom.path + '\t' + xfm_handler.xfm.path)


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

    subj_id_to_Subjec_for_within_dict = pipeline_subject_info
    if options.registration.input_space == InputSpace.native:
        # we started with input images that were not aligned whatsoever
        # in this case we should use the images that were rigidly
        # aligned files to continue the within-subject registration with
        # # type: Dict[str, Subject[XfmHandler]]
        subj_id_to_Subjec_for_within_dict = map_over_time_pt_dict_in_Subject(lambda x: x.resampled,
                                                                             subj_id_to_subj_with_lsq6_xfm_dict)

    if options.application.verbose:
        print("\n\nWithin subject registrations:")
        for s_id, subj in subj_id_to_Subjec_for_within_dict.items():
            print("ID: ", s_id)
            for time_pt, subj_img in subj.time_pt_dict.items():
                print(time_pt, " ", subj_img.path)
            print("\n")

    # dictionary that maps subject IDs to a list containing:
    # ( [(time_pt_n, time_pt_n+1, XfmHandler_from_n_to_n+1), ..., (,,,)],
    #   index_of_common_time_pt)
    chain_xfms = { s_id : s.defer(intrasubject_registrations(
                                    subj=subj,
                                    resolution=resolution,
                                    nlin_module=nlin_module,
                                    linear_conf=default_lsq12_multilevel_minctracc,
                                    # TODO why does this use almost-defaults instead of parsing?!
                                    # TODO when this was specialized to ANTS we could set iterations;
                                    # can/should we do something similar here in generality?
                                    nlin_options=options.nlin.nlin_protocol #nlin_module.get_default_conf(resolution=resolution)
                                        #.replace(file_resolution=options.registration.resolution,
                                        #         iterations="100x100x100x50"
                                    ))
                   for s_id, subj in subj_id_to_Subjec_for_within_dict.items() }

    # create a montage image for each pair of time points
    for s_id, output_from_intra in chain_xfms.items():
        for time_pt_n, time_pt_n_plus_1, transform in output_from_intra[0]:
            montage_chain = pipeline_montage_dir + "/quality_control_chain_ID_" + s_id + \
                            "_timepoint_" + str(time_pt_n) + "_to_" + str(time_pt_n_plus_1) + ".png"
            chain_images = [transform.resampled, transform.target]
            chain_verification_images = s.defer(create_quality_control_images(chain_images,
                                                                              montage_output=montage_chain,
                                                                              montage_dir=pipeline_montage_dir,
                                                                              message="the alignment between ID " + s_id + " time point " +
                                                                                      str(time_pt_n) + " and " + str(time_pt_n_plus_1)))

    if options.application.verbose:
        print("\n\nTransformations gotten from the intrasubject registrations:")
        for s_id, output_from_intra in chain_xfms.items():
            print("ID: ", s_id)
            for time_pt_n, time_pt_n_plus_1, transform in output_from_intra[0]:
                print("Time point: ", time_pt_n, " to ", time_pt_n_plus_1, " trans: ", transform.xfm.path)
            print("\n")

    ## stats
    #
    # The statistic files we want to create are the following:
    # 1) subject <----- subject_common_time_point                              (resampled to common average)
    # 2) subject <----- subject_common_time_point <- common_time_point_average (incorporates inter subject differences)
    # 3) subject_time_point_n <----- subject_time_point_n+1                    (resampled to common average)

    # create transformation from each subject to the final common time point average,
    # and from each subject to the subject's common time point
    (non_rigid_xfms_to_common_avg, non_rigid_xfms_to_common_subj) = s.defer(get_chain_transforms_for_stats(subj_id_to_Subjec_for_within_dict,
                                                                            intersubj_img_to_xfm_to_common_avg_dict,
                                                                            chain_xfms))

    # Ad 1) provide transformations from the subject's common time point to each subject
    #       These are temporary, because they still need to be resampled into the
    #       average common time point space
    determinants_from_subject_common_to_subject = map_over_time_pt_dict_in_Subject(
        lambda xfm: s.defer(determinants_at_fwhms(xfms=[s.defer(invert_xfmhandler(xfm))],
                                                  inv_xfms=[xfm],  # determinants_at_fwhms now vectorized-unhelpful here
                                                  blur_fwhms=options.stats.stats_kernels)),
        non_rigid_xfms_to_common_subj)
    # the content of determinants_from_subject_common_to_subject is:
    #
    # {subject_ID : Subject(inter_subject_time_pt, time_pt_dict)
    #
    # where time_pt_dict contains:
    #
    # {time_point : Tuple(List[Tuple(float, Tuple(MincAtom, MincAtom))],
    #                     List[Tuple(float, Tuple(MincAtom, MincAtom))])
    #
    # And to be a bit more verbose:
    #
    # {time_point : Tuple(relative_stat_files,
    #                     absolute_stat_files)
    #
    # where either the relative_stat_files or the absolute_stat_files look like:
    #
    # [blur_kernel_1, (determinant_file_1, log_of_determinant_file_1),
    #  ...,
    #  blur_kernel_n, (determinant_file_n, log_of_determinant_file_n)]
    #
    # Now the only thing we really want to do, is to resample the actual log
    # determinants that were generated into the space of the common average.
    # To make that a little easier, I'll create a mapping that will contain:
    #
    # {subject_ID: Subject(intersubject_timepoint, {time_pt_1: [stat_file_1, ..., stat_file_n],
    #                                               ...,
    #                                               time_pt_n: [stat_file_1, ..., stat_file_n]}
    # }
    for s_id, subject_with_determinants in determinants_from_subject_common_to_subject.items():
        transform_from_common_subj_to_common_avg = subj_ID_to_xfm_handler_to_common_avg[s_id].xfm
        for time_pt, determinant_info in subject_with_determinants.time_pt_dict.items():
            # here, each determinant_info is a DataFrame where each row contains
            # 'abs_det', 'nlin_det', 'log_nlin_det', 'log_abs_det', 'fwhm' fields
            # of the log-determinants, blurred at various fwhms (corresponding to different rows)
            for _ix, row in determinant_info.iterrows():
                for log_det_file_to_resample in (row.log_full_det, row.log_nlin_det):
                    # TODO the MincAtoms corresponding to the resampled files are never returned
                    new_name_wo_ext = log_det_file_to_resample.filename_wo_ext + "_resampled_to_common"
                    s.defer(mincresample(img=log_det_file_to_resample,
                                         xfm=transform_from_common_subj_to_common_avg,
                                         like=log_det_file_to_resample,
                                         new_name_wo_ext=new_name_wo_ext,
                                         subdir="stats-volumes"))

    # Ad 2) provide transformations from the common avg to each subject. That's the
    #       inverse of what was provided by get_chain_transforms_for_stats()
    determinants_from_common_avg_to_subject = map_over_time_pt_dict_in_Subject(
        lambda xfm: s.defer(determinants_at_fwhms(xfms=[s.defer(invert_xfmhandler(xfm))],
                                                  inv_xfms=[xfm],  # determinants_at_fwhms now vectorized-unhelpful here
                                                  blur_fwhms=options.stats.stats_kernels)),
        non_rigid_xfms_to_common_avg)

    # TODO don't just return an (unnamed-)tuple here
    return Result(stages=s, output=Namespace(non_rigid_xfms_to_common=non_rigid_xfms_to_common_avg,
                                             determinants_from_common_avg_to_subject=determinants_from_common_avg_to_subject,
                                             determinants_from_subject_common_to_subject=determinants_from_subject_common_to_subject))


K = TypeVar('K')
T = TypeVar('T')
U = TypeVar('U')


def get_closest_model_from_pride_of_models(pride_of_models_dict,
                                           time_point):
    """
    returns the RegistrationTargets from the "closest" initial model in the
    pride_of_models_dict. If the exact time point is not present in the
    pride_of_models_dict, and there is a tie, the RegistrationTargets from the
    larger/older time point will be returned
    """

    time_point_float = float(time_point)

    # the trivial case first: a time_point that is part of the
    # pride of models
    if time_point_float in pride_of_models_dict:
        return pride_of_models_dict[time_point_float]

    # if the exact time point is not present, get the
    # closest match
    sorted_keys = sorted(pride_of_models_dict.keys())
    for i in range(len(sorted_keys)):
        sorted_keys[i] = float(sorted_keys[i])

    index_on_the_right = bisect.bisect(sorted_keys, time_point_float)

    # because otherwise index_on_the_right - 1 < 0, which causes weird indexing ...
    if index_on_the_right == 0:
        print("Using initial model of time point: %d for file with actual time point: %s"
              % (sorted_keys[0], str(time_point_float)))
        return pride_of_models_dict[sorted_keys[0]]

    diff_with_smaller_timepoint = time_point_float - float(sorted_keys[index_on_the_right - 1])
    diff_with_larger_timepoint = float(sorted_keys[index_on_the_right]) - time_point_float

    if diff_with_smaller_timepoint >= diff_with_larger_timepoint:
        print("Using initial model of time point: " + str(sorted_keys[index_on_the_right]) +
              " for file with actual time point: " + str(time_point_float))
        return pride_of_models_dict[sorted_keys[index_on_the_right]]
    else:
        print("Using initial model of time point: " + str(sorted_keys[index_on_the_right - 1]) +
              " for file with actual time point: " + str(time_point_float))
        return pride_of_models_dict[sorted_keys[index_on_the_right - 1]]


def map_with_index_over_time_pt_dict_in_Subject(f : Callable[[T], U],
                                                d : Dict[K, Subject[T]]) -> Dict[K, Subject[U]]:
    #TODO: fix the documentation...
    ## """Map `f` non-destructively (if `f` is) over (the values of)
    ##the inner time_pt_dict of a { subject : Subject }
    ##
    ##>>> (map_over_time_pt_dict_in_Subject(lambda x: x[3],
    ##...          { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict={3:'s1_3.mnc', 4:'s1_4.mnc'}),
    ##...            's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict={4:'s2_4.mnc', 5:'s2_5.mnc'})} )
    ##...   == { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {3:'3',4:'4'}),
    ##...        's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {4:'4',5:'5'}) })
    ##True
    ##"""
    new_d = {}  # type: Dict[K, Subject[T]]
    for s_id, subj in d.items():
        new_time_pt_dict = {}  # type: Dict[int, U]
        for t, x in subj.time_pt_dict.items():
            new_time_pt_dict[t] = f(x, t)
        new_subj = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                           time_pt_dict = new_time_pt_dict)  # type: Subject[U]
        new_d[s_id] = new_subj
    return new_d


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
    return new_d


def parse_common(string : str) -> bool:
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


Row = Filename = str


# TODO standardize on pt/point
# TODO write some longer (non-doc)tests
def parse_csv(rows : Iterator[Row], common_time_pt : int) -> Dict[str, Subject[Filename]]:
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
    # FIXME the type/name of `rows` is slightly a lie; it probably has a header row which isn't an ordinary Row
    # (otherwise we wouldn't be able to use csv.DictReader on it)
    subject_info = defaultdict(lambda: Subject(intersubject_registration_time_pt=None))
    # Populate the subject -> Subject dictionary from the rows"""
    for row in csv.DictReader(rows):
        try:
            subj_id   = row['subject_id'].strip()
            timepoint = int(row['timepoint'].strip())
            filename  = row['filename'].strip()
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
            if common_time_pt not in (None, s.intersubject_registration_time_pt):
                print('note: overriding common_time_pt %d with time point %d for subject %s'
                      % (common_time_pt, s.intersubject_registration_time_pt, s_id))
                    
    return subject_info


def get_chain_transforms_for_stats(pipeline_subject_info, intersubj_xfms_dict, chain_xfms_dict):
    """
    pipeline_subject_info

    intersubj_xfms_dict

    chain_xfms_dict -- {subject_ID : ( [ (time_point_n, time_point_n+1, XfmHandler(time_point -> time_point + 1)),
                                         ..., (,,,) ],
                                       index_of_common_time_point) }


    This function takes a subject mapping (with timepoints to MincAtoms) and returns a
    subject mapping of timepoints to `XfmHandler`s. Those transformations for
    each subject will contain the non-rigid transformation to the common time point average
    
    chain_xfms_dict maps subject_ids to a tuple containing a list of tuples
    (time_point_n, time_point_n+1, transformation)
    and the index to the common time point in that list
    """

    s = Stages()
    dict_transforms_to_common_avg = {}
    dict_transforms_to_subject_common_tp = {}
    dict_transforms_from_common_tp_to_common_avg = {}
    for s_id, subj in pipeline_subject_info.items():
        # dictionary: {time_pt : XfmHandler_time_pt_to_final_common_avg}
        trans_to_final_common_avg_dict = {}
        # dictionary: {time_pt : XfmHandler_time_pt_to_subject_common_time_pt}
        trans_to_subject_common_time_pt = {}



        ##############################################################
        #                -------------
        #  time_1   ... | time_common | ...   time_n
        #                -------------
        #
        # the transformation for the common time point is easy,
        # intersubj_xfms_dict[subj.intersubject_registration_image]
        # returns the XfmHandler from the subject common time point
        # to the common time point average
        trans_to_final_common_avg_dict[subj.intersubject_registration_time_pt] = \
            intersubj_xfms_dict[subj.intersubject_registration_image]
        # there is no transform from the common time point to the common time
        # point. Technically it is the identity transformation, but there is
        # no use in generating a stats file from the identity transformation,
        # so we simply won't generate anything

        chain_transforms, index_of_common_time_pt = chain_xfms_dict[s_id]

        # will hold the XfmHandler from current to average of common time pt
        current_xfm_to_common_avg = intersubj_xfms_dict[subj.intersubject_registration_image]
        # and the XfmHandler from current to the subject common time point
        # starts out as None (technically the identity transformation)
        current_xfm_to_common_subject = None

        # we start at the common time point and are going forward at this point
        # so we will assign the concatenated transform to the target of each 
        # transform we are adding (which is why we take the inverse)
        #
        #                       - - - - - - - - >
        #  time_1   ...   time_common   ...   time_n
        #
        #
        for time_pt_n, time_pt_n_plus_1, transform in chain_transforms[index_of_common_time_pt:]:
            current_xfm_to_common_avg = s.defer(concat_xfmhandlers([s.defer(invert_xfmhandler(transform)), current_xfm_to_common_avg],
                                                               name="id_%s_pt_%s_to_common_avg" % (s_id, time_pt_n_plus_1)))
            # we are moving away from the common time point. That means that the transformation
            # we are adding here is the inverse of n -> n+1, and should be added to time point n+1
            trans_to_final_common_avg_dict[time_pt_n_plus_1] = current_xfm_to_common_avg

            if current_xfm_to_common_subject == None:
                current_xfm_to_common_subject = s.defer(invert_xfmhandler(transform))
            else:
                current_xfm_to_common_subject = s.defer(concat_xfmhandlers([s.defer(invert_xfmhandler(transform)), current_xfm_to_common_subject],
                                                               name="id_%s_pt_%s_to_common_subject" % (s_id, time_pt_n_plus_1)))
            trans_to_subject_common_time_pt[time_pt_n_plus_1] = current_xfm_to_common_subject

        # we need to do something similar moving backwards: make sure to reset
        # the current_xfm_to_common_avg here!
        #
        #    < - - - - - - - -
        #  time_1   ...   time_common   ...   time_n
        #
        #
        current_xfm_to_common_avg = intersubj_xfms_dict[subj.intersubject_registration_image]
        current_xfm_to_common_subject = None
        # we have to be careful here... if the index_of_common_time_pt is 0 (i.e. all images are
        # registered towards the first file in the time line, the following command will call:
        # .... chain_transforms[-1::-1] and that in turn will start at the end of the list
        # because -1 wraps around. To prevent this case, we ensure that the index_of_common_time_pt
        # is greater than 0
        if index_of_common_time_pt > 0:
            for time_pt_n, time_pt_n_plus_1, transform in chain_transforms[index_of_common_time_pt-1::-1]:
                current_xfm_to_common_avg = s.defer(concat_xfmhandlers([transform, current_xfm_to_common_avg],
                                                                   name="id_%s_pt_%s_to_common_avg" % (s_id, time_pt_n)))
                trans_to_final_common_avg_dict[time_pt_n] = current_xfm_to_common_avg

                if current_xfm_to_common_subject == None:
                    current_xfm_to_common_subject = transform
                else:
                    current_xfm_to_common_subject = s.defer(concat_xfmhandlers([transform, current_xfm_to_common_subject],
                                                                   name="id_%s_pt_%s_to_common_subject" % (s_id, time_pt_n)))
                trans_to_subject_common_time_pt[time_pt_n] = current_xfm_to_common_subject

        new_subj_to_common_avg = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                           time_pt_dict = trans_to_final_common_avg_dict)
        dict_transforms_to_common_avg[s_id] = new_subj_to_common_avg

        new_subj_to_common_subj = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                          time_pt_dict = trans_to_subject_common_time_pt)
        dict_transforms_to_subject_common_tp[s_id] = new_subj_to_common_subj
    return Result(stages=s, output=(dict_transforms_to_common_avg, dict_transforms_to_subject_common_tp))


def main():
    p = CompoundParser(
          [execution_parser,
           application_parser,
           registration_parser,
           lsq6_parser,
           # TODO: either switch back to automatically unpacking as part of `parse`
           # or write a helper to do \ns: C(**vars(ns))
           lsq12_parser, # should be MBM or build_model ...
           nlin_parser,
           #AnnotatedParser(parser=BaseParser(addLSQ12ArgumentGroup), namespace='lsq12-inter-subj'),
           #addNLINArgumentGroup,
           stats_parser,
           AnnotatedParser(parser=_chain_parser, prefix="chain", namespace="chain")])
    
    # TODO could abstract and then parametrize by prefix/ns ??
    options = parse(p, sys.argv[1:])
    s= Stages()
    # TODO: the registration resolution should be set somewhat outside
    # of any actual function? Maybe the right time to set this, is here
    # when options are gathered?
    if not options.registration.resolution:
        # if the target for the registration_chain comes from the pride_of_models
        # we can not use the registration_targets() function. The pride_of_models
        # works in a fairly different way, so we will separate out that option.
        if options.lsq6.target_type == TargetType.pride_of_models:
            pride_of_models_mapping = get_pride_of_models_mapping(options.lsq6.target_file,
                                                                  options.application.output_directory,
                                                                  options.application.pipeline_name)
            # all initial models that are part of the pride of models must have
            # the same resolution (it's currently a requirement). So we can get the
            # resolution from any of the RegistrationTargets:
            random_key = list(pride_of_models_mapping)[0]
            file_for_resolution = pride_of_models_mapping[random_key].registration_standard.path
        else:
            file_for_resolution = s.defer(registration_targets(lsq6_conf=options.lsq6,
                                                       app_conf=options.application,
                                                        reg_conf=options.registration)).registration_standard.path
        options.registration = options.registration.replace(
                                   resolution=get_resolution_from_file(file_for_resolution))
    
    # *** *** *** *** *** *** *** *** ***

    chain_result = chain(options)
    chain_output = chain_result.output

    # write some useful CSVs:
    analysis_csv = open("".join([options.application.pipeline_name, "_analysis_files.csv"]), "w")
    print("subject_id, timepoint, fwhm, log_det_absolute_second_level, "
          "log_det_relative_second_level, log_det_absolute_first_level, "
          "log_det_relative_first_level",
          file=analysis_csv)
    for subj_id, subject in chain_output.determinants_from_common_avg_to_subject.items():
        for timept, img in subject.time_pt_dict.items():
            # these rows contain full_det, fwhm, inv_xfm, log_full_det, log_nlin_det,... (some more)
            for i, row in img.iterrows():
                #import pdb; pdb.set_trace()
                if timept != subject.intersubject_registration_time_pt:
                    # TODO: this is not really the proper way of dealing with things. If
                    # the code above changes (i.e., if the filenames change) this won't work anymore...
                    from_subject_common_absolute = os.path.realpath(re.sub(".mnc", "_resampled_to_common.mnc", chain_output.determinants_from_subject_common_to_subject[subj_id].time_pt_dict[timept].iloc[i].log_full_det.path))
                    from_subject_common_relative = os.path.realpath(re.sub(".mnc", "_resampled_to_common.mnc", chain_output.determinants_from_subject_common_to_subject[subj_id].time_pt_dict[timept].iloc[i].log_nlin_det.path))
                else:
                    from_subject_common_absolute = "NA"
                    from_subject_common_relative = "NA"
                print(",".join([str(subj_id), str(timept), str(row.fwhm), os.path.realpath(row.log_full_det.path),
                               os.path.realpath(row.log_nlin_det.path), from_subject_common_absolute, from_subject_common_relative]),
                      file=analysis_csv)
                      #, ",",
                      #chain_output.determinants_from_subject_common_to_subject[subj_id].time_pt_dict[timept].iloc[i].log_full_det.path, ",",
                      #chain_output.determinants_from_subject_common_to_subject[subj_id].time_pt_dict[timept].iloc[i].log_nlin_det.path
                      #)
    analysis_csv.close()
    #import pdb; pdb.set_trace()

#Namespace(non_rigid_xfms_to_common=non_rigid_xfms_to_common_avg,
#                                             determinants_from_common_avg_to_subject=determinants_from_common_avg_to_subject,
#                                             determinants_from_subject_common_to_subject

    #chain_stages = chain(options).stages

    execute(chain_result.stages, options)


if __name__ == "__main__":
    main()
