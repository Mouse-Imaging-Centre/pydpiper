#!/usr/bin/env python3

import os

from pydpiper.pipelines.MAGeT import get_imgs

from pydpiper.core.arguments import lsq6_parser
from pydpiper.core.stages import Stages, Result
from pydpiper.execution.application import mk_application
from pydpiper.minc.registration import get_resolution_from_file, registration_targets, lsq6_nuc_inorm

from pydpiper.minc.registration import MincAlgorithms
from pydpiper.itk.tools import Algorithms as ITKAlgorithms

# def generic_pipeline(options):
#     s = Stages()
#
#     output_dir    = options.application.output_directory   # can't be used outside here ...
#     pipeline_name = options.application.pipeline_name
#
#     if len(options.application.files) == 0:
#         raise ValueError("Please, some files!")
#
#     # TODO this is quite tedious and duplicates stuff in the registration chain ...
#     resolution = (options.registration.resolution or
#                   get_resolution_from_file(
#                       s.defer(registration_targets(lsq6_conf=options.lsq6,  # not really generic due to use of options.lsq6 ...
#                                            app_conf=options.application, reg_conf=options.registration)).registration_standard.path))
#
#     targets = s.defer(registration_targets(lsq6_conf=options.lsq6,  # see above ...
#                                    app_conf=options.application, reg_conf = options.registration,
#                                    first_input_file=options.application.files[0]))
#
#     # This must happen after calling registration_targets otherwise it will resample to options.registration.resolution
#     options.registration = options.registration.replace(resolution=resolution)
#     return (options, targets)


def lsq6_pipeline(options):
    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain and MBM ...
    lsq6_dir      = os.path.join(output_dir, pipeline_name + "_lsq6")

    image_algorithms = { 'minc' : MincAlgorithms, 'itk' : ITKAlgorithms }[options.registration.image_format]()  #reg_method=options.mbm.nlin.reg_method)

    imgs = get_imgs(options.application)


    s = Stages()

    # FIXME: why do we have to call registration_targets *outside* of lsq6_nuc_inorm? is it just because of the extra
    # options required?
    targets = s.defer(registration_targets(lsq6_conf=options.lsq6,
                                           app_conf=options.application,
                                           reg_conf=options.registration,
                                           first_input_file=imgs[0].path))

    # TODO this is quite tedious and duplicates stuff in the registration chain ...
    resolution = (options.registration.resolution or
                    get_resolution_from_file(targets.registration_standard.path))


    # This must happen after calling registration_targets otherwise it will resample to options.registration.resolution
    options.registration = options.registration.replace(resolution=resolution)
    lsq6_result = s.defer(lsq6_nuc_inorm(imgs=imgs,
                                         resolution=resolution,
                                         registration_targets=targets,
                                         image_algorithms=image_algorithms,
                                         lsq6_dir=lsq6_dir,
                                         lsq6_options=options.lsq6))

    return Result(stages=s, output=lsq6_result)

lsq6_application = mk_application(parsers=[lsq6_parser], pipeline=lsq6_pipeline)


if __name__ == "__main__":
    lsq6_application()
