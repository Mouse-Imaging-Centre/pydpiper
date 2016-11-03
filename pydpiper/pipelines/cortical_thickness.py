#!/usr/bin/env python3

import os

from configargparse import ArgumentParser
import pandas as pd

from pydpiper.minc.containers import XfmHandler
from pydpiper.core.stages import Stages, Result
from pydpiper.execution.application import mk_application
from pydpiper.minc.files import FileAtom, MincAtom, XfmAtom
from pydpiper.minc.thickness import cortical_thickness


def cortical_thickness_pipeline(options):
    s = Stages()

    #imgs = [MincAtom(name, pipeline_sub_dir=os.path.join(options.application.output_directory,
    #                                                     options.application.pipeline_name + "_processed"))
    #        for name in options.application.files]

    pipeline_sub_dir = os.path.join(options.application.output_directory,
                                    options.application.pipeline_name + "_processed")

    #def atom(atom_type, file):
    #    return atom_type(file, pipeline_sub_dir=pipeline_sub_dir)  # TODO output_sub_dir, ....

    # TODO are all these fields actually used?  If not, omit from CSV?
    xfms = (pd.read_csv(options.thickness.xfm_csv)
            .apply(axis=1,  # TODO fill out <..>Atom(...) fields ...
                   func=lambda row: XfmHandler(
                          source=MincAtom(row.source, pipeline_sub_dir=pipeline_sub_dir),
                          target=MincAtom(row.target, pipeline_sub_dir=pipeline_sub_dir),
                          resampled=None,   #MincAtom(row.resampled, pipeline_sub_dir=pipeline_sub_dir),
                          xfm=XfmAtom(row.xfm, pipeline_sub_dir=pipeline_sub_dir))))
    # TODO better way to unpack?

    result = s.defer(cortical_thickness(xfms=xfms,
                                        atlas=NotImplemented,
                                        label_mapping=options.thickness.label_mapping,
                                        atlas_fwhm=options.thickness.atlas_fwhm,
                                        thickness_fwhm=options.thickness.thickness_fwhm))

    return Result(stages=s, output=result)


def _mk_thickness_parser(p : ArgumentParser):
    p.add_argument("--xfm-csv", dest="xfm_csv", type=str, #required=True,
                   help="CSV file containing at least 'source', 'xfm', 'target', and 'resampled' columns")  # FIXME
    p.add_argument("--label-mapping", dest="label_mapping", type=FileAtom, #required=True,
                   help="CSV file containing structure information (see minclaplace/wiki/LaplaceGrid)")
    p.add_argument("--atlas-fwhm", dest="atlas_fwhm", type=float, required=True,  # default ?!
                   help="Blurring kernel (mm) for atlas")
    p.add_argument("--thickness-fwhm", dest="thickness_fwhm", type=float, required=True,  # default??
                   help="Blurring kernel (mm) for cortical surfaces")
    return p


thickness_parser = NotImplemented


cortical_thickness_application = mk_application(parsers=[thickness_parser], pipeline=cortical_thickness_pipeline)


if __name__ == "__main__":
    cortical_thickness_application()
