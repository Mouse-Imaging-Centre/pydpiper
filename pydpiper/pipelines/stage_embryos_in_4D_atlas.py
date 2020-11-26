#!/usr/bin/env python3

import os
import subprocess
import warnings
import pandas as pd
from configargparse import ArgParser
from typing import List

from pydpiper.core.arguments        import (lsq6_parser, lsq12_parser, nlin_parser,
                                            CompoundParser, AnnotatedParser, BaseParser)
from pydpiper.execution.application import mk_application
from pydpiper.core.stages           import Stages, Result
from pydpiper.minc.analysis         import mincblob
from pydpiper.pipelines.MAGeT       import get_imgs
from pydpiper.minc.files            import MincAtom
from pydpiper.minc.registration     import (ensure_distinct_basenames, lsq6_lsq12_nlin, LSQ6Conf,
                                            MinctraccConf, get_resolution_from_file,
                                            get_linear_configuration_from_options, LinearTransType,
                                            get_nonlinear_component, invert_xfmhandler, minc_displacement)
from pydpiper.minc.nlin             import NLIN
"""
General idea:
1) single embryo input file
2) rough size estimation using 

biModalT=`mincstats -quiet -biModalT $file`; 
volume  =`mincstats -quiet -volume -floor $biModalT $file`

3) 4D embryo atlas input:
  -- directory with time points (e.g., E13.6.mnc, E13.7.mnc, E13.8.mnc, ...)
  -- masks (e.g., E13.6_mask.mnc, E13.7_mask.mnc, E13.8_mask.mnc, ...)
  -- rough size estimates of the time points based on the masks
--> stored in a csv like so:

echo volume,timepoint,file,mask_file > 4D_embryo_mapping.csv; 
for main in `seq 12 16`; do 
    for minor in `seq 0 9`; do 
        for file in E${main}.${minor}*_mask.mnc; do 
            realpathmainfile=`realpath ${file/_mask.mnc/.mnc}`; 
            realpathmask=`realpath $file`; 
            volume=`mincstats -quiet -volume -floor 1500  $realpathmainfile`; 
            echo $volume,${main}.${minor},$realpathmainfile,$realpathmask  >> 4D_embryo_mapping.csv ;  
        done; 
    done; 
done


Run an lsq6 - lsq12 - nlin (antsRegistration) registration from the input embryo
to all time points in the 4D atlas

Gather stats:

TimePoint, CrossCorrelation, TotalDeformationEnergy, NonLinearDeformationEnergy
...
...
E13.6        0.90               4.23                     3.51
E13.7        0.93               3.89                     3.45
E13.8        0.89               4.82                     4.05
...
...
"""

def instances_in_4D_atlas_from_csv(csv_4D: str, pipeline_sub_dir: str) -> pd.Series:

    def map_to_MincAtom(row):
        return MincAtom(name=row.file,
                        pipeline_sub_dir=pipeline_sub_dir,
                        mask=MincAtom(name=row.mask_file,
                                      pipeline_sub_dir=pipeline_sub_dir))
    try:
        df = pd.read_csv(csv_4D, usecols=["volume", "timepoint", "file", "mask_file"])
    except ValueError:
        warnings.warn("Could not read csv file: ", csv_4D, ". Does the csv file have the "
                      "following column names: volume, timepoint, file, mask_file?")
        raise
    df['mincatom'] = df.apply(map_to_MincAtom, axis=1)
    return df

def get_volume_estimate(imgs: List[MincAtom]):
    """
    Sometimes when the bimodalt value is calculated, the threshold
    will separate the embryo+gel from the background. In that case,
    the calculated volume will take up more than 50% of the entire volume
    which is highly unlikely. So if that's the case, we'll re-calculate the
    bimodalt value using the -kapur method, as it seems to separate out
    the "second" peak in the data.
    """
    volumes = []
    for img in imgs:
        biModalT = subprocess.check_output(["mincstats", "-quiet", "-biModalT", img.orig_path]).rstrip().decode()
        volume = subprocess.check_output(["mincstats", "-quiet", "-floor", biModalT, "-volume", img.orig_path]).rstrip().decode()
        total_file_volume = subprocess.check_output(["mincstats", "-quiet", "-volume", img.orig_path]).rstrip().decode()
        # if the ratio is larger than 0.5, we'll recompute
        if float(volume) / float(total_file_volume) > 0.5:
            biModalT = subprocess.check_output(["mincstats", "-quiet", "-biModalT", "-kapur", img.orig_path]).rstrip().decode()
            volume = subprocess.check_output(["mincstats", "-quiet", "-floor", biModalT, "-volume", img.orig_path]).rstrip().decode()
        volumes += [volume]
    return volumes

def get_index_closest_volume_match(volume, full_4D_atlas_info):
    volume_4D_atlas = full_4D_atlas_info["volume"].astype(float)
    best_index = -1
    diff = 999999999
    for i in range(volume_4D_atlas.size):
        cur_diff = abs(volume - volume_4D_atlas[i])
        if cur_diff < diff:
            best_index = i
            diff = cur_diff
    return best_index

def match_embryo_to_4D_atlas(embryo_with_volume_est,
                             full_4D_atlas_info,
                             lsq6_conf: LSQ6Conf,
                             lsq12_conf: MinctraccConf,
                             nlin_module: NLIN,
                             resolution: float,
                             nlin_options):
    s = Stages()

    # 1 what's the closest match in the 4D atlas?
    mid_index = get_index_closest_volume_match(embryo_with_volume_est["rough_volume"].astype(float), full_4D_atlas_info)

    print("Best initial match for: \n", embryo_with_volume_est["mincatom"].orig_path, " ", full_4D_atlas_info.loc[mid_index]["timepoint"])

    # register embryo to closest match +/- 5 time points
    # make sure we don't index outside the possible range
    lowest_index  = max(0, mid_index - 7)
    highest_index = min(full_4D_atlas_info.shape[0] - 1, mid_index + 7)

    all_transforms = [s.defer(lsq6_lsq12_nlin(source=embryo_with_volume_est["mincatom"],
                                              target=full_4D_atlas_info.loc[i]["mincatom"],
                                              lsq6_conf=lsq6_conf,
                                              lsq12_conf=lsq12_conf,
                                              nlin_module=nlin_module,
                                              resolution=resolution,
                                              nlin_options=nlin_options.nlin_protocol,
                                              resampled_post_fix_string="E" + str(full_4D_atlas_info.loc[i]["timepoint"]))) for
                      i in range(lowest_index, highest_index + 1, 1)]

    # gather stats on those registrations
    # the match is determined by the sum of the magnitude
    # of the inverse transformation from 4D instance -> embryo
    # using the mask of the 4D instance to limit the total sum
    # 1) calculate inverse
    all_inv_transforms = [s.defer(invert_xfmhandler(xfm)) for xfm in all_transforms]
    minc_displacement_grids = [s.defer(minc_displacement(inv_xfm)) for inv_xfm in all_inv_transforms]
    magnitudes = [s.defer(mincblob(op='magnitude', grid=disp_grid)) for disp_grid in minc_displacement_grids]

    return Result(stages=s, output=all_transforms)

def stage_embryos_pipeline(options):
    s = Stages()

    imgs = get_imgs(options.application)
    rough_volume_imgs = get_volume_estimate(imgs)
    imgs_and_rough_volume = pd.DataFrame({"mincatom" : imgs,
                                          "rough_volume" : pd.Series(rough_volume_imgs, dtype=float)})

    ensure_distinct_basenames([img.path for img in imgs])

    output_directory = options.application.output_directory
    output_sub_dir = os.path.join(output_directory,
                                  options.application.pipeline_name + "_4D_atlas")

    time_points_in_4D_atlas = instances_in_4D_atlas_from_csv(options.staging.staging.csv_4D,
                                                             output_sub_dir)

    # we can use the resolution of one of the time points in the 4D atlas
    # for all the registrations that will be run.
    resolution = get_resolution_from_file(time_points_in_4D_atlas.loc[0]["mincatom"].orig_path)

    print(options.staging.lsq12)

    lsq12_conf = get_linear_configuration_from_options(options.staging.lsq12,
                                                       transform_type=LinearTransType.lsq12,
                                                       file_resolution=resolution)

    nlin_component = get_nonlinear_component(options.staging.nlin.reg_method)

    # match each of the embryos individually
    for i in range(imgs_and_rough_volume.shape[0]):
        s.defer(match_embryo_to_4D_atlas(imgs_and_rough_volume.loc[i],
                                         time_points_in_4D_atlas,
                                         lsq6_conf=options.staging.lsq6,
                                         lsq12_conf=lsq12_conf,
                                         nlin_module=nlin_component,
                                         resolution=resolution,
                                         nlin_options=options.staging.nlin))


    return Result(stages=s, output=None)



def _mk_staging_parser(parser : ArgParser):
    group = parser.add_argument_group("Embryo staging options", "Options for staging embryos in a 4D atlas.")
    group.add_argument("--csv-4D", dest="csv_4D", type=str,
                       help="CSV containing information about the 4D altas. Should contain "
                            "the following fields: `volume`, `timepoint`, `file`, "
                            "`mask_file`.")
    return parser

staging_parser = AnnotatedParser(parser=BaseParser(_mk_staging_parser(ArgParser(add_help=False)),
                                                   "staging"),
                                 namespace="staging")


# TODO: this is a giant hack, but I acutally don't know how
# TODO: to properly change the defaults in these parsers...
# the registration targets for this pipeline are known, they
# are a number of the time points from the 4D atlas. As such,
# no target needs to be specified by the user. We'll just initialize
# it here with
lsq6_parser_with_bootstrap = lsq6_parser
lsq6_parser_with_bootstrap.parser.argparser.set_defaults(bootstrap=True)
lsq6_parser_with_bootstrap.parser.argparser.set_defaults(nuc=False)
lsq6_parser_with_bootstrap.parser.argparser.set_defaults(inormalize=False)
lsq6_parser_with_bootstrap.parser.argparser.set_defaults(lsq6_method="lsq6_centre_estimation")

staging_parser = CompoundParser([lsq6_parser_with_bootstrap,
                                 lsq12_parser,
                                 nlin_parser,
                                 staging_parser])

stage_embryos_application = mk_application(parsers=[AnnotatedParser(parser=staging_parser,
                                                                   namespace="staging")],
                                          pipeline=stage_embryos_pipeline)



if __name__ == "__main__":
    stage_embryos_application()


