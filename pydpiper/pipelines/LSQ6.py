#!/usr/bin/env python3
#TODO s/MincAtom/MincFile/g

from pydpiper.core.stages       import CmdStage, Result, Stages
from pydpiper.core.util         import raise_
from pydpiper.minc.files        import MincAtom
from pydpiper.minc.registration import multilevel_minctracc, nu_correct, inormalize

def lsq6_pipeline_part(options):
    """Run a pipeline from its command-line inputs"""
    # TODO shouldn't options.initial_model already be a MincFile?
    # if we pass it in to several parts of the pipeline (LSQ6, LSQ12, etc.) it will be re-initialized ...
    # FIXME account in a nice/uniform way for possibility of transform supplied with initial model/lsq6 target

    lsq6_target = MincAtom(options.initial_model if options.initial_model else
                           options.lsq6_target   if options.lsq6_target   else
                           imgs[0]               if options.bootstrap     else
                           raise_(ValueError("No target for LSQ6 registration supplied (via options.{initial_model,lsq6_target,bootstrap})")))
    # switch(options.target_specification,
    #  { 'bootstrap'   : imgs[0],
    #    'lsq6-target' : options.lsq6_target,   # TODO use a mutually exclusive group for these options
    #    'init-model'  : options.initial_model })
    # NotImplemented # initial model or imgs[0] or chosen image ...
    # TODO make a new file here based on lsq6_target but which can have its own writeable directory ...
    raise NotImplemented
   

def lsq6(imgs, target, options): # [mnc], mnc, LSQ6Conf -> Result(..., output=???)
    s = Stages()

    # TODO verify that options (lsq6_{method,target,...}) make sense ... or should this be done before lsq6 call?
    lsq6_method = options.lsq6_method or 'lsq6_large_rotations'

    def switch(k, choices): # A, { A : () -> B } -> B
        # TODO move to util module, add optional default, add wildcards, turn into preprocessor ...
        try:
            v = choices[k]
        except KeyError:
            raise SwitchError('unhandled case %s (no default supplied); valid: %s' % (k, list(choices.keys())))
        else:
            return v()

    # I don't like the fact that the whole LSQ6 module seems to be a switch around the lsq6 registration ... or maybe I do ...
    lsq6_xfms = s.defer(switch(lsq6_method, # TODO do we need an `initial_model` field or does `target` suffice? registration should be agnostic to source of target, right?
      { 'lsq6_simple'            : lambda :  # TODO the options args shouldn't be string-indexed dictionaries but conf objects ...
          lsq6_hierarchical_minctracc(sources=imgs, target=lsq6_target, options={ 'initial_transform' : 'identity', 'lsq6_protocol' : NotImplemented }),
        'lsq6_centre_estimation' : lambda :
          lsq6_hierarchical_minctracc(sources=imgs, target=lsq6_target, options={}),
        'lsq6_large_rotations'   : lambda :
          lsq6_rotational_minctracc(sources=imgs,   target=lsq6_target, options={})
      }))

    return Result(stages=s, output=lsq6_xfms)

# TODO make higher-order way to run part of a pipeline on certain images, then create new xfmhs with replaced inputs/resampled files
# (idea: images have same 'geometry'/coordinates but may be modified, e.g., tagged kidney tip images)
# may need to supply some fns to determine how to get/set outputs as our interfaces aren't uniform enough to know this in general

if __name__ == "__main__":
    pass