from   __future__ import print_function
from   collections import namedtuple
#import copy
import os.path
import shlex

from atom.api import (Atom, Bool, Int, Float, List,
                      Enum, Tuple, Instance, Str)

from pydpiper.minc.files            import MincAtom
from pydpiper.minc.containers       import XfmHandler, Registration
from pydpiper.core.stages     import CmdStage, cmd_stage, Stages
from pydpiper.core.containers import Result
from pydpiper.core.util       import flatten
from pydpiper.core.conversion import InputFile, OutputFile

# FIXME a lot of input=..., output=... dependency specifications are broken as they specify a dependence
# on a handler rather than on a path (i.e., a string) (although this is OK since we can always call
# <whatever>.path -- the important thing is to make this **uniform**; need to add path accessor to xfmh)

# TODO canonicalize all names of form 'source', 'src', 'src_img', 'dest', 'target', ... in program

# TODO should these atoms/modules be remade as distinct classes with public outf and stage fields (only)?
# TODO output_dir (work_dir?) isn't used but could be useful; assert not(subdir and output_dir)?
def mincblur(img, fwhm, gradient=False, subdir='tmp'): # mnc -> mnc  #, output_dir='.'):
    """
    >>> img = MincAtom(name='/images/img_1.mnc', curr_dir='/scratch')
    >>> img_blur = mincblur(img=img, fwhm='0.056')
    >>> img_blur.output.path
    '/scratch/img_1/tmp/img_1_fwhm0.056.mnc'
    >>> [i.render() for i in img_blur.stages]
    ['mincblur -clobber -no_apodize -fwhm 0.056 /images/img_1.mnc /scratch/img_1/tmp/img_1_fwhm0.056.mnc']
    """
    outf  = img.newname_with_suffix("_fwhm%s" % fwhm + ("_dxyz" if gradient else ""), subdir=subdir)
    stage = CmdStage(
              inputs = [img], outputs = [outf],
              #this used to a function render_fn : conf, inf, outf -> string
              cmd = shlex.split('mincblur -clobber -no_apodize -fwhm %s %s %s' % (fwhm, img.path, outf.path)) \
                  + (['-gradient'] if gradient else []))
    return Result(stages=Stages([stage]), output=outf)

def mincresample_simple(img, xfm, like, extra_flags): # mnc -> mnc
    #TODO should extra_args (might contain -invert or -sinc) be(/be part of a conf object/list/dict/... ?
    """Resample an image, ignoring mask/labels"""
    outf = img.newname_with(lambda _old: xfm.name + '-resampled', subdir='resampled') # TODO update the mask/labels here
    #outf = MincAtom(name=xname + '-resampled' + '.mnc', subdir='resampled')
    stage = cmd_stage(['mincresample', '-clobber', '-2',
                       '-transform %s' % InputFile(xfm.path),
                       '-like %s' % InputFile(like.path),
                       InputFile(img.path), OutputFile(outf.path)] + extra_flags)
    return Result(stages=Stages([stage]), output=outf)

# TODO mincresample_simple could easily be replaced by a recursive call to mincresample
def mincresample(img, xfm, like, extra_flags): # mnc -> mnc
    s = Stages()
    # FIXME remove interpolation (-sinc, -tricubic, ...) from mask/label extra_flags -- from minc_atoms.py, seems default is nearest_neighbour?
    # TODO better solution - separate interpolation argument taking a nullable(?) enum val
    # FIXME add keep_real_range?
    new_mask   = s.defer(mincresample_simple(img.mask, xfm, like, extra_flags))   if img.mask   is not None else None
    new_labels = s.defer(mincresample_simple(img.labels, xfm, like, extra_flags)) if img.labels is not None else None
    new_img    = s.defer(mincresample_simple(img, xfm, like, extra_flags))
    # TODO stealing another file's name (and updating mask) could be made cleaner/easier ...
    # Note that new_img can't be used for anything until the mask/label files are also resampled.
    # This shouldn't create a problem with stage dependencies as long as masks/labels appear in inputs/outputs of CmdStages.
    # (If this isn't automatic, a relevant helper function would be trivial.)
    new_img.mask   = new_mask
    new_img.labels = new_labels
    return Result(stages=s, output=new_img)

def xfmconcat(xfms): # [xfm] -> xfm
    """
    >>> stages, xfm = xfmconcat([MincAtom('/tmp/%s' % i, curr_dir='/scratch') for i in ['t1.xfm', 't2.xfm']])
    >>> stages.pop().render()
    'xfmconcat /tmp/t1.xfm /tmp/t2.xfm /scratch/t1/concat_of_t1_and_t2.xfm'
    """
    if len(xfms) == 0:
        raise ValueError("`xfmconcat` arg `xfms` was empty (can't concat zero files)")
    elif len(xfms) == 1:
        return Result(stages=Stages(), output=xfms[0])
    else:
        outf = xfms[0].newname_with(
            lambda orig: "concat_of_%s" % "_and_".join([x.name for x in xfms])) #could do names[1:] if dirname contains names[0]?
        stage = CmdStage(
            inputs = xfms, outputs = [outf],
            cmd = shlex.split('xfmconcat %s %s' % (' '.join([x.path for x in xfms]), outf.path)))
        return Result(stages=Stages([stage]), output=outf)

"""xfmconcat lifted to work on XfmHandlers instead of MincAtoms"""
def concat(ts, extra_flags=[]): # TODO remove extra flags OR make ['-sinc'] default
    s = Stages()
    t = s.defer(xfmconcat([t.xfm for t in ts]))
    res = s.defer(mincresample(img=ts[0].source,
                            xfm=t,
                            like=ts[-1].target,
                            extra_flags=["-sinc"])) #TODO move extra flags to mincresample?
    return Result(stages=s,
                  output=XfmHandler(source=ts[0].source,
                                    target=ts[-1].target,
                                    xfm=t,
                                    resampled=res))

def nu_estimate(src): # MincAtom -> Result(stages=Set(Stages), MincAtom)
    out = src.newname_with_suffix("_nu_estimate", ext=".imp")

    # TODO finish dealing with masking as per lines 436-448 of the old LSQ6.py.  (note: the 'singlemask' option there is never used)
    # (currently we don't allow for a single mask or using the initial model's mask if the inputs don't have them)
    cmd = CmdStage(inputs = [src], outputs = [out],
                   cmd = shlex.split("nu_estimate -clobber -distance 8 -iterations 100 -stop 0.001 -fwhm 0.15 -shrink 4 -lambda 5.0e-02")
                       + (['-mask', src.mask.path] if src.mask else []) + [src.path, out.path])
    return Result(stages=Stages([cmd]), output=out)

def nu_evaluate(img, field): #mnc, file -> result(..., mnc)
    out = img.newname_with_suffix("_nuc")
    cmd = CmdStage(inputs = [img, field], outputs = [out],
                   cmd = ['nu_evaluate', '-clobber', '-mapping', field.path, img.path, out.path])
    return Result(stages=Stages([cmd]), output=out)

def nu_correct(src): # mnc -> result(..., mnc)
    s = Stages()
    return Result(stages=s, output=s.defer(nu_evaluate(src, s.defer(nu_estimate(src)))))

class INormalizeConf(Atom):
    const  = Int(1000)
    method = Enum('ratioOfMedians', 'medianOfRatios', 'meanOfRatios', 'meanOfLogRatios', 'ratioOfMeans')
    # NB the inormalize default is actually '-medianOfRatios'
    # FIXME how do we want to deal with situations where our defaults differ from the tools' defaults,
    # and in the latter case should we output the actual settings if the user doesn't explicitly set them?
    # should we put defaults into the classes or populate with None (which will often raise an error if disallowed)
    # and create default objects?

def inormalize(src, conf): # mnc, INormalizeConf -> result(..., mnc)
    out = src.newname_with('_inorm')
    cmd = CmdStage(inputs = [src], outputs = [out],
                   cmd = shlex.split('inormalize -clobber -const %s -%s' % (conf.const, conf.method))
                       + (['-mask', src.mask.path] if src.mask else [])
                       + [src.path, out.path])
    return Result(stages=Stages([cmd]), output=out)

class RotationalMinctraccConf(Atom):
    blur                = Instance(float)
    # these could also be floats, but then ints will be converted, hence printed with decimals ...
    # TODO we could subclass `Instance` to add a `default` field instead of using `factory`/`args`/...
    resample_step       = Instance((int, float)) # can use `factory=lambda : ...` to set default
    registration_step   = Instance((int, float))
    w_translations      = Instance((int, float))
    rotational_range    = Instance((int, float))
    rotational_interval = Instance((int, float))

# again, should this be folded into the class?  One could argue that doing so might be dangerous if
# one forgets to set a certain variable, but one might also forget to override a certain value from the default conf
# and overriding class defaults can actually be done in a MORE functional-seeming way than copying/setting values
# (without hacking on __dict__, etc.)
default_rotational_minctracc_conf = RotationalMinctraccConf(
  blur=0.56,
  resample_step=4,
  registration_step=10,
  w_translations=8,
  rotational_range=50,
  rotational_interval=10)

#FIXME consistently require that masks are explicitely added to inputs array (or not?)
def rotational_minctracc(source, target, conf):
    """Calls rotational_minctracc.py"""
    out = source.newname_with_suffix("_rotational_minctracc_FIXME")
    cmd = CmdStage(inputs = [source, target] + ([source.mask] if source.mask else []), outputs = [out],
        cmd = ["rotational_minctracc.py", 
               "-t", "/dev/shm/", 
               "-w", str(conf.w_trans),
               "-s", str(conf.resample_step),
               "-g", str(conf.registration_step),
               "-r", str(conf.rotational_range),
               "-i", str(conf.rotational_interval),
               source.path,
               target.path,
               out.path,
               "/dev/null"] + (['-m', source.mask.path] if source.mask else []))
    return Result(stages=Stages([cmd]), output=out)

class MinctraccConf(Atom):
    step_sizes        = Tuple((int, float), default=None)
    use_masks         = Instance(bool)
    is_nonlinear      = Instance(bool)
    # linear part
    simplex           = Instance((int, float))
    transform_type    = Enum(None, 'lsq3', 'lsq6', 'lsq7', 'lsq9', 'lsq10', 'lsq12', 'procrustes')
    tolerance         = Instance(float)
    w_rotations       = Tuple(float, default=None) # TODO should be Tuple((int,float),d=...) to preserve format
    w_translations    = Tuple(float, default=None) # TODO define types R3, ZorR3 for this
    w_scales          = Tuple(float, default=None) # TODO add length=3 validation
    w_shear           = Tuple(float, default=None)
    # nonlinear options
    iterations        = Instance(int)
    use_simplex       = Instance(bool)
    stiffness         = Instance(float)
    weight            = Instance(float)
    similarity        = Instance(float)
    objective         = Enum(None,'xcorr','diff','sqdiff','label',
                             'chamfer','corrcoeff','opticalflow')
    lattice_diameter  = Tuple(float, default=None)
    sub_lattice       = Instance(int)
    max_def_magnitude = Tuple((int, float), default=None)

class LinearMinctraccConf(Atom):
    """Generates a call to `minctracc` for linear registration only."""
    step_sizes        = Tuple((int, float), default=None)
    use_masks         = Instance(bool)
    simplex           = Instance((int, float))
    transform_type    = Enum(None, 'lsq3', 'lsq6', 'lsq7', 'lsq9', 'lsq10', 'lsq12', 'procrustes')
    tolerance         = Instance(float)
    w_rotations       = Tuple(float, default=None) # TODO should be Tuple((int,float),d=...) to preserve format
    w_translations    = Tuple(float, default=None) # TODO define types R3, ZorR3 for this
    w_scales          = Tuple(float, default=None) # TODO add length=3 validation
    w_shear           = Tuple(float, default=None)

class NonlinearMinctraccConf(Atom):
    """Generates a call to `minctracc` for nonlinear registration only."""

# should we even keep (hence render) the defaults which are the same as minctracc's?
# does this even get used in the pipeline?  LSQ6/12 get their own
# protocols, and many settings here are the minctracc default
def default_linear_minctracc_conf(transform_type):
  return MinctraccConf(
    is_nonlinear=False,
    step_sizes=(0.5,) * 3,
    use_masks=True,
    simplex=1,
    transform_type=transform_type,
    tolerance=0.001,
    w_scales=(0.02,) * 3,
    w_shear=(0.02,) * 3,
    w_rotations=(0.0174533,) * 3,
    w_translations=(1.0,) * 3)

def default_LSQ6_conf(resolution):
  pass

def default_LSQ12_conf(resolution):
  return MinctraccConf(
    
    )

default_lsq6_minctracc_conf, default_lsq12_minctracc_conf = map(default_linear_minctracc_conf, ('lsq6', 'lsq12'))

# FIXME wrapping this in a function gives a weird weird error ...
#def default_nonlinear_minctracc_conf():
step_size = 0.5
step_sizes = (step_size,) * 3
default_nonlinear_minctracc_conf = MinctraccConf(
    is_nonlinear=True,
    step_sizes=step_sizes,
    use_masks=True,
    iterations=40,
    similarity=0.8,
    use_simplex=True,
    stiffness=0.98,
    weight=0.8,
    objective='corrcoeff',
    lattice_diameter=tuple(map(lambda x: x * 3, step_sizes)),
    sub_lattice=6)
#  return c

#TODO move to utils
def space_sep(x):
    if isinstance(x, list) or isinstance(x, tuple): #ugh
        return ' '.join(map(str,x))
    else:
        return str(x)

# TODO add memory estimation hook
# TODO write separate function/args wrappers for linear/nonlinear minctracc?  (If only we had ADTs) ... in general one might want both parts (but we don't) - how to say this? ...
def minctracc(source, target, conf, transform=None):
    #if not conf.transform_type in [None, 'pat', 'lsq3', 'lsq6', 'lsq7', 'lsq9', 'lsq10', 'lsq12', 'procrustes']:
    #    raise ValueError("minctracc: invalid transform type %s" % conf.transform_type) # not needed if MinctraccConfig(Atom) given
    # FIXME this line should produce a MincAtom (or FileAtom?? doesn't matter if
    # we use only structural properties) with null mask/labels
    out_xfm = source.newname_with(lambda x: "%s_to_%s" % (x, target.name), ext='.xfm') #TODO destroy orig_name?
    #outf = MincAtom(name = "%s_to_%s.xfm" % (source.name, target.name))            #TODO don't get an orig_name
    stage = cmd_stage(flatten(
              ['minctracc', '-clobber', '-debug', '-xcorr'],
              #TODO remove -xcorr in nonlinear case?
              #TODO remove -debug in purely linear case?
              (['-transformation', InputFile(transform.path)] if transform else []),
              (['-' + conf.transform_type] if conf.transform_type else []),
              (['-use_simplex'] if conf.use_simplex is not None else []),
              # FIXME add -est_centre, -est_translations/-identity if not transform (else add transform) !!
              flatten(*map(lambda (f,v): [f, space_sep(v)], filter(lambda (f,v): v is not None, [
                ['-step',                  conf.step_sizes],
                ['-simplex',               conf.simplex],
                ['-tol',                   conf.tolerance],
                ['-w_shear',               conf.w_shear],
                ['-w_scales',              conf.w_scales],
                ['-w_rotations',           conf.w_rotations],
                ['-w_translations',        conf.w_translations],
                ['-iterations',            conf.iterations],
                ['-similarity_cost_ratio', conf.similarity],
                ['-lattice_diameter',      conf.lattice_diameter],
                ['-sub_lattice',           conf.sub_lattice],
               ]))),
              (['-nonlinear %s' % (conf.objective if conf.objective else '')] if conf.is_nonlinear else []),
              (['-source_mask', InputFile(source.mask.path)] if source.mask and conf.use_masks else []),
              (['-model_mask',  InputFile(target.mask.path)] if target.mask and conf.use_masks else []),
              [InputFile(source.path), InputFile(target.path), OutputFile(out_xfm.path)]))
    return Result(stages=Stages([stage]), output=out_xfm)

class SimilarityMetricConf(Atom):
    metric             = Str("CC")
    weight             = Float(1)
    blur_resolution    = Instance(float)
    radius_or_bins     = Int(3)
    use_gradient_image = Bool(False)

class MincANTSConf(Atom):
    iterations           = Str("100x100x100x150")
    transformation_model = Str("'Syn[0.1]'")
    regularization       = Str("'Gauss[2,1]'")
    use_mask             = Bool(True)
    sim_metric_params    = List(item=SimilarityMetricConf,
                                default = [SimilarityMetricConf(),
                                           SimilarityMetricConf(blur_resolution=0.056,
                                                                use_gradient_image=True)]) 
    
mincANTS_default_conf = MincANTSConf()

def mincANTS(source, target, conf, transform=None):
    s = Stages()
    # TODO add a way to add _nlin_0 or whatever to name based on generation or whatever
    xfm = source.newname_with(lambda x: "%s_to_%s" % (source.name, target.name), ext='.xfm') #TODO fix dir, orig_name, ...

    similaritycmds = []
    for similarity_element in conf.sim_metric_params:
        similaritycmds.append("-m")
        
        subcmd = ""
    for ix, st in enumerate(['img', 'grad']):
    # no coherence enforced between 2-elt arrays and blur/grad ... make protocol/conf better e.g. parse into set or nested structure
        if conf.blurs[ix] is not None:
            src  = s.defer(mincblur(source, fwhm=conf.blurs[ix], gradient=True))
            dest = s.defer(mincblur(target, fwhm=conf.blurs[ix], gradient=True))
        else:
            src  = source
            dest = target
        inner = ','.join([src.path, dest.path, str(conf.weight[ix]), str(conf.radius_or_histo[ix])])
        subcmds[st] = "'" + "".join([conf.similarity_metric[ix], '[', inner, ']']) + "'"
    # /hack
    stage = CmdStage(inputs = [source.path, target.path] + ([target.mask.path] if target.mask else []),
                     # hard to use cmd_stage wrapper here due to complicated subcmds ...
                     outputs = [xfm],
                     cmd = ['mincANTS', '3',
                            '--number-of-affine-iterations', '0',
                            '-m', subcmds['img'],
                            '-m', subcmds['grad'],
                            '-t', conf.transformation_model,
                            '-r', conf.regularization,
                            '-i', conf.iterations,
                            '-o', xfm.path] \
                         + (['-x', target.mask.path] if conf.use_mask and target.mask else []))
    s.add(stage)
    resampled = s.defer(mincresample(img=source, xfm=xfm, like=target, extra_flags=['-sinc']))
    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=xfm,
                                    resampled=resampled))

#def lsq12_NLIN_build_model(...):
#    raise NotImplemented

#def NLIN_build_model(imgs, initial_target, reg_method, nlin_dir, confs):
#    functions = { 'mincANTS'  : mincANTS_NLIN,
#                  'minctracc' : minctracc_NLIN }
#
#    function  = functions[reg_method]  #...[conf.nlin_reg_method] ???
#
#    return function(imgs=imgs, initial_target=initial_target, nlin_dir=nlin_dir, confs=confs)

'''
    
'''
def mincANTS_NLIN_build_model(imgs, initial_target, nlin_dir, confs):
    s = Stages()
    avg = initial_target
    avg_imgs = []
    for i, conf in enumerate(confs):
        xfms = [s.defer(mincANTS(source=img, target=avg, conf=conf)) for img in imgs]
        avg  = s.defer(mincaverage([xfm.resampled for xfm in xfms], name='nlin-%d' % i, output_dir='nlin'))
        avg_imgs.append(avg)
    return Result(stages=s, output=Registration(xfms=xfms, avg_img=avg, avg_imgs=avg_imgs))


def intrasubject_registrations(subj):
    # don't need if lsq12_nlin acts on a map with values being imgs
    # TODO temp configuration!!
    test_conf = MincANTSConf(iterations = "40x30x20",
                             transformation_model = mincANTS_default_conf.transformation_model,
                             regularization = "'Gauss[5,1]'",
                             similarity_metric = mincANTS_default_conf.similarity_metric,
                             weight = mincANTS_default_conf.weight,
                             blurs = mincANTS_default_conf.blurs,
                             radius_or_histo = [4,4],
                             gradient = mincANTS_default_conf.gradient,
                             use_mask = False)
    s = Stages()
    timepts = list((t,img) for t,img in subj.time_pt_dict.iteritems())
    for source_index in range(len(timepts) - 1):
        print(timepts[source_index][1])
        print(timepts[source_index + 1][1])
        xfms = [s.defer(mincANTS(source=timepts[source_index][1],
                                 target=timepts[source_index + 1][1],
                                 conf=test_conf))]
    return Result(stages=s, output=Registration(xfms=xfms, avg_img=None, avg_imgs=None))


#def multilevel_registration(source, target, registration_function, conf, curr_dir, transform=None):
#    ...

def multilevel_minctracc(source, target, conf, curr_dir, transform=None):
    # TODO fold curr_dir into conf?
    p = Stages()
    for conf in conf.single_gen_confs:
        # having the basic cmdstage fns act on single items rather than arrays is a bit inefficient,
        # e.g., we create many blur stages (which will later be eliminated, but still ...)
        src_blur  = p.defer(mincblur(source, conf.fwhm)) # TODO use conf.use_gradients
        dest_blur = p.defer(mincblur(target, conf.fwhm))
        transform = p.defer(minctracc(src_blur, dest_blur, conf=conf, transform=transform))
    return Result(stages=p,
                  output=XfmHandler(xfm=transform,
                                    source=src_blur,
                                    target=dest_blur,
                                    resampled=None))

#"""Multilevel registration of many images to a single target"""
#def multilevel_minctracc_all(sources, target, conf, resolutions, transforms=None):
#    p = Stages()
#    transforms = transforms or [None] * len(sources)
#    for res in resolutions:
#        ss_blurred = (p.add_stages(mincblur(s, res)) for s in sources)
#        t_blurred  = p.add_stages(mincblur(target, res))
#        transforms = [p.extract_stages(minctracc(s, t, conf, res, transform=t_blurred))
#                      for (s,t) in zip(ss_blurred, transforms)]
#    return Result(stages=p, output=transforms)


def multilevel_pairwise_minctracc(imgs, # list(MincAtom) 
                                  conf, 
                                  transforms=None, 
                                  like=None, 
                                  curr_dir="."): 
    """
    imgs -- should be array of MincAtoms
    conf --
    transforms -- 
    Pairwise registration of all images"""
    #check_valid_input_images(imgs)
    #if type(imgs) != list:
    #    raise something
    #for element in imgs:
    #    if element.__class__ != MincAtom
    #        raise something
    
    p = Stages()
    output_dir = os.path.join(curr_dir, 'pairs')
    def avg_xfm_from(src_img):
        """Compute xfm from src_img to each other img, average them, and resample along the result"""
        # TODO to save creation of lots of duplicate blurs, could use multilevel_minctracc_all,
        # being careful not to register the img against itself
        xfms = [p.defer(multilevel_minctracc(src_img, target_img, conf=conf, curr_dir=output_dir))
                for target_img in imgs if src_img != target_img]   # TODO src_img.name != ....name ??
        avg_xfm = p.defer(xfmaverage(xfms, output_dir=curr_dir))
        res  = p.defer(mincresample(img=src_img,
                                 xfm=avg_xfm,
                                 like=like or src_img,
                                 extra_flags=["-sinc"]))
        return XfmHandler(xfm = avg_xfm, source = src_img,
                          target = None, resampled = res) ##FIXME the None here borks things interface-wise ...
                                                          ## does putting `target = res` make sense? could a sum be used?
    return Result(stages=p, output=map(avg_xfm_from, imgs)) ##FIXME similarly, a list of xfmhs ... weird
    #xfms = map(xfm_to_avg, imgs)
    #avg_img = p.defer(mincaverage(xfms)
    #return Result(stages=p, output=Registration(xfms=xfms, average = avg_img))

MultilevelMinctraccConf = namedtuple('MultilevelMinctraccConf',
  ['resolution',       # used to choose step size...shouldn't be here
   'single_gen_confs', # list of minctracc confs for each generation; could fold res/transform_type into these ...
   'transform_type'])  # TODO add file_res ?

MinctraccConf = namedtuple('MinctraccConf', ['transform_type', 'w_translations', 'w_rotations'])

# TODO move LSQ12 stuff to an LSQ12 file
LSQ12_default_conf = MultilevelMinctraccConf(transform_type='lsq12', resolution = NotImplemented,
  single_gen_confs = [])

""" Pairwise lsq12 registration, returning array of transforms and an average image
    Assumption: given that this is a pairwise registration, we assume that all the input
                files to this function have the same shape (i.e. resolution, and dimension sizes.
                This allows us to use any of the input files as the likefile for resampling if
                none is provided. """
# TODO all this does is call multilevel_pairwise_minctracc and then return an average; fold into that procedure?
# TODO eliminate/provide default val for resolutions, move resolutions into conf, finish conf ...
def lsq12_pairwise(imgs, conf, lsq12_dir, like=None): #lsq12_dir = pipeline_dir_names.lsq12, like=None):
    output_dir = os.path.join(lsq12_dir, 'lsq12')
    #conf.transform_type='-lsq12' # hack ... copy? or set external to lsq12 call ? might be better
    p = Stages()
    xfms = p.defer(multilevel_pairwise_minctracc(imgs=imgs, conf=conf, like=like, curr_dir=output_dir))
    avg_img  = p.defer(mincaverage([x.resampled for x in xfms], output_dir=output_dir))
    return Result(stages = p, output = Registration(avg_imgs=[avg_img], avg_img=avg_img, xfms=xfms))

""" This is a direct copy of lsq12_pairwise, with the only difference being that
    that takes dictionaries as input for the imgs, and returns the xfmhandlers as
    dictionaries as well. This is necessary (amongst others) for the registration_chain
    as using dictionaries is the easiest way to keep track of the input files. """
def lsq12_pairwise_on_dictionaries(imgs, conf, lsq12_dir, like=None):
    l  = [(k,v) for k, v in sorted(imgs.iteritems())]
    ks = [k for k, _ in l]
    vs = [v for _, v in l]
    stages, outputs = lsq12_pairwise(imgs=vs, conf=conf, curr_dir=lsq12_dir, like=like)
    return Result(stages=stages, output=Registration(avg_imgs=outputs.avg_imgs,
                                                      avg_img=outputs.avg_img,
                                                      xfms=dict(zip(ks, outputs.xfms))))

def mincaverage(imgs, name="average", output_dir='.'):
    if len(imgs) == 0:
        raise ValueError("`mincaverage` arg `imgs` is empty (can't average zero files)")
    # TODO propagate masks/labels??
    avg = MincAtom(name = os.path.join(output_dir, '%s.mnc' % name), orig_name = None)
    #avg = imgs[0].copy_with(name = "average", orig_name = None); imgs.work_dir=...
    sdfile = MincAtom(name = os.path.join(output_dir, '%s_sd.mnc' % name), orig_name = None)
    #sdfile = outf.newname_with(fn = lambda x: x + '_sd')
    s = CmdStage(inputs=imgs, outputs=[avg, sdfile],
          cmd = ['mincaverage', '-clobber', '-normalize',
                 '-max-buffer-size-in-kb', '409620',
                 '-sdfile', sdfile.path] + [img.path for img in imgs] + [avg.path])
    return Result(stages=Stages([s]), output=avg)

def xfmaverage(xfms, output_dir): #mnc -> mnc
    if len(xfms) == 0:
        raise ValueError("`xfmaverage` arg `xfms` is empty (can't average zero files)")
    #outf  = xfms[0].xfm.newname_with(fn = lambda _ : 'average') # ugh
    #outf.orig_path = None #ugh
    outf  = MincAtom(name=os.path.join(output_dir, 'average'), orig_name=None)
    stage = CmdStage(inputs=xfms, outputs=[outf],
                     cmd=["xfmaverage"] + [x.xfm.path for x in xfms] + [outf.path])
    return Result(stages=Stages([stage]), output=outf)

def xfminvert(xfm): #mnc -> mnc #TODO find a new naming scheme for lifted/unlifted operations?
    inv_xfm = xfm.newname_with_suffix('_inverted')
    s = CmdStage(inputs=[xfm], outputs=[inv_xfm],
                 cmd=['xfminvert', '-clobber', xfm.path, inv_xfm.path])
    return Result(stages=Stages([s]), output=inv_xfm)

def invert(xfm): #xfm -> xfm
    """xfminvert lifted to work on XfmHandlers instead of MincAtoms"""
    s = Stages()
    inv_xfm = s.defer(xfminvert(xfm.xfm))
    return Result(stages=s, #FIXME add a resampling stage to get rid of null `resampled` field?
                  output=XfmHandler(xfm=inv_xfm, source=xfm.target, target=xfm.source, resampled=NotImplemented))

#June 30: Matthijs thinks we shouldn't always supply a resampled by default, since, e.g., lin_from_nlin resamplings
#would never be used.  But this means there will be null fields in most/all (if we don't generated
#resampled images even for staples like minctracc/mincANTS) xfmhandlers.  Perhaps this means that most stages shouldn't
#create xfmhs and only return xfms, and the user should create the xfmhandler when he/she believes this makes sense.
#add another type (xfhm/resxfmh)??
