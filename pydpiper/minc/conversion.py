import os

from pydpiper.core.stages import Stages, Result, CmdStage


def generic_converter(renamer, mk_cmd):
    def f(img):
        s = Stages()
        def run_cmd(i, o):
            c = CmdStage(inputs=(i,), outputs=(o,),
                          cmd = mk_cmd(i.path, o.path))
            # FIXME because nii2mnc doesn't know -clobber and generic_converter isn't generic enough:
            def h(s):
                try:
                    os.remove(o.path)
                except FileNotFoundError:
                    pass
            c.when_runnable_hooks.append(h)
            s.add(c)
        out_img = renamer(img)
        if img.mask:
            out_mask = renamer(img.mask)
            out_img.mask = out_mask
            run_cmd(img.mask, out_img.mask)
        if img.labels:
            out_labels = renamer(img.labels)
            out_img.labels = out_labels
            run_cmd(img.labels, out_img.labels)
        run_cmd(img, out_img)
        return Result(stages=s, output=out_img)
    return f