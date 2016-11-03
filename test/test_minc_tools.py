import pytest

from pydpiper.minc.registration import MincAtom, mincblur


# TODO factor out these fixtures common to several files

@pytest.fixture()
def img(mask, labels):
    return MincAtom('/images/img_1.mnc', pipeline_sub_dir='/scratch', mask=mask, labels=labels)

@pytest.fixture()
def mask():
    return MincAtom('/images/img_1_mask.mnc')

@pytest.fixture()
def labels():
    return MincAtom('/images/img_1_labels.mnc')

@pytest.fixture()
def img_blur_56um_result(img):
    return mincblur(img=img, fwhm=0.056)

@pytest.fixture()
def imgs():
    return [MincAtom('/images/img_%d.mnc' % i) for i in range(1,4)]

class TestMincblur():
    def test_output_naming(self, img, img_blur_56um_result):
        assert img_blur_56um_result.output.img.path == '/scratch/img_1/tmp/img_1_fwhm0.056_blur.mnc'
    def test_stage_creation(self, img, img_blur_56um_result):
        assert ([s.render() for s in list(img_blur_56um_result.stages)]
             == ['mincblur -clobber -no_apodize -fwhm 0.056 /images/img_1.mnc /scratch/img_1/tmp/img_1_fwhm0.056 -gradient'])

