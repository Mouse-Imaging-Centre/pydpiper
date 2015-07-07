import pytest

from pydpiper.minc.files import *

@pytest.fixture()
def img():
    return MincAtom('/images/img_1.mnc')

@pytest.fixture()
def imgs():
    return [MincAtom('/images/img_%d.mnc' % i) for i in range(1,4)]

class TestMincAtom():
    # TODO add mask and labels fixtures, with some tests
    def test_name(self, img):
        assert img.name == 'img_1'
    def test_path(self, img):
        assert img.path == '/images/img_1.mnc'
    def test_newname_immutable(self, img):
        img2 = img.newname_with(lambda x: x + '_new', ext='.new')
        assert img == img2
    def test_images(self, imgs):
        pass #assert 0 == imgs
