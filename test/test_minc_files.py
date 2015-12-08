import pytest

from pydpiper.minc.files import MincAtom

@pytest.fixture()
def img():
    return MincAtom('/images/img_1.mnc')

@pytest.fixture()
def imgs():
    return [MincAtom('/images/img_%d.mnc' % i) for i in range(1,4)]

# these shouldn't duplicate the FileAtom tests, but can we automatically rerun those here?
class TestMincAtom():
    # TODO add mask and labels fixtures, with some tests
    pass
    #def test_ext(self, img):
    #   assert f.newname_with_fn(lambda x)
