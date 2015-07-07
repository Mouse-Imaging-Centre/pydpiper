import pytest

from pydpiper.core.files import *

@pytest.fixture()
def f():
    return FileAtom('/path/to/a/file.ext')

class TestFileAtom():
    def test_file_name(self, f):
        assert f.name == 'file'
    def test_file_path(self, f):
        assert f.path == '/path/to/a/file.ext'
    def test_newname_immutable(self, f):
        f2 = f.newname_with(lambda x: x + '_new', ext='.new')
        assert f == f2
