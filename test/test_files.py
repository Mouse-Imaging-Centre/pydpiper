import copy
import pytest

from pydpiper.core.files import FileAtom

@pytest.fixture()
def f():
    return FileAtom('/path/to/a/file.ext')

@pytest.fixture()
def g():
    return f()

class TestFileAtom():
    def test_file_name(self, f):
        assert f.filename_wo_ext == 'file'
    def test_file_path(self, f):
        assert f.path == '/path/to/a/file.ext'
    def test_newname_immutable(self, f):
        f2 = copy.deepcopy(f)
        f3 = f.newname_with_fn(lambda x: x + '_new', ext='.new')
        assert f == f2