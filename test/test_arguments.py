import pytest
from configargparse import Namespace

from pydpiper.core.arguments import CompoundParser, AnnotatedParser, application_parser, parse
from pydpiper.pipelines.MBM  import mk_mbm_parser


@pytest.fixture
def two_mbm_parser():
    mbm_parser = mk_mbm_parser(with_common_space=False)
    return CompoundParser([AnnotatedParser(parser=mbm_parser, prefix="mbm1", namespace="mbm1"),
                           AnnotatedParser(parser=mbm_parser, prefix="mbm2", namespace="mbm2")])


@pytest.fixture
def four_mbm_parser(two_mbm_parser):
    return CompoundParser([AnnotatedParser(parser=two_mbm_parser, prefix="first-two", namespace="first_two"),
                           AnnotatedParser(parser=two_mbm_parser, prefix="last-two", namespace="last_two")])


@pytest.fixture()
def two_mbm_parse(two_mbm_parser):
    return parse(two_mbm_parser, ["--mbm1-lsq12-max-pairs=22", "--mbm1-bootstrap", "--mbm2-bootstrap"])


@pytest.fixture()
def four_mbm_parse(four_mbm_parser):
    return parse(four_mbm_parser, ["--first-two-mbm1-lsq12-max-pairs=22",
                                   "--first-two-mbm2-lsq12-max-pairs", "23",
                                   "--last-two-mbm1-lsq12-max-pairs=24",
                                   "--last-two-mbm2-lsq12-max-pairs=25",
                                   "--first-two-mbm1-bootstrap", "--first-two-mbm2-bootstrap",
                                   "--last-two-mbm1-bootstrap", "--last-two-mbm2-bootstrap"])

@pytest.fixture()
def application_parse(two_mbm_parser):
    return parse(CompoundParser([application_parser,
                                 AnnotatedParser(parser=two_mbm_parser, prefix="two-mbms", namespace="two-mbms")]),
                 ["--two-mbms-mbm1-bootstrap", "--two-mbms-mbm2-bootstrap",
                  "--two-mbms-mbm1-lsq12-max-pairs", "23", "--two-mbms-mbm2-lsq12-max-pairs=24", "--file", "img_1.mnc"])


def is_recursive_subnamespace(n1, n2):
    """Is n1 a recursive substructure (not a subset!) of n2?"""
    return all((f in n2.__dict__ and
                  (n1.__dict__[f] == n2.__dict__[f] or
                   (type(n1.__dict__[f]) == type(n2.__dict__[f]) == Namespace and
                    #   isinstance(n1.__dict__[f], Namespace) and isinstance(n1.__dict__[f], Namespace) and
                                                      is_recursive_subnamespace(n1.__dict__[f], n2.__dict__[f])))
                for f in n1.__dict__))

class TestArgumentParsing:
    def test_nested_parsing(self, two_mbm_parse):
        assert is_recursive_subnamespace(Namespace(mbm1=Namespace(lsq12=Namespace(max_pairs=22)),
                                                   mbm2=Namespace(lsq12=Namespace(max_pairs=25))),
                                         two_mbm_parse)
    def test_deeper_nesting(self, four_mbm_parse):
        assert is_recursive_subnamespace(Namespace(first_two=Namespace(mbm1=Namespace(lsq12=Namespace(max_pairs=22)),
                                                                       mbm2=Namespace(lsq12=Namespace(max_pairs=23))),
                                                   last_two=Namespace(mbm1=Namespace(lsq12=Namespace(max_pairs=24)),
                                                                      mbm2=Namespace(lsq12=Namespace(max_pairs=25)))),
                                         four_mbm_parse)
    def test_with_files(self, application_parse):
        assert is_recursive_subnamespace(Namespace(application=Namespace(files=["img_1.mnc"])),
                                         application_parse)
