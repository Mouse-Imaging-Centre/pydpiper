# Stubs for pydpiper.core.arguments (Python 3.5)

from typing import Any, List
from configargparse import ArgParser  # type: ignore # TODO: make a stub for configargparse

class PydParser(ArgParser):
    epilog = ... # type: str
    def format_epilog(self, formatter): ...

def parse_nullable_int(string : str) -> int: ...

class Parser(object): ...

class BaseParser(Parser):
    argparser  = ... # type: ArgParser
    group_name = ... # type: str
    def __init__(self, argparser : ArgParser, group_name : str) -> None: ...

class CompoundParser(Parser):
    parsers = ...    # type: List[AnnotatedParser]
    def __init__(self, annotated_parsers : List[AnnotatedParser]) -> None: ...

class AnnotatedParser(object):
    parser    = ...  # type: Parser
    prefix    = ...  # type: str
    namespace = ...  # type: str
    cast      = ...  # type: Any
    def __init__(self, parser, namespace, prefix='', cast=None): ...

def parse(parser : Parser, args : List[str]): ...
def with_parser(p): ...

class RegistrationConf:
    input_space = ... # type: str
    resolution  = ... # type: float
    def __init__(self, input_space : str, resolution : float) -> None: ...

class LSQ6Conf:
    lsq6_method = ... # type: str
    # can't be enum if we're supporting Python 2
    def __init__(self, lsq6_method : str) -> None: ...

application_parser  = ... # type: Parser
execution_parser    = ... # type: Parser
registration_parser = ... # type: Parser
lsq6_parser  = ... # type: Parser
lsq12_parser = ... # type: Parser
stats_parser = ... # type: Parser
chain_parser = ... # type: Parser

