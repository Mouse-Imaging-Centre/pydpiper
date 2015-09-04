import shlex

from traits.api import *

from pydpiper.execution.pipeline import PipelineFile, InputFile, OutputFile

class CmdStage(object):
    """A simplified command stage - one could write a simple conversion
    function or simply adopt the old one.  I prefer separating static
    (command, memory limits) from dynamic (status, retries) information
    in part because it avoids many empty fields for stages which never
    become part of a `live` pipeline."""
    def __init__(self, inputs, outputs, cmd, memory=None):  #, conf, cmd):
        self.inputs  = inputs
        self.outputs = outputs
        #self.conf    = conf            # not needed at present -- see note on render_fn
        self._cmd    = cmd

        self.when_runnable_hooks = []  # TODO make the hooks accessible via the constructor
        self.when_finished_hooks = []
        self.memory = memory
    # NB: __hash__ and __eq__ ignore hooks, memory
    # Also, we assume cmd determines inputs, outputs
    def __hash__(self):
        return hash(str(self._cmd))
    def __eq__(self, c):
        return self._cmd == c._cmd
    # Originally I had `render_fn` : inputs, outputs, conf -> [str] instead of `cmd` : [str] to
    # (1) reduce duplication by not encoding the inputs/outputs in two places
    # (2) abstract away some of the boring parts, like getting the name fields out of atoms
    # the Input(...), Output(...) idea used in the old-style CmdStage is also OK, at the expense
    # of heterogeneous arrays (and problems with object equality/hashing/reading/printing/...)
    # Using a property, one could write stage.cmd instead of stage.cmd()
    #TODO: not very clear what render means?
    def render(self):
        #return self.render_fn(self.conf, self.inputs, self.outputs)
        return self.cmd_to_string()
    def cmd_to_string(self):
        return ' '.join(self._cmd)
    def to_array(self):
        """Form usable for Python subprocess call."""
        return self._cmd
    #def execute(self, backend):    # could also be elsewhere...
    #    raise NotImplemented

def cmd_stage(cmd): # [string|InputFile|OutputFile] -> CmdStage
    """'Smart constructor' for command stages using heterogeneous list API of the old command stage"""
    inputs  = [s.filename for s in cmd if isinstance(s, InputFile)]
    outputs = [s.filename for s in cmd if isinstance(s, OutputFile)]
    cmd = [(s.filename if isinstance(s, PipelineFile) else s) for s in cmd]
    return CmdStage(inputs = inputs, outputs = outputs, cmd = cmd)

class Stages(set):
    """A set of stages to be run.  In addition to the usual set operations,
    contains a single extra method, `defer`.  The idea here is as follows: procedures
    to create various commands and pipelines will return both
    (1) a set of stages to be run, and
    (2) a data structure representing the files created as a result.
    In PydPiper 1.x, (2) is done implicitly by modifying the procedures' inputs,
    alleviating the annoyance of handling two conceptually unrelated return values.
    One option would be to make (1) implicit by creating a global (i.e., module-level)
    Stages variable to which all stages would be added (though this can be rather
    mysterious, and loses the flexibility to, e.g., add stages to various separate pipelines,
    such as on both local/remote processors).  I currently opt to return both values, hence 
    `defer`, which accumulates the stages from a command and passes on the relevant files unchanged.

    The name is not too descriptive and I'm open to suggestions,
    but it's short since it will be called often.  Possible names: extract_stages,
    extract_stages_and_return_output_files, defer(red(_result)), output, ....
    Part of the problem is that this method really is somewhat silly (since
    two return values is silly as well).
    
    One could always just manually unpack the return value and avoid this method:
      xfms = p.defer(cmd(...)); ...xfms...
    is the same as
      stages, xfms = cmd(...)
      p.add(stages)
      ...xfms...

    Note: PydPiper 1.x uses a Pipeline where we use a Stages, but 
    we create many intermediate structures for which the extra fields of a pipeline
    don't have any meaning, so this might be worth changing for clarity.
    """
    def defer(self, result):
        self.update(result.stages)
        return result.output
    # FIXME this doesn't remember the order stages were added (see ordereddict/orderedset)
    # so they won't appear in the pipeline-stages.txt file in a nice order ...

# TODO make it possible to inline many inputs somehow (using cooperation from the string formatter?)
def parse(cmd_str):
    """Create a CmdStage object from a string.  (Per Jason's suggestion, we could make
    an even more clever parser that simply guesses the output based on position
    or a few flags like -o and tags anything else that looks like a file as input.)
    >>> val = 2
    >>> c = parse('mincfoo -flag {val} ,input1.mnc ,input2.mnc @output.mnc'.format(**vars()))
    >>> c.outputs
    ['output.mnc']
    >>> c.inputs
    ['input1.mnc', 'input2.mnc']
    """
    cmd = shlex.split(cmd_str)
    inputs  = [s[1:] for s in cmd if s[0] == ',']  #TODO what about I(), O() ?
    outputs = [s[1:] for s in cmd if s[0] == '@']
    s = CmdStage(inputs = inputs, outputs = outputs, cmd = [c if c[0] not in [',','@'] else c[1:] for c in cmd])
    return s
