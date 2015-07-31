
import csv
from collections import defaultdict

from atom.api import Atom, Int, Str, Dict, Enum
#import atom.api as atom

from pydpiper.minc.registration import Stages
from pydpiper.pipelines.LSQ6 import lsq6

# TODO (general for all option records, not just for the registration chain):
# namedtuples are better than Argparse Namespaces for specification 
# (more descriptive name -- other benefits?) and
# for being able to get the data back out again (used in PP for filtering & re-sending
# arguments via submit scripts, which is the wrong approach, but ...),
# but they don't have the built-in type/arity checking or monoid-ish operations
# of Argparse, so we should create our own type which can be sent to both ??

# ALSO it's unclear whether our attempt at 'nominal' typing is a good idea, since, e.g.,
# a chain calls LSQ6, so will have to have lsq6 options as a subset of its own options
# ... so we could have a nested record, but this might not be best if both the chain
# and lsq6 need a particular option ... (also executor options, ...) ??

class ChainConf(Atom):
    input_space            = Enum('native', 'lsq6', 'lsq12')
    common_time_point      = Int(None)
    common_time_point_name = Str("common")
    #field_names            = Str("subject,time,filename,is_common")

class Subject(Atom):
    common_time_pt = Int(None) # FIXME should be named registration_time_pt or similar
    time_pt_dict   = Dict()    # FIXME validation (key=Int, value=Str) doesn't work? ...

    def __eq__(self, other):
        return (self is other or
                (self.common_time_pt == other.common_time_pt
                 and self.time_pt_dict == other.time_pt_dict
                 and self.__class__ == other.__class__)) # ugh

class TimePointError(Exception):
    pass

# TODO need to mincify the filename ... but where will we have appropriate knowledge
# of directory, etc. ??
# TODO should timepts be ints? floats? ...
# TODO should read_csv really take common_time_pt arg?  seems OK
# TODO what should happen if common_time_pt 
def parse_csv(rows, common_time_pt): # row iterator, int -> { subject_id(str) : Subject }
    """
    Read subject information from a csv file containing at least the columns
    'subject_id', 'timepoint', and 'filename', and optionally a 'bitfield' column
    'is_common' containing one 1 per subject and 0 or empty fields for the other scans.
    Return a map from subject IDs to a dict of timepoints and a specific timepoint
    to be used for inter-subject registration.

    >>> csv_data = "subject_id,timepoint,filename,wt\\ns1,1,'s1_1.mnc,1'\\n".split('\\n')
    >>> parse_csv(csv_data) == { 'subject_id ' : Subject(1, { 1 : 's1_1.mnc' })}
    """
    subject_info = defaultdict(Subject)
    for row in csv.DictReader(rows):
        try:
            subj_id   = row['subject_id']
            timepoint = int(row['timepoint'])
            filename  = row['filename']
        except KeyError as e:
            raise KeyError("csv file must contain at least "
                           "'subject_id', 'timepoint', 'filename' fields; "
                           "missing: %s" % e.message)
        else:
            subject_info[subj_id].time_pt_dict[timepoint] = filename
            if row.get('is_common_field', None):
                if subject_info[subj_id].common_time_pt is not None:
                    raise TimePointError(
                        "duplicate common time point specified for subject '%s'"
                        % subj_id)
                else:
                    subject_info[subj_id].common_time_pt = timepoint
    # could make this part into a separate fn that copies input, returns updated version:
    for s_id, s in subject_info.iteritems():
        if s.common_time_pt is None:
            if common_time_pt is None:
                raise TimePointError("no subject-specific or default inter-subject "
                                     "time point provided for subject '%s'" % s_id)
            elif common_time_pt in s.time_pt_dict:
                s.common_time_pt = common_time_pt
            else:
                raise TimePointError("subject '%s' didn't have a scan for "
                                     "the common time point specified (%d); "
                                     "fix this or specify a different timepoint "
                                     "for this subject by putting a value in an"
                                     "'is_common' column of your table"
                                     % (s_id, common_time_pt))
        else:
            # At this point, we are overriding common_time_pt (if specified)
            # with a (valid, since a relevant row exists!) time point for
            # this specific subject, so everything is fine (though perhaps
            # a warning?)
            pass
                    
    return subject_info
    
# NOTE I've moved the optional lsq6 stuff outside this function to promote re-use;
# actual call could look something like this:
#def chain_with_optional_lsq6(inputs, options):
#    def native():
#        pass
#    def lsq6():
#        pass
#
#    fns = { 'native' : native, 'lsq6' : lsq6 }
#
#    try:
#        f = fns[options.input_space]
#    except KeyError:
#        raise ValueError("illegal input space: %s; allowed options: %s" % \
#                         (options.input_space, ','.join(map(str,fns.keys()))))
#    # call f...


def chain(subject_info, options):
    # { subject : Subject }, RegistrationChainConf -> stages, ...

    s = Stages()
    
    #all_imgs = {(s_id,t):img for s_id, subj in subject_info.iteritems()
    #            for (t,img) in subj.time_point_dict.iteritems()}

    all_imgs = [img for subj in subject_info.itervalues()
                for img in subj.time_point_dict.itervalues()]

    # TODO how to associate images in the above dict with their xfm ??
    # put result of LSQ6 into a map img_name => xfm
    lsq6_xfms = s.defer(LSQ6(all_imgs, options.lsq6_conf))
    
    { xfm.source : xfm for xfm in lsq6_xfms}
    
    for subj_id, subj in subject_info.iteritems():
        pass
        
    #TODO input file/registration method checking that isn't done in reading fn

    raise NotImplemented


