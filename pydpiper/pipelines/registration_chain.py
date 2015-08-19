
import csv
from collections import defaultdict

from atom.api import Atom, Int, Str, Dict, Enum, Instance
#import atom.api as atom

from pydpiper.minc.analysis import determinants_at_fwhms
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
    csv_file               = Str(None)

class Subject(Atom):
    intersubject_registration_time_pt = Instance(int)
    time_pt_dict   = Dict()    # validation (key=Int, value=Str) doesn't work? ...

    def __eq__(self, other):
        return (self is other or
                (self.intersubject_registration_time_pt == other.intersubject_registration_time_pt
                 and self.time_pt_dict == other.time_pt_dict
                 and self.__class__ == other.__class__))
    # ugh; also, should this be type(self) == ... ?

    def get_intersubject_registration_image(self):
        return self.time_pt_dict[subject.intersubjection_registration_time_pt]

    intersubject_registration_image = property(get_intersubject_registration_image,
                                               'intersubject_registration_image property')
    
# could be a method/property; unsure how well this works with Traits/Atom
def intersubject_registration_image(subject):
    return subject.time_pt_dict[subject.intersubjection_registration_time_pt]

class TimePointError(Exception):
    pass
                 
def map_data(f, d): # TODO find a new name for this
    # TODO this is probably too big to be a doctest ...
    """Map `f` non-destructively (if `f` is) over (the values of)
    the inner time_pt_dict of a { subject : Subject }
    
    >>> (map_data(lambda x: x[3],
    ...          { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict={3:'s1_3.mnc', 4:'s1_4.mnc'}),
    ...            's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict={4:'s2_4.mnc', 5:'s2_5.mnc'})} )
    ...   == { 's1' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {3:'3',4:'4'}),
    ...        's2' : Subject(intersubject_registration_time_pt=4, time_pt_dict= {4:'4',5:'5'}) })
    True
    """
    new_d = {}
    for s_id, subj in d.iteritems():
        new_time_pt_dict = {}
        for t,x in subj.time_pt_dict.iteritems():
            new_time_pt_dict[t] = f(x)
        new_subj = Subject(intersubject_registration_time_pt = subj.intersubject_registration_time_pt,
                           time_pt_dict   = new_time_pt_dict)
        new_d[s_id] = new_subj
    return new_d

def parse_common(string):
    truthy_strings = ['1','True','true','T','t']
    falsy_strings  = ['','0','False','false','F','f']
    def fmt(strs):
        return "'" + "','".join(strs) + "'"
    string = string.strip()
    if string in truthy_strings:
        return True
    elif string in falsy_strings:
        return False
    else:
        raise ValueError('Unrecognized value %s; ' % string
                         + 'Please use one of ' + fmt(truthy_strings)
                         + ' in the "is_common" field of your csv file ' 
                         + 'to use this file for intersubject registration, or '
                         + 'one of ' + fmt(falsy_strings) + 'to specify otherwise')

# TODO standardize on pt/point
# TODO write some longer (non-doc)tests
def parse_csv(rows, common_time_pt): # row iterator, int -> { subject_id(str) : Subject }
    """
    Read subject information from a csv file containing at least the columns
    'subject_id', 'timepoint', and 'filename', and optionally a 'bitfield' column
    'is_common' containing one 1 per subject and 0 or empty fields for the other scans.
    Return a map from subject IDs to a dict of timepoints and a specific timepoint
    to be used for inter-subject registration.

    >>> csv_data = "subject_id,timepoint,filename,genotype\\ns1,1,s1_1.mnc,1\\n".split('\\n')
    >>> (parse_csv(csv_data, common_time_pt=1)
    ...   == { 's1' : Subject(intersubject_registration_time_pt=1, time_pt_dict={ 1 : 's1_1.mnc' })})
    True
    """
    subject_info = defaultdict(Subject)
    # Populate the subject -> Subject dictionary from the rows"""
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
            if parse_common(row.get('is_common', '')):
                if subject_info[subj_id].intersubject_registration_time_pt is not None:
                    raise TimePointError(
                        "duplicate common time point specified for subject '%s'"
                        % subj_id)
                else:
                    subject_info[subj_id].intersubject_registration_time_pt = timepoint
    # could make this part into a separate fn that copies input, returns updated version:
    # Iterate through subjects, filling in intersubject-registration time points with the common
    # time point if unspecified for a given subject, and raising an error if there's no timepoint
    # available or no scan for the specified timepoint
    for s_id, s in subject_info.iteritems():
        if s.intersubject_registration_time_pt is None:
            if common_time_pt is None:
                raise TimePointError("no subject-specific or default inter-subject "
                                     "time point provided for subject '%s'" % s_id)
            elif common_time_pt in s.time_pt_dict:
                s.intersubject_registration_time_pt = common_time_pt
            elif common_time_pt == -1:
                s.intersubject_registration_time_pt = s.time_pt_dict[max(s.time_pt_dict.keys())]
            else:
                raise TimePointError("subject '%s' didn't have a scan for "
                                     "the common time point specified (%d); "
                                     "fix this or specify a different timepoint "
                                     "for this subject by putting a value in an"
                                     "'is_common' column of your table"
                                     % (s_id, common_time_pt))
        else:
            if common_time_pt != s.intersubject_registration_time_pt:
                print('note: overriding common_time_pt %d with time point %d for subject %s'
                      % (common_time_pt, s.intersubject_registration_time_pt, s_id))
                    
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


def chain(options):

    s = Stages()
    
    with open(options.csv_file, 'r') as f:
        subject_info = parse_csv(f, options.common_time_point)
    
    if options.input_space == 'native':
        raise NotImplemented
    elif options.input_space not in ['lsq6', 'lsq12']:
        raise ValueError('unrecognized input space: %s; choices: %s' % ())
    

    
    
    #all_imgs = {(s_id,t):img for s_id, subj in subject_info.iteritems()
    #            for (t,img) in subj.time_point_dict.iteritems()}

    # NB currently LSQ6 expects an array of files, but we have a map.
    # possibilities:
    # - note that pairwise is enough (except for efficiency -- redundant blurring, etc.)
    #   and just use the map fn above with an LSQ6 fn taking only a single source
    # - rewrite LSQ6 to use such a (nested) map
    # - write conversion which creates a tagged array from the map, performs LSQ6,
    #   and converts back
    # - write 'over' which takes a registration, a data structure, and 'get/set' fns ...?

    #all_imgs = [img for subj in subject_info.itervalues()
    #            for img in subj.time_point_dict.itervalues()]

    # TODO how to associate images in the above dict with their xfm ??
    # put result of LSQ6 into a map img_name => xfm
    #lsq6_xfms = s.defer(LSQ6(all_imgs, options.lsq6_conf))


    # LSQ12/NLIN registration of common-timepoint images:
    
    
    #{ xfm.source : xfm for xfm in lsq6_xfms}

    ## intersubject registration
    intersubj_imgs = { s_id : subj.intersubject_registration_image
                       for s_id, subj in subject_info.iteritems() }

    conf = ....
    lsq12_directory = ... {pipename}_{common_name}_lsq12

    intersubj_xfms = lsq12_NLIN_build_model_on_dictionaries(imgs=intersubj_imgs,
                                                            conf=conf,
                                                            lsq12_dir=lsq12_directory
                                                            #, like={atlas_from_init_model_at_this_tp}
                                                            )
    ## within-subject registration
    def intrasubject_registrations(subj):
        # don't need if lsq12_nlin acts on a map with values being imgs
        #timepts = sorted(((t,img) for t,img in subj.time_pt_dict.iteritems()))
        timepts = subject_info[subj]
        raise NotImplemented
    
    chain_xfms = { s_id : s.defer(intrasubject_registrations(subj))
                   for s_id, subj in subject_info.iteritems() }
    

    # TODO n

    ## longitudinal registration
    #for subj_id, subj in subject_info.iteritems():
    #    pass

    map_data(lambda xfm: determinants_at_fwhms(xfm, options.fwhms), subject_info)
    
    raise NotImplemented


