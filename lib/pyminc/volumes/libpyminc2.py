#/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
from numpy import *
import sys

# load the library
#minclocation = find_library("minc2")
#print "PYMINC: using", minclocation

try:
    libminc = cdll.LoadLibrary("libminc2.dylib")
except OSError:
    try:
        libminc = cdll.LoadLibrary("libminc2.so")
    except OSError:
        sys.stderr.write("ERROR: Neither libminc2.so nor libminc2.dylib found on search path\n.")
        sys.exit(3)


# sizes used by MINC and numpy
# mincSizes contains all acceptable MINC datatype sizes. Each item has
# four dictionary elements:
# "minc"  -> the integer value used by get_hyperslab, etc.
# "numpy" -> the dtype string to be used to create a numpy array of that size
# "ctype" -> the ctypes function corresponding to that datatype
# "min"   -> the minimum value the integer can contain
# "max"   -> the maximum value the integer can contain
minSigned = lambda x: -2**x/2
maxSigned = lambda x: (2**x/2)-1
maxUnsigned = lambda x: (2**x)-1
mincSizes = {}
mincSizes["byte"] = {"minc": 1, "numpy": "int8", "ctype": c_byte,
		     "min": minSigned(8), "max": maxSigned(8),
		     "type": "normalized"}
mincSizes["short"] = {"minc": 3, "numpy": "int16", "ctype": c_short,
		      "min": minSigned(16), "max": maxSigned(16),
		      "type": "normalized"}
mincSizes["int"] = {"minc": 4, "numpy": "int32", "ctype": c_int,
		    "min": minSigned(32), "max": maxSigned(32),
		    "type": "normalized"}
mincSizes["float"] = {"minc": 5, "numpy": "float32", "ctype": c_float,
		      "type": "real"}
mincSizes["double"] = {"minc": 6, "numpy": "float64", "ctype": c_double,
		       "type": "real"}
mincSizes["ubyte"] = {"minc": 100, "numpy": "uint8", "ctype": c_ubyte,
		      "min": 0, "max" : maxUnsigned(8),
		      "type": "normalized"}
mincSizes["ushort"] = {"minc": 101, "numpy": "uint16", "ctype": c_ushort,
		       "min": 0, "max": maxUnsigned(16),
		       "type": "normalized"}
mincSizes["uint"] = {"minc": 102, "numpy": "unit32", "ctype": c_uint,
		     "min": 0, "max": maxUnsigned(32),
		     "type": "normalized"}

# some typedef definitions
MI_DIMCLASS_ANY = c_int(0)
MI_DIMCLASS_SPATIAL = c_int(1)
MI_DIMCLASS_RECORD = c_int(6)
MI_DIMATTR_ALL = c_int(0)
MI_DIMATTR_REGULARLY_SAMPLED = c_int(1)


MI_DIMORDER_FILE = c_int(0)
MI_DIMORDER_APPARENT = c_int(0)

MI_TYPE_DOUBLE = c_int(6)
MI_TYPE_UBYTE = c_int(100)
MI_CLASS_REAL = c_int(0)

# opaque minc structs can be represented as pointers
mihandle = c_void_p
midimhandle = c_void_p

# some type information
dimensions = c_void_p * 5
voxel = c_double
location = c_ulong * 5
int_sizes = c_int * 5
long_sizes = c_ulong * 5
double_sizes = c_double * 5

# argument declarations - not really necessary but does make
# segfaults a bit easier to avoid.
libminc.miopen_volume.argtypes = [c_char_p, c_int, POINTER(mihandle)]
libminc.miget_real_value.argtypes = [mihandle, location, c_int, POINTER(voxel)]
libminc.miget_volume_dimensions.argtypes = [mihandle, c_int, c_int, c_int,
					    c_int, dimensions]
libminc.miget_dimension_sizes.argtypes = [dimensions, c_int, int_sizes]
libminc.miget_dimension_name.argtypes = [c_void_p, POINTER(c_char_p)]
libminc.miget_dimension_separations.argtypes = [dimensions, c_int, c_int, 
											    double_sizes]
libminc.miget_dimension_starts.argtypes = [dimensions, c_int, c_int,
										   double_sizes]
										  
#libminc.miget_real_value_hyperslab.argtypes = [mihandle, c_int, long_sizes,
#					       long_sizes, POINTER(c_double)]
libminc.micopy_dimension.argtypes = [c_void_p, POINTER(c_void_p)]
libminc.micreate_volume.argtypes = [c_char_p, c_int, dimensions, c_int, c_int,
				    c_void_p, POINTER(mihandle)]
libminc.micreate_volume_image.argtypes = [mihandle]
libminc.miset_volume_valid_range.argtypes = [mihandle, c_double, c_double]
libminc.miset_volume_range.argtypes = [mihandle, c_double, c_double]
libminc.miget_volume_range.argtypes = [mihandle, POINTER(c_double), POINTER(c_double)]
#libminc.miset_real_value_hyperslab.argtypes = [mihandle, c_int, long_sizes,
#					       long_sizes, POINTER(c_double)]
libminc.miclose_volume.argtypes = [mihandle]
libminc.miget_volume_dimension_count.argtypes = [mihandle, c_int, c_int,
						 POINTER(c_int)]
libminc.mifree_dimension_handle.argtypes = [c_void_p]
libminc.micreate_dimension.argtypes = [c_char_p, c_int, c_int, c_uint, POINTER(c_void_p)]
libminc.miset_dimension_separation.argtypes = [c_void_p, c_double]
libminc.miset_dimension_start.argtypes = [c_void_p, c_double]
