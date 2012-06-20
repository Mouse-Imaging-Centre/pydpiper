#!/usr/bin/env python

from pyminc.volumes.factory import *
from numpy import *
from scipy.stats import *
from optparse import OptionParser

if __name__ == "__main__":

    usage = "Usage text"
    description = "Description text"
    
    parser = OptionParser(usage=usage, description=description)
    parser.add_option("--clobber", dest="clobber",
                      help="clobber output file",
                      type="string")

    (options, args) = parser.parse_args()

    if len(args) < 3:
        parser.error("Incorrect number of arguments")

    outfilename = args[-1]
    
    
    # clobber check should go here
    
    volhandles = []

    nfiles = len(args)-1
    for i in range( nfiles ):
        volhandles.append(volumeFromFile(args[i], dtype='ubyte'))

    outfile = volumeFromInstance(volhandles[0], outfilename)

    sliceArray = zeros( (nfiles,
                         volhandles[0].sizes[1],
                         volhandles[0].sizes[2]))
                         
    for i in range(volhandles[0].sizes[0]):
        for j in range(nfiles):
            t = volhandles[j].getHyperslab((i,0,0),
                                           (1,volhandles[0].sizes[1],
                                            volhandles[0].sizes[2]))
            t.shape = (volhandles[0].sizes[1], volhandles[0].sizes[2])
            sliceArray[j::] = t
        
        outfile.data[i::] = mode(sliceArray)[0]

    outfile.writeFile()
    outfile.closeVolume()


                                                          
    
