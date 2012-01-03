"""factories for creating mincVolumes"""

from volumes import mincException,mincVolume

def volumeFromFile(filename, dtype="float"):
    """creates a new mincVolume from existing file"""
    v = mincVolume(filename, dtype)
    v.openFile()
    return(v)
    
def volumeFromInstance(volInstance, outputFilename, dtype="float", data=False,
                       dims=None, volumeType="ubyte"):
    """creates new mincVolume from another mincVolume"""
    v = mincVolume(outputFilename, dtype)
    v.copyDimensions(volInstance, dims)
    v.copyDtype(volInstance)
    v.createVolumeHandle(volumeType)
    if data:
        if not volInstance.dataLoaded:
            volInstance.loadData()
        v.createVolumeImage()  
        v.data = volInstance.data.copy()
    
    return(v)

def volumeLikeFile(likeFilename, outputFilename, dtype="float", volumeType="ubyte"):
    """creates a new mincVolume with dimension info taken from an existing file"""
    lf = volumeFromFile(likeFilename)
    v = volumeFromInstance(lf, outputFilename, dtype, volumeType=volumeType)
    lf.closeVolume()
    return(v)

def volumeFromDescription(outputFilename, dimnames, sizes, starts, steps, volumeType="ubyte",
                          dtype="float"):
    """creates a new mincVolume given starts, steps, sizes, and dimension names"""
    v = mincVolume(outputFilename, dtype)
    v.createNewDimensions(dimnames, sizes, starts, steps)
    v.createVolumeHandle(volumeType)
    v.createVolumeImage()
    return(v)