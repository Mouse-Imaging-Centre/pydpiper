import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.registration_functions as rf
from pydpiper.pipeline import CmdStage

class SurfaceMask(CmdStage):
    """inputFile   : mnc file (handler)
       surfaceFile : obj file 
       outputFile  : mnc file (handler)"""
    def __init__(self, inputFile, surfaceFile, outputFile):
        CmdStage.__init__(self, None)
        self.name = "surface_mask2"
        # TODO choose a colour
        if isFileHandler(inputFile, outputFile):
            inf  =   inputFile.getLastBasevol()
            surf = surfaceFile.getLastBasevol()
            outf =  outputFile.getLastBasevol()
            self.inputFiles  = [inf, surf]
            self.outputFiles = [outf]
        else:
            self.inputFiles  = [inputFile, surfaceFile]
            self.outputFiles = [outputFile]
        self.cmd = ["surface_mask2", "-binary_mask"] + self.inputFiles + self.outputFiles

class TransformObjects(CmdStage):
    """inputFile  : obj file (handler)
       outputFile : obj file (handler)
       xfmFile    : xfm file"""
    def __init__(self, inputFile, outputFile, xfmFile):
        # transform_objects doesn't actually require 3rd arg but don't want to clobber input file
        CmdStage.__init__(self, None)
        self.name = "transform_objects"
        # TODO choose a colour
        if isFileHandler(inputFile, outputFile):
            inf  =  inputFile.getLastBasevol()
            outf = outputFile.getLastBasevol()
            xfm  =  inputFile.getLastXfm()
            self.inputFiles  = [inf, xfm]
            self.outputFiles = [outf]
        else:
            self.inputFiles  = [inputFile, xfmFile]
            self.outputFiles = [outputFile]
        self.cmd = ["transform_objects", inputFile, xfmFile] + self.outputFiles

class ReconstituteLaplacianGrid(CmdStage):
    def __init__(self, gridFile, midlineFile, cortexFile, outputFile):
        CmdStage.__init__(self, None)
        self.name = "reconstitute_laplacian_grid"
        # TODO colour
        self.inputFiles  = [gridFile, midlineFile, cortexFile]
        self.outputFiles = [outputFile]
        self.cmd = [self.name, midlineFile, cortexFile, gridFile, outputFile]

class LaplacianThickness(CmdStage):
    def __init__(self, inputFile, gridFile, outputFile, maxIterations):
        CmdStage.__init__(self, None)
        self.name = "laplacian_thickness"
        # TODO colour
        self.inputFiles  = [inputFile, File]
        self.outputFiles = [outputFile]
        self.cmd = [self.name, '-from_grid', gridFile, '-use_third_boundary,'
                    '-max_iterations', maxIterations, '-object_eval', inputFile, outputFile]

class Diffuse(CmdStage):
    def __init__(self, inputFile, inputAtlas, fwhm, outputFile, iterations=1000):
        CmdStage.__init__(self, None)
        self.name = "diffuse"
        # TODO colour
        self.inputFiles  = [inputFile, inputAtlas, fwhm]
        self.outputFiles = [outputFile]
        # TODO add kwarg for '-iterations'
        self.cmd = [self.name, '-kernel', fwhm, '-iterations', iterations,
                    inputAtlas, inputFile, outputFile]

