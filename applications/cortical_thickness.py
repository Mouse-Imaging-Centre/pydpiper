from os.path import abspath
from pydpiper.application import AbstractApplication
from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import configargparse
import logging

logger = logging.getLogger(__name__)

def addCorticalThicknessOptionGroup(parser):
    group = parser.add_argument_group("Cortical thickness options",
                                      "Options for controlling the cortical thickness determination.")
    group.add_argument("-transform", dest="init_transform_file", type=str,
                       help="Initial transform from c57bl6 atlas space to study's non-linear atlas")
    group.add_argument("-grid", dest="grid_file", type=str,
                       help="MINC volume containing inside and outside boundary definitions.")
    group.add_argument("-cortex", dest="cortex_file", type=str,
                       help="OBJ file containing cortex definition.")
    group.add_argument("-grid_fwhm", dest="grid_fwhm", type=float, default=0.06,
                       help="Blurring kernel to be used when resampling grid.")
    group.add_argument("-fwhm", dest="fwhm", type=float, default=0.5,
                       help="Blurring kernel to be used on thickness map.")
    group.add_argument("-max_iterations", dest="max_iterations", type=int, default=200,
                       help="Maximum number of iterations in relaxation.")

class CorticalThicknessApplication(AbstractApplication): # TODO need a better name  # ~ LSQ6Reg
    def setup_options(self):
        addCorticalThicknessOptionGroup(self.parser)
        rf.addGenRegArgumentGroup(self.parser)
        # TODO set a different usage message??

    def setup_appName(self):
        return "CorticalThickness"

    def run(self):
        options    = self.options
        args       = self.args
        basedir    = args[0]
        transforms = args[1:]

        dirs = rf.setupDirectories(self.outputDir, options.pipeline_name, module="CorticalThickness")

        input_files = [options.grid_file, options.cortex_file] + transforms
        if options.init_transform_file is not None:
            input_files.append(options.init_transform_file)

        # TODO this will fail as not all input_files are MINC files
        inputFHs = rf.initializeInputFiles(input_files, dirs.processedDir) # TODO add surface_dir=options.surface_dir??

        targetPipeFH = rfh.RegistrationPipeFH(filename=abspath(output_prefix), basedir=self.outputDir)

        self.pipeline.addPipeline(CorticalThicknessPipeline(xfmFile=input_transform,
                                                            gridFile=options.grid_file,
                                                            cortexFile=options.cortex_file,
                                                            targetPipeFH=targetPipeFH,
                                                            options=options).p)
        # TODO handle options.init_transform_file

class CorticalThicknessPipeline(): # TODO need a better name    # ~ LSQ6Base(object)
    """ """
    def __init__(self, xfmFile, gridFile, cortexFile, targetPipeFH, options):
        CmdStage.__init__(self, None)
        self.options = options
        self.p = Pipeline()
        self.inputFiles = [xfmFile, gridFile, cortexFile]
        
        surfaceMaskStage = SurfaceMask(gridFile, cortexFile, targetPipeFH)
        p.addPipeline(surfaceMaskStage.p)

if __name__ == "__main__":
    application = CorticalThicknessApplication()
    application.start()
