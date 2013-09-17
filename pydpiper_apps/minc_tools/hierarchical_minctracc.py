from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import sys

"""LinearHierarchicalMinctracc and RotationalMinctracc are currently broken/not fully tested."""

class LinearHierarchicalMinctracc:
    """Default LinearHierarchicalMinctracc class
       Assumes lsq6 registration using the identity transform"""
    def __init__(self, 
                 inputPipeFH, 
                 templatePipeFH,
                 blurs=[1, 0.5, 0.3]):
        
        self.p = Pipeline()
        self.inputPipeFH = inputPipeFH
        self.templatePipeFH = templatePipeFH
        
        self.blurFiles(blurs)
        
    def blurFiles(self, blurs):
        for b in blurs:
            if b != -1:
                tblur = ma.blur(self.templatePipeFH, b, gradient=True)
                iblur = ma.blur(self.inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)

class RotationalMinctracc(CmdStage):
    """
        This class runs a rotational_minctracc.py call on its two input 
        files.  That program performs a 6 parameter (rigid) registration
        by doing a brute force search in the x,y,z rotation space.  Normally
        the input files have unknown orientation.
        
        The input files are assumed to have already been blurred appropriately
        
        There are a number of parameters that have to be set and this 
        will be done using factors that depend on the resolution of the
        input files.  Here is the list:
        
        argument to be set   --  default factor  -- (for 56 micron, translates to)
                blur                  10                        (560 micron)
          resample stepsize            4                        (224 micron)
        registration stepsize         10                        (560 micron)
          w_translations               8                        (448 micron)
         
        Specifying -1 for the blur argument will not perform any blurring.
        The two other parameters that can be set are (in degrees) have defaults:
        
            rotational range          50
            rotational interval       10
        
        Whether or not a mask will be used is based on the presence of a mask. 
    """
    def __init__(self, 
                 inSource, 
                 inTarget,
                 output = None, # ability to specify output transform when using strings for input
                 logFile = None,
                 defaultDir="transforms",
                 blur=10,
                 resample_step=4,
                 registration_step=10,
                 w_translations=8,
                 rotational_range=50,
                 rotational_interval=10):
        
        CmdStage.__init__(self, None) #don't do any arg processing in superclass
        self.name   = "rotational-minctracc"
        self.colour = "green"

        highestResolution = rf.getHighestResolution(inSource)
        adjustedBlur = blur
        if(adjustedBlur != -1):
            adjustedBlur = adjustedBlur * highestResolution
        
        # handling of the input files
        try: 
            if rf.isFileHandler(inSource, inTarget):
                self.source = inSource.getBlur(fwhm=adjustedBlur)
                self.target = inTarget.getBlur(fwhm=adjustedBlur)
                self.inputFiles = [self.source, self.target] 
                self.output = inSource.registerVolume(inTarget, defaultDir)
                self.outputFiles = [self.output]
                self.logFile = fh.logFromFile(inSource.logDir, self.output)
            else:
                self.source = inSource
                self.target = inTarget
        except:
            print "Failed in putting together RotationalMinctracc command."
            print "Unexpected error: ", sys.exc_info()
        
        #self.blurFiles(adjustedBlur)
        self.buildCmd(resample_step     * highestResolution,
                      registration_step * highestResolution,
                      w_translations    * highestResolution,
                      int(rotational_range),
                      int(rotational_interval))

    def buildCmd(self,
                 resamp_step,
                 reg_step,
                 w_trans,
                 rot_range,
                 rot_interval):
        
        w_trans_string = str(w_trans) + ',' + str(w_trans) + ',' + str(w_trans)
        cmd = ["rotational_minctracc.py", 
               "-t", "/dev/shm/", 
               "-w", w_trans_string,
               "-s", str(resamp_step),
               "-g", str(reg_step),
               "-r", str(rot_range),
               "-i", str(rot_interval),
               self.source,
               self.target,
               self.output,
               "/dev/null"]
        
        #mask = self.target.getMask()
        #if mask:
        #    cmd += ["-m", InputFile(mask)]
        self.cmd = cmd
        

class HierarchicalMinctracc:
    """Default HierarchicalMinctracc currently does:
        1. 2 lsq12 stages with a blur of 0.25
        2. 5 nlin stages with a blur of 0.25
        3. 1 nlin stage with no blur"""
    def __init__(self, 
                 inputPipeFH, 
                 templatePipeFH,
                 steps=[1,0.5,0.5,0.2,0.2,0.1],
                 blurs=[0.25,0.25,0.25,0.25,0.25, -1], 
                 gradients=[False, False, True, False, True, False],
                 iterations=[60,60,60,10,10,4],
                 simplexes=[3,3,3,1.5,1.5,1],
                 w_translations=0.2,
                 linearparams = {'type' : "lsq12", 'simplex' : 1, 'step' : 1}, 
                 defaultDir="tmp"):
        
        self.p = Pipeline()
        
        for b in blurs:
            #MF TODO: -1 case is also handled in blur. Need here for addStage.
            #Fix this redundancy and/or better design?
            if b != -1:
                tblur = ma.blur(templatePipeFH, b, gradient=True)
                iblur = ma.blur(inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
            
        # Do standard LSQ12 alignment prior to non-linear stages 
        lsq12 = mm.LSQ12(inputPipeFH, 
                         templatePipeFH, 
                         defaultDir=defaultDir)
        self.p.addPipeline(lsq12.p)
        
        # create the nonlinear registrations
        for i in range(len(steps)):
            """For the final stage, make sure the output directory is transforms."""
            if i == (len(steps) - 1):
                defaultDir = "transforms"
            nlinStage = ma.minctracc(inputPipeFH, 
                                     templatePipeFH,
                                     defaultDir=defaultDir,
                                     blur=blurs[i],
                                     gradient=gradients[i],
                                     iterations=iterations[i],
                                     step=steps[i],
                                     similarity=0.8,
                                     w_translations=w_translations,
                                     simplex=simplexes[i])
            self.p.addStage(nlinStage)
