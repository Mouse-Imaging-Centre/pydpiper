from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper.file_handling as fh

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

class RotationalMinctracc:
    """Default RotationalMinctracc class
       Currently just calls rotational_minctracc.py 
       with minimal updates. Ultimately, we will do
       a more substantial overhaul. 
    """
    def __init__(self, 
                 inputPipeFH, 
                 templatePipeFH,
                 blurs=[0.5]):
        
        self.p = Pipeline()
        self.inputPipeFH = inputPipeFH
        self.templatePipeFH = templatePipeFH
        
        self.blurFiles(blurs) 
        for b in blurs:
            self.buildCmd(b)
        
    def blurFiles(self, blurs):
        for b in blurs:
            if b != -1:
                tblur = ma.blur(self.templatePipeFH, b, gradient=True)
                iblur = ma.blur(self.inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
    
    def buildCmd(self, b):
        """Only -w_translations override rotational_minctracc.py defaults. 
           Keep this here. Rather than giving the option to override other
           defaults. We will eventually re-write this code.
        """
        w_trans = str(0.4)
        cmd = ["rotational_minctracc.py", "-t", "/dev/shm/", "-w", w_trans, w_trans, w_trans]
        source = self.inputPipeFH.getBlur(b)
        target = self.templatePipeFH.getBlur(b)
        mask = self.templatePipeFH.getMask()
        if mask:
            cmd += ["-m", InputFile(mask)]
        outputXfm = self.inputPipeFH.registerVolume(self.templatePipeFH)
        cmd +=[InputFile(source), InputFile(target), OutputFile(outputXfm), "/dev/null"]
        rm = CmdStage(cmd)
        rm.setLogFile(LogFile(fh.logFromFile(self.inputPipeFH.logDir, outputXfm)))
        self.p.addStage(rm)
        

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
