from pydpiper.pipeline import Pipeline
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.registration_functions as rf

class HierarchicalMinctracc:
    """Default HierarchicalMinctracc currently does:
        1. 2 lsq12 stages with a blur of 0.25
        2. 5 nlin stages with a blur of 0.25
        3. 1 nlin stage with no blur
       To override these defaults, lsq12 and nlin protocols may be specified. """
    def __init__(self, 
                 inputFH, 
                 targetFH,
                 lsq12_protocol=None,
                 nlin_protocol=None,
                 includeLinear = True,
                 subject_matter = None,  
                 defaultDir="tmp"):
        
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.lsq12_protocol = lsq12_protocol
        self.nlin_protocol = nlin_protocol
        self.includeLinear = includeLinear
        self.subject_matter = subject_matter
        self.defaultDir = defaultDir
        
        try: # the attempt to access the minc volume will fail if it doesn't yet exist at pipeline creation
            self.fileRes = rf.getFinestResolution(self.inputFH)
        except: 
            # if it indeed failed, get resolution from the original file specified for 
            # one of the input files, which should exist. 
            # Can be overwritten by the user through specifying a nonlinear protocol.
            self.fileRes = rf.getFinestResolution(self.inputFH.inputFileName)
        
        self.buildPipeline()
        
    def buildPipeline(self):
            
        # Do LSQ12 alignment prior to non-linear stages if desired
        if self.includeLinear: 
            lp = mp.setLSQ12MinctraccParams(self.fileRes,
                                            subject_matter=self.subject_matter,
                                            reg_protocol=self.lsq12_protocol)
            lsq12reg = lsq12.LSQ12(self.inputFH, 
                                   self.targetFH, 
                                   blurs=lp.blurs,
                                   step=lp.stepSize,
                                   gradient=lp.useGradient,
                                   simplex=lp.simplex,
                                   w_translations=lp.w_translations,
                                   defaultDir=self.defaultDir)
            self.p.addPipeline(lsq12reg.p)
        
        # create the nonlinear registrations
        np = mp.setNlinMinctraccParams(self.fileRes, reg_protocol=self.nlin_protocol)
        for b in np.blurs: 
            if b != -1:           
                self.p.addStage(ma.blur(self.inputFH, b, gradient=True))
                self.p.addStage(ma.blur(self.targetFH, b, gradient=True))
        for i in range(len(np.stepSize)):
            #For the final stage, make sure the output directory is transforms.
            if i == (len(np.stepSize) - 1):
                self.defaultDir = "transforms"
            nlinStage = ma.minctracc(self.inputFH, 
                                     self.targetFH,
                                     defaultDir=self.defaultDir,
                                     blur=np.blurs[i],
                                     gradient=np.useGradient[i],
                                     iterations=np.iterations[i],
                                     step=np.stepSize[i],
                                     w_translations=np.w_translations[i],
                                     simplex=np.simplex[i],
                                     optimization=np.optimization[i])
            self.p.addStage(nlinStage)
