#!/usr/bin/env python

from os.path import abspath
import csv
import sys

"""
    Each of these classes allows users to read a specified set of registration 
    parameters from a .csv file. These parameters can override the specified 
    defaults, which are currently optimized for mouse brains scanned at 56 
    micron isotropic resolution. Although the structure of both the minctracc
    and mincANTS classes are the same, the parameters each uses are different
    enough that a common base class doesn't make sense. 
    
    In each __init__ function, if a protocol is specified, it is used. 
    Otherwise, defaults are set based on the file resolution. This means that
    a protocol specified as a csv must include all values necessary for an 
    LSQ6, LSQ12, or NLIN minctracc registration or NLIN ANTS registration.  
            
    Currently, a specified protocol MUST be a csv file that uses a SEMI-COLON 
    to separate the fields. Examples are:
    applications_testing/test_data/minctracc_example_nlin_protocol.csv
    applications_testing/test_data/minctracc_example_linear_protocol.csv
    applications_testing/test_data/mincANTS_example_protocol.csv
    
    Note:
    The minctracc_example_linear_protocol.csv may be used for LSQ6 or LSQ12 registrations
    and the minctracc_example_nlin_protocol.csv contains extra parameters necessary
    for non-linear registrations. 
           
    Each row in the csv is a different input to the either a minctracc or mincANTS call
    Although the number of entries in each row (e.g. generations) is variable, the
    specific allowed parameters are fixed. For example, one could specify a subset of the
    allowed parameters (e.g. blurs only) but could not rename any parameters
    or use additional ones that haven't already been defined without subclassing. See
    documentation for additional details. 
             
    Based on the length of these parameter arrays specified, the number of generations
    for a series of minctracc and mincANTS calls is set. 
    
    Current classes:
        1. setMincANTSParams -- sets all parameters for a non-linear mincANTS registration
        2. setOneGenMincANTSParams -- class to override params for a single ANTS call. 
        3. setNlinMinctraccParams -- base class for minctracc. All potential parameter options
           are set here. 
        4. setLSQ12MinctraccParams -- inherits from setNlinMinctraccParams. Uses only parameters
           and defaults that are needed for an LSQ12 registration. 
        5. setLSQ6MinctraccParams -- inherits setLSQ12MinctraccParams. Uses parameters and defaults
           for either an identity or centre estimation alignment. Not appropriate for large rotations. 
    
    Several functions have been created for adding protocol options to the ArgumentParser of various
    modules. The paramsArguments() class creates options for lsq6, lsq12 and nlin protocols. The 
    following functions have been created:
    1. addLSQ6ArgumentGroup: adds --lsq6-protocol option
    2. addLSQ12ArgumentGroup: adds --lsq12-protocol option
    3. addNLINArgumentGroup: adds --nlin-protocol option
    4. addRegParamsArgumentGroup: adds options for all protocols
    5. addLSQ12NLINParamsArgumentGroup: adds --lsq12-protocol and --nlin-protocol options
"""

class setMincANTSParams(object):
    def __init__(self, fileRes, reg_protocol=None):
        self.fileRes = fileRes
        self.regProtocol=reg_protocol
        self.blurs = []
        self.gradient = []
        self.similarityMetric = []
        self.weight = []
        self.radiusHisto = []
        self.transformationModel = []
        self.regularization = []
        self.iterations = []
        self.useMask = []
        
        #Setup protocol
        self.setupProtocol()
        
        #Get number of iterations
        self.generations = self.getGenerations()
        
        #If a registration protocol exists, use it to setup parameters. Assume it is a csv. If this fails,
        #the except clause assumes it is an object of type(self).
    def setupProtocol(self): 
        if self.regProtocol:
            # FIXME: if a regProtocol is specified, it must currently specify ALL
            # registration options as the defaults are not used in this case.
            # It might make sense just to set default params anyway before doing this
            try:
                self.setParams()
            except:
                try:
                    self.blurs = self.regProtocol.blurs
                    self.gradient = self.regProtocol.gradient
                    self.similarityMetric = self.regProtocol.similarityMetric
                    self.weight = self.regProtocol.weight
                    self.radiusHisto = self.regProtocol.radiusHisto
                    self.transformationModel = self.regProtocol.transformationModel
                    self.regularization = self.regProtocol.regularization
                    self.iterations = self.regProtocol.iterations
                    self.useMask = self.regProtocol.useMask
                    self.memoryRequired = self.regProtocol.memoryRequired
                except:
                    print "The non-linear protocol you have specified is in an unrecognized form: \n%s\n Exiting..." % self.regProtocol
                    sys.exit()
        else:
            if self.fileRes:
                self.defaultParams()
            else:
                print "Unable to set default registration parameters due to lack of file resolution."
                print "Try specifying a non-linear protocol, or investigate why this is happening."
                sys.exit()
        self.generations = self.getGenerations()

    def defaultParams(self):
        """
            Default mincANTS parameters. 
           
            Note that for each generation, the blurs, gradient, similarity_metric,
            weight and radius/histogram are arrays. This is to allow for the one
            or more similarity metrics (and their associated parameters) to be specified
            for each mincANTS call. We typically use two, but the mincANTS atom allows
            for more or less if desired. 
        """
        self.blurs = [[-1, self.fileRes], [-1, self.fileRes],[-1, self.fileRes]] 
        self.gradient = [[False,True], [False,True], [False,True]]
        self.similarityMetric = [["CC", "CC"],["CC", "CC"],["CC", "CC"]]
        self.weight = [[1,1],[1,1],[1,1]]
        self.radiusHisto = [[3,3],[3,3],[3,3]]
        self.transformationModel = ["SyN[0.1]", "SyN[0.1]", "SyN[0.1]"]
        self.regularization = ["Gauss[2,1]", "Gauss[2,1]", "Gauss[2,1]"]
        self.iterations = ["100x100x100x0", "100x100x100x20", "100x100x100x100"]
        self.useMask = [False, True, True]
        self.memoryRequired = [0.177, 1.385e-7, 2.1e-7]
        
    def setParams(self):
        """Set parameters from specified protocol"""
        
        """Read parameters into array from csv."""
        inputCsv = open(abspath(self.regProtocol), 'rb')
        csvReader = csv.reader(inputCsv, delimiter=';', skipinitialspace=True)
        params = []
        for r in csvReader:
            params.append(r)
        """Parse through rows and assign appropriate values to each parameter array.
           Everything is read in as strings, but in some cases, must be converted to 
           floats, booleans or gradients. 
        """
        for p in params:
            if p[0]=="blur":
                """Blurs must be converted to floats."""
                self.blurs = []
                for i in range(1,len(p)):
                    b = []
                    for j in p[i].split(","):
                        b.append(float(j)) 
                    self.blurs.append(b)
            elif p[0]=="gradient":
                self.gradient = []
                """Gradients must be converted to bools."""
                for i in range(1,len(p)):
                    g = []
                    for j in p[i].split(","):
                        if j=="True" or j=="TRUE":
                            g.append(True)
                        elif j=="False" or j=="FALSE":
                            g.append(False)
                    self.gradient.append(g)
            elif p[0]=="similarity_metric":
                self.similarityMetric = []
                """Similarity metric does not need to be converted, but must be stored as an array for each generation."""
                for i in range(1,len(p)):
                    g = []
                    for j in p[i].split(","):
                        g.append(j)
                    self.similarityMetric.append(g)
            elif p[0]=="weight":
                self.weight = []
                """Weights are strings but must be converted to an int."""
                for i in range(1,len(p)):
                    w = []
                    for j in p[i].split(","):
                        w.append(int(j)) 
                    self.weight.append(w)
            elif p[0]=="radius_or_histo":
                self.radiusHisto = []
                """The radius or histogram parameter is a string, but must be converted to an int"""
                for i in range(1,len(p)):
                    r = []
                    for j in p[i].split(","):
                        r.append(int(j)) 
                    self.radiusHisto.append(r)
            elif p[0]=="transformation":
                self.transformationModel = []
                for i in range(1,len(p)):
                    self.transformationModel.append(p[i])
            elif p[0]=="regularization":
                self.regularization = []
                for i in range(1,len(p)):
                    self.regularization.append(p[i])
            elif p[0]=="iterations":
                self.iterations = []
                for i in range(1,len(p)):
                    self.iterations.append(p[i])
            elif p[0]=="useMask":
                self.useMask = []
                for i in range(1,len(p)):
                    """useMask must be converted to a bool."""
                    if p[i] == "True" or p[i] == "TRUE":
                        self.useMask.append(True)
                    elif p[i] == "False" or p[i] == "FALSE":
                        self.useMask.append(False)
            elif p[0]=="memoryRequired":
                """linear estimate of memRequired: mem = p[0] + voxels * p[i]
                   where i = 2 (for NxMxPx0 stages) or i = 3 (for highest res stages)"""
                self.memoryRequired = (float(p[1]), float(p[2]), float(p[3]))
            else:
                print "Improper parameter specified for mincANTS protocol: " + str(p[0])
                print "Exiting..."
                sys.exit()
                
    def getGenerations(self):
        arrayLength = len(self.blurs)
        errorMsg = "Array lengths in non-linear mincANTS protocol do not match."
        if (len(self.gradient) != arrayLength 
            or len(self.similarityMetric) != arrayLength
            or len(self.weight) != arrayLength
            or len(self.radiusHisto) != arrayLength
            or len(self.transformationModel) != arrayLength
            or len(self.regularization) != arrayLength
            or len(self.iterations) != arrayLength
            or len(self.useMask) != arrayLength):
            print errorMsg
            raise
        else:
            return arrayLength

class setOneGenMincANTSParams(setMincANTSParams):
    def __init__(self, fileRes, reg_protocol=None):
        setMincANTSParams.__init__(self, fileRes, reg_protocol=reg_protocol)
    
    def defaultParams(self):
        """
            Default mincANTS parameters for a single generation. 
        """
        self.blurs = [[-1, self.fileRes]] 
        self.gradient = [[False,True]]
        self.similarityMetric = [["CC", "CC"]]
        self.weight = [[1,1]]
        self.radiusHisto = [[3,3]]
        self.transformationModel = ["SyN[0.1]"]
        self.regularization = ["Gauss[2,1]"]
        self.iterations = ["100x100x100x150"]
        self.useMask = [False]

class setNlinMinctraccParams(object):
    def __init__(self, fileRes, reg_protocol=None):
        self.fileRes = fileRes
        self.regProtocol = reg_protocol
        self.blurs = []
        self.stepSize = []
        self.iterations = []
        self.simplex = []
        self.useGradient = []
        self.optimization = []
        self.w_translations = []
        self.stiffness = []
        self.weight = []
        self.similarity = []
        
        #Setup protocol
        self.setupProtocol()
        
        #Get number of iterations
        self.generations = self.getGenerations()
        
    def setupProtocol(self):
        #If a registration protocol exists, use it to setup parameters. Assume it is a csv. If this fails,
        #the except clause assumes it is an object of type(self). 
        if self.regProtocol:
            try:
                self.setParams()
            except:
                try:
                    self.blurs = self.regProtocol.blurs
                    self.stepSize = self.regProtocol.stepSize
                    self.iterations = self.regProtocol.iterations
                    self.simplex = self.regProtocol.simplex
                    self.useGradient = self.regProtocol.useGradient
                    self.optimization = self.regProtocol.optimization
                    self.w_translations = self.regProtocol.w_translations
                    self.similarity = self.regProtocol.similarity
                    self.weight = self.regProtocol.weight
                    self.stiffness = self.regProtocol.stiffness
                except:
                    print "The non-linear protocol you have specified is in an unrecognized form: \n%s\n Exiting..." % self.regProtocol
                    sys.exit()
        else:
            if self.fileRes:
                self.defaultParams()
            else:
                print "Unable to set default registration parameters due to lack of file resolution."
                print "Try specifying a non-linear protocol, or investigate why this is happening."
                sys.exit()
        

    def defaultParams(self):
        """ Default minctracc parameters """
        
        #TODO: Rewrite this so it looks more like LSQ6 with matrices of factors
        self.blurs = [self.fileRes*5.0, self.fileRes*(10.0/3.0), self.fileRes*(10.0/3.0),
                      self.fileRes*(10.0/3.0), self.fileRes*(5.0/3.0), self.fileRes]
        self.stepSize = [self.fileRes*(35.0/3.0), self.fileRes*10.0, self.fileRes*(25.0/3.0),
                      self.fileRes*4.0, self.fileRes*2.0, self.fileRes]
        self.iterations = [20,6,8,8,8,8]
        self.simplex =    [5,2,2,2,2,2]
        self.useGradient = [True, True, True, True, True, True]
        self.optimization = ["-use_simplex", "-use_simplex", "-use_simplex", "-use_simplex", 
                             "-use_simplex", "-use_simplex"]
        self.w_translations = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4 ]
        self.stiffness  =     [0.98,0.98,0.98,0.98,0.98,0.98]
        self.weight     =     [0.8, 0.8, 0.8, 0.8, 0.8, 0.8 ]
        self.similarity =     [0.8, 0.8, 0.8, 0.8, 0.8, 0.8 ]
            
    def setParams(self):
        """Set parameters from specified protocol"""
        
        """Read parameters into array from csv."""
        inputCsv = open(abspath(self.regProtocol), 'rb')
        csvReader = csv.reader(inputCsv, delimiter=';', skipinitialspace=True)
        params = []
        for r in csvReader:
            params.append(r)
        """Parse through rows and assign appropriate values to each parameter array.
           Everything is read in as strings, but in some cases, must be converted to 
           floats, booleans or gradients. 
        """
        # the following parameters do not need to be present in the protocol
        # keep track of whether they were found, otherwise set defaults at the end
        stiffness_found = 0
        weight_found = 0
        similarity_found = 0
        for p in params:
            if p[0]=="blur":
                self.blurs = []
                """Blurs must be converted to floats."""
                for i in range(1,len(p)):
                    self.blurs.append(float(p[i]))
            elif p[0]=="step":
                self.stepSize = []
                """Steps are strings but must be converted to a float."""
                for i in range(1,len(p)):
                    self.stepSize.append(float(p[i]))
            elif p[0]=="iterations":
                self.iterations = []
                """The iterations parameter is a string, but must be converted to an int"""
                for i in range(1,len(p)):
                    self.iterations.append(int(p[i]))
            elif p[0]=="simplex":
                self.simplex = []
                """Simplex must be converted to a float."""
                for i in range(1,len(p)):
                    self.simplex.append(float(p[i]))
            elif p[0]=="gradient":
                self.useGradient = []
                """Gradients must be converted to bools."""
                for i in range(1,len(p)):
                    if p[i]=="True" or p[i]=="TRUE":
                        self.useGradient.append(True)  
                    elif p[i]=="False" or p[i]=="FALSE":
                        self.useGradient.append(False)          
            elif p[0]=="optimization":
                self.optimization = []
                for i in range(1,len(p)):
                    self.optimization.append(p[i])
            elif p[0]=="w_translations":
                self.w_translations = []
                """w_translations are strings but must be converted to a float."""
                for i in range(1,len(p)):
                    self.w_translations.append(float(p[i]))
            elif p[0] == "stiffness":
                stiffness_found = 1
                self.stiffness = []
                """stiffness is a minctracc weighting factor, should be float"""
                for i in range(1,len(p)):
                    self.stiffness.append(float(p[i]))
            elif p[0] == "weight":
                weight_found = 1
                self.weight = []
                """weight is a minctracc weighting factor, should be float"""
                for i in range(1,len(p)):
                    self.weight.append(float(p[i]))
            elif p[0] == "similarity":
                similarity_found = 1
                self.similarity = []
                """similarity is a minctracc weighting factor, should be float"""
                for i in range(1,len(p)):
                    self.similarity.append(float(p[i]))
            else:
                print "Improper parameter specified for minctracc protocol: " + str(p[0])
                print "Exiting..."
                sys.exit()
        # set defaults that don't have to appear in the protocol
        if(not stiffness_found):
            # default: 0.98 for each stage. Take length of stepSize list as reference for number of stages
            self.stiffness = [0.98 for i in range(len(self.stepSize))]
        if(not weight_found):
            # default: 0.8 for each stage. Take length of stepSize list as reference for number of stages
            self.weight = [0.8 for i in range(len(self.stepSize))]
        if(not similarity_found):
            # default: 0.8 for each stage. Take length of stepSize list as reference for number of stages
            self.similarity = [0.8 for i in range(len(self.stepSize))]

      
    def getGenerations(self):
        arrayLength = len(self.blurs)
        errorMsg = "Number of parameters in non-linear minctracc protocol is not consistent."
        if (len(self.stepSize) != arrayLength 
            or len(self.iterations) != arrayLength
            or len(self.simplex) != arrayLength
            or len(self.useGradient) != arrayLength
            or len(self.w_translations) != arrayLength
            or len(self.optimization) != arrayLength
            or len(self.similarity) != arrayLength
            or len(self.weight) != arrayLength
            or len(self.stiffness) != arrayLength):
            print errorMsg
            sys.exit()
        else:
            return arrayLength
        
class setLSQ12MinctraccParams(setNlinMinctraccParams):
    def __init__(self, fileRes, subject_matter=None, reg_protocol=None):
        self.subject_matter = subject_matter
        setNlinMinctraccParams.__init__(self, fileRes, reg_protocol=reg_protocol)
        
    def setupProtocol(self):
        if self.subject_matter:
            self.defaultParams()
        else:
            super(setLSQ12MinctraccParams, self).setupProtocol()

    def defaultParams(self):
        """ 
            Default minctracc parameters based on resolution of file, unless
            a particular subject matter was provided
        """
        
        blurfactors      = [       5,   10.0/3.0,         2.5]
        stepfactors      = [50.0/3.0,   25.0/3.0,         5.5]
        simplexfactors   = [      50,         25,    50.0/3.0]
        
        if(self.subject_matter == "mousebrain"):
            self.blurs =    [0.3, 0.2, 0.15]
            self.stepSize=  [1,   0.5, 1.0/3.0]
            self.simplex=   [3,   1.5, 1]
        else:
            self.blurs = [i * self.fileRes for i in blurfactors]
            self.stepSize=[i * self.fileRes for i in stepfactors]
            self.simplex=[i * self.fileRes for i in simplexfactors]
        
        self.useGradient=[False,True,False]
        self.w_translations = [0.4,0.4,0.4]
        
    def getGenerations(self):
        arrayLength = len(self.blurs)
        errorMsg = "Number of parameters in linear minctracc protocol is not consistent."
        if (len(self.stepSize) != arrayLength 
            or len(self.useGradient) != arrayLength
            or len(self.simplex) != arrayLength
            or len(self.w_translations) != arrayLength):
            print errorMsg
            sys.exit()
        else:
            return arrayLength 

class setLSQ6MinctraccParams(setLSQ12MinctraccParams):
    def __init__(self, fileRes, initial_transform="estimate", reg_protocol=None):
        self.initial_transform = initial_transform
        setLSQ12MinctraccParams.__init__(self, fileRes, reg_protocol=reg_protocol)
    
    def defaultParams(self):
        """ 
            Default minctracc parameters based on resolution of file, and whether or not
            initial transform is identity or estimate. 
            
            Example values for 56 micron files.
            If using the identity transform:
                blurs in mm at 56micron files: [0.952, 0.504, 0.224]
                simplex in mm at 56mircon files: [2.24, 1.568, 0.896]
            
            If using the centre estimation:
                blurs in mm at 56micron files: [5.04,  1.96, 0.952, 0.504, 0.224]
                simplex in mm at 56mircon files: [7.168, 3.584, 2.24, 1.568, 0.896]
        """
        
        if self.initial_transform == "identity":
            blurfactors      = [   17,   9,    4]
            simplexfactors   = [   40,  28,   16]
            stepfactors      = [   17,   9,    4]
            gradientdefaults = [False,True,False]
            translations     = [0.4,0.4,0.4]
        elif self.initial_transform == "estimate":
            blurfactors      = [   90,   35,   17,   9,    4]
            simplexfactors   = [  128,   64,   40,  28,   16]
            stepfactors      = [   90,   35,   17,   9,    4]
            gradientdefaults = [False,False,False,True,False]
            translations     = [0.4,0.4,0.4,0.4,0.4]
        else:
            print "An improper initial transform was specified: " + str(self.initial_transform)
            print "Exiting..."
            sys.exit()
            
        self.blurs    = [i * self.fileRes for i in blurfactors]
        self.simplex  = [i * self.fileRes for i in simplexfactors]
        self.stepSize = [i * self.fileRes for i in stepfactors]
        self.useGradient = gradientdefaults
        self.w_translations = translations
