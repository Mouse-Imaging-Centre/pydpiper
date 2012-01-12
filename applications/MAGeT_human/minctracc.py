#!/usr/bin/env python

from pydpiper.pipeline import * 
from optparse import OptionParser
from os.path import basename,dirname,isdir,abspath
from os import mkdir
import time
import networkx as nx

Pyro.config.PYRO_MOBILE_CODE=1

class mincFileHandling(FileHandling):
    def __init__(self):
        FileHandling.__init__(self)
    def createBlurOutputAndLogFiles(self, output_base, log_base, b):
        argArray = ["fwhm", b, "blur"]
        return (self.createOutputAndLogFiles(output_base, log_base, ".mnc", argArray))
    def createXfmAndLogFiles(self, output_base, log_base, argArray):
        return (self.createOutputAndLogFiles(output_base, log_base, ".xfm", argArray))
    def createResampledAndLogFiles(self, output_base, log_base, argArray):
        argArray.insert(0, "resampled")
        return (self.createOutputAndLogFiles(output_base, log_base, ".mnc", argArray))

class minctracc(CmdStage):
    def __init__(self, source, target, output, logfile,
		 linearparam="nlin",
                 source_mask=None, 
                 target_mask=None,
                 iterations=15,
                 step=0.5,
                 transform=None,
                 weight=1,
                 stiffness=1,
                 similarity=0.3, 
                 sub_lattice=6, 
                 ident=False):
        CmdStage.__init__(self, None) #don't do any arg processing in superclass
        self.source = source
        self.target = target
        self.output = output
        self.logFile = logfile
        self.linearparam = linearparam
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.iterations = str(iterations)
        self.lattice_diameter = str(step*3)
        self.step = str(step)
        self.transform = transform
        self.weight = str(weight)
        self.stiffness = str(stiffness)
        self.similarity = str(similarity)
        self.sub_lattice = str(sub_lattice)
        self.ident = ident
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        self.colour = "red"

    def setName(self):
        self.name = "minctracc nlin step: " + self.step 
    def addDefaults(self):
        self.cmd = ["minctracc",
                    "-clobber",
                    "-similarity", self.similarity,
                    "-weight", self.weight,
                    "-stiffness", self.stiffness,
                    "-step", self.step, self.step, self.step,

                    self.source,
                    self.target,
                    self.output]
        
        # assing inputs and outputs
        # assing inputs and outputs
        self.inputFiles = [self.source, self.target]
        if self.source_mask:
            self.inputFiles += [self.source_mask]
            self.cmd += ["-source_mask", self.source_mask]
        if self.target_mask:
            self.inputFiles += [self.target_mask]
            self.cmd += ["-model_mask", self.target_mask]
        if self.transform:
            self.inputFiles += [self.transform]
            self.cmd += ["-transform", self.transform]
        if self.ident:
            self.cmd += ["-ident"]
        self.outputFiles = [self.output]

    def finalizeCommand(self):
        """add the options for non-linear registration"""
        # build the command itself
        self.cmd += ["-iterations", self.iterations,
                    "-nonlinear", "corrcoeff", "-sub_lattice", self.sub_lattice,
                    "-lattice_diameter", self.lattice_diameter,
                     self.lattice_diameter, self.lattice_diameter]

class linearminctracc(minctracc):
    def __init__(self, source, target, output, logfile, linearparam,
                 source_mask=None, target_mask=None):
        minctracc.__init__(self,source,target,output,
			   logfile,linearparam,
                           source_mask=source_mask,
                           target_mask=target_mask)

    def finalizeCommand(self):
        """add the options for a linear fit"""
	_numCmd = "-" + self.linearparam
        self.cmd += ["-xcorr", _numCmd]
    def setName(self):
        self.name = "minctracc" + self.linearparam + " "
        
class mritotal(CmdStage):
    def __init__(self, input, output, modeldir, model, logfile):
        CmdStage.__init__(self,None)
        self.inputFiles = [input]
        self.outputFiles = [output]
        self.logFile = logfile
        self.cmd = ["mritotal", input, output, "-modeldir", modeldir, "-model", model]
        self.name = "mritotal " + basename(input)
        
class nu_correct(CmdStage):
    def __init__(self, input, output, logfile):
        CmdStage.__init__(self, None)
        self.inputFiles = [input]
        self.outputFiles = [output]
        self.logFile = logfile
        self.cmd = ["nu_correct", "-clobber", input, output]
        self.name = "nu_correct " + basename(input)
        
class xfmconcat(CmdStage):
    def __init__(self, inputs, output, logfile):
        CmdStage.__init__(self, None)
        if type(inputs) != list: 
            inputs = [inputs]
            
        self.inputFiles = inputs
        self.outputFiles = [output]
        self.logFile = logfile
        self.cmd = ["xfmconcat", "-clobber"] + inputs + [output]
        self.name = "xfmconcat " + basename(output)
               
class blur(CmdStage):
    def __init__(self, input, output, logfile, fwhm):
        # note - output should end with _blur.mnc
        CmdStage.__init__(self, None)
        self.base = output.replace("_blur.mnc", "")
        self.inputFiles = [input]
        self.outputFiles = [output]
        self.logFile = logfile
        self.cmd = ["mincblur", "-clobber", "-fwhm", str(fwhm),
                    input, self.base]
        self.name = "mincblur " + str(fwhm) + " " + basename(input)
        self.colour="blue"

class mincresample(CmdStage):
    def __init__(self, inputFile, outputFile, logfile, argarray=[], like=None, cxfm=None):
        if argarray:
            argarray = ["mincresample"] + argarray
        else:
            argarray = ["mincresample"]
        CmdStage.__init__(self, argarray)
        
        # inputFile and outputFile are required, everything else optional
        # This class assuming use of the most commonly used flags (-2, -clobber, -like, -transform)
        # Any commands above and beyond the standard will be read in from argarray
        # argarray could contain like and/or output files
        if like:
            self.cmd += ["-like", like]
        if cxfm:
            self.inputFiles += [cxfm]
            self.cmd += ["-transform", cxfm]
        self.inputFiles += [inputFile]
        self.outputFiles += [outputFile]
        self.logFile = logfile
        self.cmd += ["-2", "-clobber", inputFile, outputFile]

