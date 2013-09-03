#!/usr/bin/env python

from pydpiper.queueing import runOnQueueingSystem
import pytest
from os.path import isfile
    
class TestPbsQueueing():        
    """All of the tests below require specifying --queue=pbs on the command line"""
    
    """The following scenarios test launching the pipeline from within an application.
                test_roq_init_defaults
                test_create_job_0exec
                test_create_job_1exec
                test_create_job_multiproc
                test_create_job_3execs
                test_create_job_with_args
        This requires calling the createPbsScripts() method. 
        
        test_create_executor_only tests the case when the executor is launched separately
        from the main pipeline
    """
    
    def test_roq_init_defaults(self, setupopts):
        """Makes sure that pbs related defaults are correctly initialized"""
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        assert roq.queue == "pbs"
        assert roq.arguments == None
        assert roq.jobName == "pydpiper"
        assert roq.time == "2:00:00:00"
        
    def test_create_job_0exec(self, setupopts):
        """Test defaults and file creation when no executors are specified"""
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.numexec != 0:
            pytest.skip("This test will not run if --num-executors is specified")
        roq.createPbsScripts()
        assert roq.jobName == "pydpiper"
        assert isfile(roq.jobFileName) == True
    
    def test_create_job_1exec(self, setupopts):
        """This test requires specifying --num-executors=1 on command line
           Verifies defaults and file creation when only one executor is specified"""
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.numexec != 1:
            pytest.skip("specify --num-executors=1 to continue with this test")
        roq.createPbsScripts()
        assert roq.jobName == "pydpiper"
        assert isfile(roq.jobFileName) == True
        callsExec = False
        jobFileString = open(roq.jobFileName).read()
        if "pipeline_executor.py" in jobFileString:
                callsExec = True
        assert callsExec == True
    
    def test_create_job_multiproc(self, setupopts):
        """This test requires specifying --proc=24 on command line
           Verifies defaults and file creation when multiple nodes are needed"""
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.ppn != 8:
            pytest.skip("specify --ppn=8 to continue with this test")
        if roq.numexec != 1:
            pytest.skip("specify --num-executors=1 to continue with this test")
        if roq.proc != 24:
            pytest.skip("specify --proc==24 to continue with this test")
        roq.createPbsScripts()
        correctNodesPpn = False
        jobFileString = open(roq.jobFileName).read()
        if "nodes=3:ppn=8" in jobFileString:
                correctNodesPpn = True
        assert correctNodesPpn == True
    
    def test_create_job_newppn(self, setupopts):
        """This test requires specifying --ppn=10, --num-executors=1 and --proc=24
           Verifies appropriate node selection"""
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.ppn != 10:
            pytest.skip("specify --ppn=10 to continue with this test")
        if roq.numexec != 1:
            pytest.skip("specify --num-executors=1 to continue with this test")
        if roq.proc != 24:
            pytest.skip("specify --proc==24 to continue with this test")
        roq.createPbsScripts()
        roq.createPbsScripts()
        correctPpn = False
        correctProc = False
        jobFileString = open(roq.jobFileName).read()
        if "nodes=2:ppn=10" in jobFileString:
            correctPpn=True
        if "--proc=24" in jobFileString:
            correctProc = True
        assert correctPpn == True
        assert correctProc == True
        
    def test_create_job_3execs(self, setupopts):
        """This test requires specifying --num-executors=3 on command line
           Verifies defaults and file creation when three executors are specified"""
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.numexec != 3:
            pytest.skip("specify --num-executors=3 to continue with this test")
        roq.createPbsScripts()
        # The following assert statements apply to the final .job file created
        correctFileName = False
        callsExec = False
        correctName = False
        if "pydpiper-executor-2" in roq.jobFileName:
            correctFileName = True
        jobFileString = open(roq.jobFileName).read()
        if "pipeline_executor.py" in jobFileString:
            callsExec = True
        if "pydpiper-executor" in jobFileString:
            correctName = True
        assert correctFileName == True
        assert callsExec == True
        assert correctName == True
        
    def test_create_job_with_args(self, setupopts):
        """This test verifies the appropriate script creation for a sample
           set of arguments."""
        allOptions = setupopts.returnAllOptions()
        sampleArgs = setupopts.returnSampleArgs()
        roq = runOnQueueingSystem(allOptions, sysArgs=sampleArgs)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.numexec != 1:
            pytest.skip("specify --num-executors=1 to continue with this test")
        roq.createPbsScripts()
        assert roq.jobName == "TestProgName.py"
        mncFilesIncluded = False
        sleepIncluded = False
        jobFileString = open(roq.jobFileName).read()
        if "img_A.mnc" and "img_B.mnc" in jobFileString:
            mncFilesIncluded = True
        if "sleep 60" in jobFileString:
            sleepIncluded = True
        assert mncFilesIncluded == True
        assert sleepIncluded == True
    
    def test_create_executor_only(self, setupopts):
        allOptions = setupopts.returnAllOptions()
        roq = runOnQueueingSystem(allOptions)
        if roq.queue is None:
            pytest.skip("specify --queue=pbs to continue with this test")
        if roq.numexec != 1:
            pytest.skip("specify --num-executors=1 to continue with this test")
        roq.createExecutorJobFile(1)
        assert roq.jobName == "pydpiper"
        assert isfile(roq.jobFileName) == True
        correctFileName = False
        callsExec = False
        correctName = False
        if "pydpiper-executor-1" in roq.jobFileName:
            correctFileName = True
        jobFileString = open(roq.jobFileName).read()
        if "pipeline_executor.py" in jobFileString:
            callsExec = True
        if "pydpiper-executor" in jobFileString:
            correctName = True
        assert correctFileName == True
        assert callsExec == True
        assert correctName == True