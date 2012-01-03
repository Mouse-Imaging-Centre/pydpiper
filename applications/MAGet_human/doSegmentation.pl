#! /usr/bin/env perl

##############
#Subcortical Segmentation module using basal ganglia atlas
#author: Mallar Chakravarty 
#send issues to mallar.chak@gmail.com
#
###############


use strict;
use File::Basename;
use File::Temp qw/ tempdir /;
use Getopt::Tabular;


##############
# Define usage and help stuff
##############

my $me     = &basename($0);


my $Usage = "$0 performs subcortical segmentation of the \n".
"striatum, globus pallidus, and thalamus\n".
"Usage: $0 <inputMRI> <outputDir> \n";


my $colinDir     = qw(models);
my $colinGlobal  = qw(colin27_t1_tal_lin);
my $colinSubCort = qw(colin_bg_generous_0.3mm.mnc);
my $labelsLeft   = qw(models/mask_left_oncolinnl_7.mnc);
my $labelsRight  = qw(models/mask_right_oncolinnl_7.mnc);


##############
#Set up options
##############

&Getopt::Tabular::SetHelp($Usage);


my @argTbl = 
	(
		["Model Options", "section"],
		["-colindir", "string", 1, \$colinDir,
		"set the directory to search for model files."],
		["-model", "string", 1, \$colinGlobal,
		"set the base name of the fit model files. Basename only."],
		["-subcortical_model", "string", 1, \$colinSubCort,
		"set the base name of the fit model files."],
	        ["-labels_left", "string", 1, \$labelsLeft,
		 "set the file for labels on the left side"],
	        ["-labels_right", "string", 1, \$labelsRight,
		 "set the file for the labels on the right side"]
	);



&GetOptions(\@argTbl, \@ARGV);





my $inputMRI  = shift(@ARGV) or die "$Usage\n";
my $outputDir = shift(@ARGV) or die "$Usage \n";


if(!(-e $inputMRI)){
    print "Input file does not exist \n"; die;
}

if(!(-e $outputDir)){
    print "Output directory $outputDir does not exist \n";
    print "Will make $outputDir \n";
    do_cmd("mkdir", "$outputDir");
}

else{
    print "Output directory $outputDir exists \n";
    print "Let's do this business!!! \n";
}



#############
# Make TempDir
#############

my $tmpdir = &tempdir("$me-XXXXXXXXXXX", TMPDIR => 1, CLEANUP => 1);



my $NUCDir          = "${outputDir}/NUC";
my $outXFMS         = "${outputDir}/XFMS";
my $outResample     = "${outputDir}/resample";
my $outSegmentation = "${outputDir}/segmentation";

my $colinGlobalFull      = "${colinDir}/${colinGlobal}.mnc";
my $colinSubcorticalFull = "${colinDir}/${colinSubCort}";


my @dirs = ($NUCDir, $outXFMS, $outResample, $outSegmentation);

foreach(@dirs){ do_cmd("mkdir", $_);}


### define outputs

my @base = &basename($inputMRI, ".mnc");
my $nuc              = "${NUCDir}/$base[0]_nuc.mnc";
my $linXFM           = "${outXFMS}/$base[0]_lin.xfm";
my $nlXFM            = "${outXFMS}/$base[0]_nl.xfm";
my $linRes           = "${outResample}/$base[0]_lin.mnc";
my $nlRes            = "${outResample}/$base[0]_nl.mnc";
my $outLabelsLeft    = "${outSegmentation}/$base[0]_labels_left.mnc";
my $outLabelsRight   = "${outSegmentation}/$base[0]_labels_right.mnc"; 

### define tmp outputs 

my $nl0 = "${tmpdir}/nl0.xfm";
my $nl1 = "${tmpdir}/nl1.xfm";
my $nl2 = "${tmpdir}/nl2.xfm";

do_cmd('nu_correct', $inputMRI, $nuc);
do_cmd('mritotal', 
       $nuc, $linXFM,
       '-model', $colinGlobal,
       '-modeldir', $colinDir);
do_cmd('mincresample',
       '-transformation', $linXFM,
       '-like', $colinGlobalFull,
       '-sinc', '-width', '2',
       $nuc, $linRes);


##### Now the nonlinear stuff
my @minctraccArgs = qw(-stiffness 1 -weight 1 -similarity 0.3 -iterations 15  -nonlinear corrcoeff -debug);


do_cmd('minctracc',
       '-step', '4', '4', '4',
       '-lattice_diam', '12', '12', '12',
       '-sub_lattice', '8',
       '-ident',
      # '-transformation', $linXFM,
       @minctraccArgs,
       $linRes, $colinSubcorticalFull, $nl0);
			  

do_cmd('minctracc',
       '-step', '2', '2', '2',
       '-lattice_diam', '6', '6', '6',
       '-sub_lattice', '8',
       '-transformation', $nl0,
       '-ident',
       @minctraccArgs,
       $linRes, $colinSubcorticalFull, $nl1);

do_cmd('minctracc',
       '-step', '1', '1', '1',
       '-lattice_diam', '3', '3', '3',
       '-sub_lattice', '6',
       '-transformation', $nl1,
       @minctraccArgs,
       $linRes, $colinSubcorticalFull, $nl2);

do_cmd('xfmconcat', $linXFM,$nl2, $nlXFM);

do_cmd('mincresample',
       '-transformation', $nlXFM,
       '-like', $colinGlobalFull,
       '-sinc', '-width', '2',
       $nuc, $nlRes);


##### Resample labels back

do_cmd('mincresample',
       '-transformation', $nlXFM,
       '-invert', '-near', '-keep_real_range',
       '-like', $nuc,
       $labelsLeft, $outLabelsLeft);
do_cmd('mincresample',
       '-transformation', $nlXFM,
       '-invert', '-near', '-keep_real_range',
       '-like', $nuc,
       $labelsRight, $outLabelsRight);       

sub do_cmd{
    print "@_ \n";
    system(@_) == 0 or die;
}
