#!/usr/bin/env bash

module purge
export fh=$(mktemp)
module load use.own gcc intel/14.0.1 python/2.7.8 gotoblas hdf5 gnuplot Xlibraries octave quarantine_tigger 2> $fh
cat $fh | tee /dev/stderr | grep 'ERROR' && exit 13
rm $fh #TODO ensure it's removed even on error
