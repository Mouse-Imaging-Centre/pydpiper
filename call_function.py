#!/usr/bin/env python
#
#  call_function.py
#
#  May 30, 2011, MICe
#  Last Updated: May 30, 2011, MICe


import pipeline 
import commands 
import string 
import optparse
import os
import cPickle as pickle
import Pyro.core
import sys
import networkx as nx


Pyro.config.PYRO_MOBILE_CODE=1

#---------------------------------------------------------------------------------
#

program_name = 'call_function.py'

if __name__ == '__main__':	
	Pyro.core.initClient()	

	desc="""This small program requests pickled information from the running pipeline, unpickles it, and parses it for useful information."""
	usage = "No command-line options, just call it."
	parser = optparse.OptionParser(description=desc, usage=usage)	
	#parser.add_option("--clobber", action="store_true", dest="clobber", help="MINC volume containing labelled structures")
	
	#(options,args) = parser.parse_args() 
	#input_file=args[0]
	#input_file_short = os.path.basename(input_file)[:-4]
	#input_filer = os.path.expanduser("~/Desktop/"+input_file_short)
	
	uf = open('uri')
	uri = Pyro.core.processStringURI(uf.readline())
	uf.close()
	p = Pyro.core.getProxyForURI(uri)

	try:
		number = p.howFarDone()
		print number
	except Pyro.errors.ConnectionClosedError:
		sys.exit("Connection with server closed. Server shutdown and system exit.")
	#except:
	#	sys.exit("An error has occurred. Pipeline may not have completed properly. Check logs and restart if needed.")
