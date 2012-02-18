#!/usr/bin/env python 
import sys
import re
import socket

log = sys.argv[1]
f = open(log)


commands = []
started  = {}
finished = []
for line in f:
	m = re.match(r"""\d+\s+(.*)$""", line)
	if m: 
		commands.append( m.group(1) )
		continue		
	m = re.match(r"""Starting Stage \d+: (.*) \(PYRO://(.*):.*\)$""", line)
	if m:
		started[ m.group(1) ] = m.group(2) 
		continue

	m = re.match(r"""Finished Stage \d+: (.*)$""", line)
	if m:
		finished.append( m.group(1) )
		continue

active = [(started[i], i) for i in set(started.keys()) - set(finished)]
active.sort(key=lambda x: x[0])
hosts = {}
print len(active), "active tasks."
print len(set(commands) - set(finished)), " task remaining."
lasthostname = None
LINELENGTH = 270
for ip, cmd in active: 
	hostname = hosts.get(ip, socket.gethostbyaddr(ip)[0])
	if not hostname == lasthostname:
		print
		print hostname
		print "-"*(LINELENGTH+5)
	print "%s" % (cmd[:LINELENGTH]+" ... ")
	lasthostname = hostname
