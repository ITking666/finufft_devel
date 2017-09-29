#!/usr/bin/python3

from accuracy_speed_tests import accuracy_speed_tests

import sys
import os, inspect

# append the parent path to search directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir+'/../../../mlpython1/mlpy') 

# imports from mlpy
from mlpy import ProcessorManager

class Processor:
	name='finufft.accuracy_speed_tests'
	inputs=[]
	outputs=[]
	parameters=[
		{"name":"num_nonuniform_points","optional":True,"default_value":1000},
		{"name":"num_uniform_points","optional":True,"default_value":1000},
		{"name":"eps","optional":True,"default_value":"1e-6"},
		{"name":"num_trials","optional":True,"default_value":"10"}
	]
	opts={"cache_output":False}
	def run(self,args):
		accuracy_speed_tests(int(args['num_nonuniform_points']),int(args['num_uniform_points']),float(args['eps']),int(args['num_trials']))
		return True

PM=ProcessorManager()
PM.registerProcessor(Processor())
if not PM.run(sys.argv):
	exit(-1)
