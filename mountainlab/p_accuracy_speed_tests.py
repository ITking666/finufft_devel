#!/usr/bin/python3

from accuracy_speed_tests import *

class Processor:
	name='finufft.accuracy_speed_tests'
	inputs=[]
	outputs=[]
	parameters=[
		{"name":"num_nonuniform_points","optional":True,"default_value":1000},
		{"name":"num_uniform_points","optional":True,"default_value":1000},
		{"name":"eps","optional":True,"default_value":"1e-6"}
	]
	opts={"cache_output":False}
	def run(self,args):
		accuracy_speed_tests(int(args['num_nonuniform_points']),int(args['num_uniform_points']),float(args['eps']))
		return True