import os
import json
import inspect
from . import docscrape
import traceback

class ProcessorManager:
    _processors=[]
    def registerProcessor(self,processor):
        self._processors.append(processor)
    def run(self,argv):
        if (len(argv)<=1):
            arg1='spec' #the default
        else:
            arg1=argv[1]
        if (arg1 == 'spec'):
            spec=self.getSpec(argv)
            print (json.dumps(spec, sort_keys=True, indent=4))
            return True
        if (arg1 == 'test'):
            test_mode=True
            processor_name = argv[2] if 2<len(argv) else ''
        else:
            test_mode=False
            processor_name=arg1
        args=self._get_args_from_argv(argv)
        if not test_mode:
            P=self.findProcessor(processor_name)
            if P is None:
                print ("Unable to find processor: {}".format(processor_name))
                return False
            if not self._check_args(P,args):
                return False
            return P(args)
        else:
            # test mode
            if (processor_name):
                self._run_test(processor_name,args)
            else:
                for ii in range(len(self._processors)):
                    P=self._processors[ii]
                    self._run_test(P.name,args)
            return True
    def _run_test(self,processor_name,args):
        P=self.findProcessor(processor_name)
        if P is None:
            print ("Unable to find processor: {}".format(processor_name))
            return
        if hasattr(P,'test'):
            print ('')
            print ('----------------------------------------------')
            print ('Testing %s' % (P.name))
            try:
                if P.test(args):
                    print ('SUCCESS')
                else:
                    print ('FAILURE')
            except:
                traceback.print_exc()
                print ('FAILURE')
            print ('----------------------------------------------')
        else:
            print ('No test function defined for %s' % (P.name))
    def getSpec(self,argv):
        spec={"processors":[]}
        for j in range(0,len(self._processors)):
            obj=self.getProcessorSpec(self._processors[j])
            program=os.path.abspath(argv[0])
            obj["exe_command"]="python3 {} {} $(arguments)".format(program,self._processors[j].name)
            spec["processors"].append(obj)
        return spec
    def getProcessorSpec(self,P):
        assert(callable(P))
        spec={"name":P.name,"version":P.version}
        npdoc=docscrape.FunctionDoc(P)
        #npdoc={"Summary":"","Parameters":[]}
        spec["description"]=npdoc["Summary"];
        params0=npdoc["Parameters"]
        argspec0=inspect.getfullargspec(P.__call__ if inspect.isclass(P) else P)
        defaults0=argspec0.kwonlydefaults;
        inputs,outputs,parameters = [],[],[]
        for j in range(len(params0)):
            pp=params0[j]
            pname=pp[0]
            ptype=pp[1]
            pdescr=pp[2]
            qq={"name":pname,"description":pdescr}
            if pname in defaults0:
                qq["optional"]=True
                qq["default_value"]=defaults0[pname]
            if (ptype=='INPUT'):
                inputs.append(qq)
            elif (ptype=='OUTPUT'):
                outputs.append(qq)
            else:
                qq["datatype"]=ptype
                parameters.append(qq)
        spec['inputs']=inputs
        spec['outputs']=outputs
        spec['parameters']=parameters
        if hasattr(P,'test'):
            spec['has_test']=True
        return spec
    def findProcessor(self,processor_name):
        for j in range(0,len(self._processors)):
            if (self._processors[j].name == processor_name):
                return self._processors[j]
        return None
    def _get_args_from_argv(self,argv):
        args={}
        for j in range(1,len(argv)):
            arg0=argv[j]
            if (arg0.startswith("--")):
                tmp=arg0[2:].split("=")
                if (len(tmp)==2):
                    if (tmp[0] in args):
                        if type(args[tmp[0]]=='list'):
                            args[tmp[0]].append(tmp[1]) #already a list, so append
                        else:
                            args[tmp[0]]=[args[tmp[0]],tmp[1]] #not a list yet, so make it a list and append
                    else:
                        args[tmp[0]]=tmp[1] #not a list
                else:
                    print ("Warning: problem with argument: {}".format(arg0))
                    exit(-1)
            #else:
            #    print ("Warning: problem with argument: {}".format(arg0))
            #    exit(-1)
        return args
    def _check_args(self,P,args):
        valid_params={}
        for j in range(0,len(P.inputs)):
            valid_params[P.inputs[j]["name"]]=1
            if not P.inputs[j]["name"] in args:
                print ("Missing input path: {}".format(P.inputs[j]["name"]))
                return False
        for j in range(0,len(P.outputs)):
            valid_params[P.outputs[j]["name"]]=1
            if not P.outputs[j]["name"] in args:
                print ("Missing output path: {}".format(P.outputs[j]["name"]))
                return False
        for j in range(0,len(P.parameters)):
            valid_params[P.parameters[j]["name"]]=1
            if not P.parameters[j]["optional"]:
                if not P.parameters[j]["name"] in args:
                    print ("Missing required parameter: {}".format(P.parameters[j]["name"]))
                    return False
        for key in args:
            if not key in valid_params:
                if not key.startswith("_"):
                    print ("Invalid parameter: {}".format(key))
                    return False
        return True
