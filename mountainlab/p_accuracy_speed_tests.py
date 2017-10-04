import sys
import numpy as np

from mlpy import ProcessorManager
from accuracy_speed_tests import accuracy_speed_tests

def p_accuracy_speed_tests(*,num_nonuniform_points=1000,num_uniform_points=1000,eps=1e-6,num_trials=10,random_seed=0):
    """
    Perform accuracy and speed tests for finufft

    Parameters
    ----------    
    num_nonuniform_points : int
        (Optional) Number of non-uniform points for the tests
    num_uniform_points : int
        (Optional) Number of uniform points for the tests
    eps : double
        (Optional) The precision for calculating the nufft
    num_trials : int
        (Optional) Number of runs per test for purpose of timing
    random_seed : int
        (Optional) A random seed to initialize the random number generator prior to generating the example arrays
    """    
    np.random.seed(random_seed)
    accuracy_speed_tests(num_nonuniform_points,num_uniform_points,eps,num_trials)
    return True
p_accuracy_speed_tests.name='finufft.accuracy_speed_tests'
p_accuracy_speed_tests.version="0.1"
def test_accuracy_speed_tests(args):
    ret=p_accuracy_speed_tests()
    assert(ret)
    return True
p_accuracy_speed_tests.test=test_accuracy_speed_tests

if len(sys.argv)==1:
    sys.argv.append('test')
    sys.argv.append('finufft.accuracy_speed_tests')

PM=ProcessorManager()
PM.registerProcessor(p_accuracy_speed_tests)
if not PM.run(sys.argv):
    exit(-1)
