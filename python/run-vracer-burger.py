import argparse
### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--numactions', help='Number of actions', required=False, type=int, default=1)
parser.add_argument('--numexp', help='Number of experiences', required=False, type=int, default=5e5)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)
parser.add_argument('--test', action='store_true', help='Run tag', required=False)

args = parser.parse_args()

### Import modules

import sys
sys.path.append('_model')
import burger_environment as be


### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_result_burger_{}_{}_{}_{}_{}/'.format(args.ic, args.N, args.numactions, args.episodelength, args.run)
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n")

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Custom Settings"]["Mode"] = "Testing" if args.test else "Training"
e["Problem"]["Environment Function"] = lambda s : be.environment( s, args.N, args.numactions, args.episodelength, args.ic )
e["Problem"]["Testing Frequency"] = 100
e["Problem"]["Policy Testing Episodes"] = 1

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if args.test else "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 1.
e["Solver"]["Mini Batch"]["Size"] = 256

### Defining Variables

nState  = args.N*2
# States (flow at sensor locations)
for i in range(nState):
	e["Variables"][i]["Name"] = "Field Information " + str(i)
	e["Variables"][i]["Type"] = "State"

for i in range(args.numactions):
    e["Variables"][nState+i]["Name"] = "Forcing " + str(i)
    e["Variables"][nState+i]["Type"] = "Action"
    e["Variables"][nState+i]["Lower Bound"] = -0.1
    e["Variables"][nState+i]["Upper Bound"] = +0.1
    e["Variables"][nState+i]["Initial Exploration Noise"] = 0.01

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Start Size"] = 16384
e["Solver"]["Experience Replay"]["Maximum Size"] = 524288

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 256

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 256

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Generations"] = 1e6
e["Solver"]["Termination Criteria"]["Max Experiences"] = args.numexp
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 500
e["File Output"]["Path"] = resultFolder

if args.test:
    fileName = 'test_burger_{}_{}_{}_{}_{}'.format(args.ic, args.N, args.numactions, args.episodelength, args.run)
    e["Solver"]["Testing"]["Sample Ids"] = [0]
    e["Problem"]["Custom Settings"]["Filename"] = fileName

### Running Experiment

k.run(e)
