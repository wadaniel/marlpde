import argparse
import numpy as np
### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=128)
parser.add_argument('--NDNS', help='Discretization / number of grid points', required=False, type=int, default=512)
parser.add_argument('--numAgents', help='Number of agents', required=False, type=int, default=1)
#parser.add_argument('--dt', help='Time discretization', required=False, type=float, default=0.01)
parser.add_argument('--exp', help='Number of experiences', required=False, type=int, default=1e6)
parser.add_argument('--width', help='Size of hidden layer', required=False, type=int, default=128)
parser.add_argument('--iex', help='Initial exploration', required=False, type=float, default=0.05)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--noise', help='Standard deviation of IC', required=False, type=float, default=0.)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='sinus')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--nu', help='Viscosity', required=False, type=float, default=0.5)
parser.add_argument('--tf', help='Testing frequency in episodes', required=False, type=int, default=500)
parser.add_argument('--nt', help='Number of testing runs', required=False, type=int, default=20)
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)
parser.add_argument('--version', help='Version tag', required=False, type=int, default=0)
parser.add_argument('--test', action='store_true', help='Run tag', required=False)

args = parser.parse_args()
T = 2*np.pi/args.nu
dt = T / args.episodelength

### Import modules
import sys
sys.path.append('_model')
import advection_environment_simple as ad

### Defining Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

#dns_default = ad.setup_dns_default(args.ic, args.NDNS, dt, args.nu, T, args.seed)

### Defining results folder and loading previous results, if any

resultFolder = '_result_advection_simple_{}/'.format(args.run)
if args.test:
    found = e.loadState(resultFolder + '/latest')
    if found == True:
	    print("[Korali] Continuing execution from previous run...\n")

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Agents Per Environment"] = args.numAgents
e["Problem"]["Policies Per Environment"] = 1
e["Problem"]["Custom Settings"]["Mode"] = "Testing" if args.test else "Training"
e["Problem"]["Environment Function"] = lambda s : ad.environment( 
        s,
        N = args.N,
        tEnd = T,
        dtSgs = dt, 
        nu = args.nu,
        episodeLength = args.episodelength, 
        ic = args.ic, 
        noise = args.noise, 
        seed = args.seed, 
        nunoise = False,
        numAgents = args.numAgents,
        dnsDefault = None
        )

e["Problem"]["Testing Frequency"] = args.tf
e["Problem"]["Policy Testing Episodes"] = args.nt if args.noise > 0. else 1
#e["Problem"]["Policy Testing Episodes"] = 0.

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if args.test else "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.95
e["Solver"]["Mini Batch"]["Size"] = 256

### Defining Variables

nState = args.N if args.numAgents == 1 else int(args.N/args.numAgents)+2
# States (flow at sensor locations)
for i in range(nState):
	e["Variables"][i]["Name"] = "Field Information " + str(i)
	e["Variables"][i]["Type"] = "State"

for i in range(1):
    e["Variables"][nState+i]["Name"] = "Forcing " + str(i)
    e["Variables"][nState+i]["Type"] = "Action"
    e["Variables"][nState+i]["Lower Bound"] = -2
    e["Variables"][nState+i]["Upper Bound"] = +2
    e["Variables"][nState+i]["Initial Exploration Noise"] = args.iex

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Start Size"] = 32768
e["Solver"]["Experience Replay"]["Maximum Size"] = 1048576


e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.width

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.width

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.exp
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 500
e["File Output"]["Path"] = resultFolder

if args.test:
    e["Solver"]["Testing"]["Sample Ids"] = [0]

### Running Experiment
k.run(e)
print(args)
