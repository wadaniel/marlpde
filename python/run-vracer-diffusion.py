import argparse
### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
#parser.add_argument('--NDNS', help='Discretization / number of grid points', required=False, type=int, default=512)
parser.add_argument('--dt', help='Time discretization', required=False, type=float, default=0.001)
parser.add_argument('--tend', help='Length of simulation', required=False, type=float, default=5)
parser.add_argument('--NA', help='Number of actions', required=False, type=int, default=32)
parser.add_argument('--NE', help='Number of experiences', required=False, type=int, default=5e5)
parser.add_argument('--width', help='Size of hidden layer', required=False, type=int, default=256)
parser.add_argument('--iex', help='Initial exploration', required=False, type=float, default=0.0001)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--noise', help='Standard deviation of IC', required=False, type=float, default=0.)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--dforce', help='Do direct forcing', action='store_true', required=False)
parser.add_argument('--nunoise', help='Enable noisy nu', action='store_true', required=False)
parser.add_argument('--tnoise', help='Enable noisy timestep', action='store_true', required=False)
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--nu', help='Viscosity', required=False, type=float, default=0.1)
parser.add_argument('--nt', help='Number of testing runs', required=False, type=int, default=1)
parser.add_argument('--tf', help='Testing frequenct in episodes', required=False, type=int, default=100)
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)
parser.add_argument('--version', help='Version tag', required=False, type=int, default=0)
parser.add_argument('--test', action='store_true', help='Run tag', required=False)


args = parser.parse_args()

### Import modules

import sys
sys.path.append('_model')
import diffusion_environment as de


### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

#dns_default = de.setup_dns_default(args.NDNS, args.dt, args.nu, args.tend, args.seed)
dns_default = None

### Defining results folder and loading previous results, if any

resultFolder = '_result_diffusion_{}/'.format(args.run)
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n")

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Custom Settings"]["Mode"] = "Testing" if args.test else "Training"
e["Problem"]["Environment Function"] = lambda s : de.environment( 
        s,
        N = args.N,
        tEnd = args.tend,
        dt_sgs = args.dt, 
        numActions = args.NA, 
        nu = args.nu,
        episodeLength = args.episodelength, 
        ic = args.ic, 
        dforce = args.dforce, 
        noise = args.noise, 
        seed = args.seed, 
        nunoise = args.nunoise,
        tnoise = args.tnoise,
        version = args.version,
        dns_default = dns_default
        )

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

nState = args.N if args.version == 0 else args.N*2
# States (flow at sensor locations)
for i in range(nState):
	e["Variables"][i]["Name"] = "Field Information " + str(i)
	e["Variables"][i]["Type"] = "State"

for i in range(args.NA):
    e["Variables"][nState+i]["Name"] = "Forcing " + str(i)
    e["Variables"][nState+i]["Type"] = "Action"
    e["Variables"][nState+i]["Lower Bound"] = -1.
    e["Variables"][nState+i]["Upper Bound"] = +1.
    e["Variables"][nState+i]["Initial Exploration Noise"] = args.iex

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
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.width

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.width

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Generations"] = 1e6
e["Solver"]["Termination Criteria"]["Max Experiences"] = args.NE
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 500
e["File Output"]["Path"] = resultFolder

if args.test:
    fileName = 'test_diffusion_{}_{}_{}_{}_{}'.format(args.ic, args.N, args.NA, args.seed, args.run)
    e["Solver"]["Testing"]["Sample Ids"] = [0]
    e["Problem"]["Custom Settings"]["Filename"] = fileName

if args.test:

    dts = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2]

    for dt in dts:
        fileName = 'test_diffusion_{}_{}_{}'.format(args.ic, dt, args.run)
        e["Solver"]["Testing"]["Sample Ids"] = [0]
        e["Problem"]["Custom Settings"]["Filename"] = fileName
        e["Problem"]["Custom Settings"]["Timestep"] = nu
        k.run(e)


### Running Experiment

k.run(e)
