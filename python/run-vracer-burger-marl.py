import argparse
import numpy as np

### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--NDNS', help='Discretization / number of grid points of DNS', required=False, type=int, default=512)
parser.add_argument('--N', help='Discretization / number of grid points of UGS', required=False, type=int, default=32)
parser.add_argument('--NA', help='Number of actions', required=False, type=int, default=32)
parser.add_argument('--NE', help='Number of experiences', required=False, type=int, default=5e5)
parser.add_argument('--width', help='Size of hidden layer', required=False, type=int, default=256)
parser.add_argument('--iex', help='Initial exploration', required=False, type=float, default=0.1)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--noise', help='Standard deviation of IC', required=False, type=float, default=0.)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='sinus')
parser.add_argument('--L', help='Length of domain', required=False, type=float, default=2*np.pi)
parser.add_argument('--dforce', help='Do direct forcing', action='store_true', required=False)
parser.add_argument('--ssmforce', help='Use SSM forcing', action='store_true', required=False)
parser.add_argument('--specreward', help='Use spectral reward', action='store_true', required=False)
parser.add_argument('--forcing', help='Use forcing term in equation', action='store_true', required=False)
parser.add_argument('--nunoise', help='Enable noisy nu', action='store_true', required=False)
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--stepper', help='Step factor in URG', required=False, type=int, default=1)
parser.add_argument('--dt', help='Simulator time step', required=False, type=float, default=0.001)
parser.add_argument('--T', help='Duration simulation', required=False, type=float, default=5.)
parser.add_argument('--nu', help='Viscosity', required=False, type=float, default=0.02)
parser.add_argument('--nt', help='Number of testing runs', required=False, type=int, default=20)
parser.add_argument('--tf', help='Testing frequenct in episodes', required=False, type=int, default=50)
parser.add_argument('--ssm', help='Static Smagorinksy Model', action='store_true', required=False)
parser.add_argument('--dsm', help='Dynamic Smagorinksy Model', action='store_true', required=False)
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)
parser.add_argument('--version', help='Version tag', required=False, type=int, default=1)
parser.add_argument('--ndns', help='Number of dns', required=False, type=int, default=1)
parser.add_argument('--test', action='store_true', help='Run tag', required=False)

# MARL configs
parser.add_argument('--nagents', help='Number of agents', required=False, type=int, default=2)
parser.add_argument('--mar', help='Multi Agent Relationship', required=False, type=str, default="Individual")
parser.add_argument('--mac', help='Multi Agent Correlation', required=False, type=bool, default=False)

args = parser.parse_args()
print(args)

assert args.nagents > 1, "Script must be run with 2 agents or more."


### Import modules

import sys
sys.path.append('_model')
import burger_environment as be

dns_default = None
dns_default = [ be.setup_dns_default(args.L, args.NDNS, args.T, args.dt, args.nu, args.ic, args.forcing, args.seed+i, args.stepper) for i in range(args.ndns) ]

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_result_vracer_marl_{}/'.format(args.run)

found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n");

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Testing Frequency"] = args.tf
e["Problem"]["Policy Testing Episodes"] = args.nt

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Custom Settings"]["Mode"] = "Testing" if args.test else "Training"
e["Problem"]["Environment Function"] = lambda s : be.environment( 
        s,
        L = args.L,
        T = args.T, 
        N = args.NDNS, 
        gridSize = args.N, 
        numActions = args.NA, 
        dt = args.dt, 
        nu = args.nu, 
        episodeLength = args.episodelength, 
        ic = args.ic, 
        spectralReward = args.specreward,
        forcing = args.forcing,
        dforce = args.dforce, 
        ssmforce = args.ssmforce,
        noise = args.noise, 
        seed = args.seed, 
        stepper = args.stepper,
        nunoise = args.nunoise,
        version = args.version,
        ssm = args.ssm,
        dsm = args.dsm,
        dns_default = dns_default,
        numAgents = args.nagents )

e["Problem"]["Testing Frequency"] = args.tf
e["Problem"]["Policy Testing Episodes"] = args.nt

#Defining MARL setup
e["Problem"]["Agents Per Environment"] = args.nagents
e["Problem"]["Policies Per Environment"] = 1
 
e["Solver"]["Multi Agent Relationship"] = args.mar
e["Solver"]["Multi Agent Correlation"] = args.mac
#e["Solver"]["Strong Truncation Variant"] = False

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if args.test else "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 0.5
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 1.
e["Solver"]["Mini Batch"]["Size"] = 256

### Defining Variables

if args.version == 0:
    nState  = args.N
elif args.version == 1:
    nState  = 2*args.N
elif args.version == 2:
    nState  = 2*args.N
elif args.version == 3:
    nState  = args.N + args.nagents * int(0.5*args.N)
else:
    print("[run-vracer-burger] version not recognized")
    sys.exit()

# States 
assert nState/args.nagents % 1 == 0., "Number of agents must be a divisor of num states"
numEffectiveStates = nState // args.nagents + 2
for i in range(numEffectiveStates):
	e["Variables"][i]["Name"] = "Field Information " + str(i)
	e["Variables"][i]["Type"] = "State"

# Actions 
assert args.NA/args.nagents % 1 == 0., "Number of agents must be a divisor of num actions"
numEffectiveActions = args.NA // args.nagents
for i in range(numEffectiveActions):
    e["Variables"][numEffectiveStates+i]["Name"] = "Forcing " + str(i)
    e["Variables"][numEffectiveStates+i]["Type"] = "Action"
    e["Variables"][numEffectiveStates+i]["Lower Bound"] = -5.
    e["Variables"][numEffectiveStates+i]["Upper Bound"] = +5.
    e["Variables"][numEffectiveStates+i]["Initial Exploration Noise"] = args.iex

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Start Size"] = 20000 * args.episodelength // 500
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000 * args.episodelength // 500

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 1.0

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
e["File Output"]["Frequency"] = 15
e["File Output"]["Path"] = resultFolder
e["File Output"]["Use Multiple Files"] = False

if args.test:

    nus = [0.02]

    for nu in nus:
        fileName = './plots/test_burger_marl_{}_{}_{}'.format(args.ic, nu, args.run)
        e["Solver"]["Testing"]["Sample Ids"] = [0]
        e["Problem"]["Custom Settings"]["Filename"] = fileName
        e["Problem"]["Custom Settings"]["Viscosity"] = nu
        k.run(e)

### Running Experiment
else:
    k.run(e)
