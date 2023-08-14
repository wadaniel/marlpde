import argparse
### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--numAgents', help='Number of agents', required=False, type=int, default=32)
parser.add_argument('--dt', help='Time discretization', required=False, type=float, default=0.01)
parser.add_argument('--exp', help='Number of experiences', required=False, type=int, default=1e6)
parser.add_argument('--width', help='Size of hidden layer', required=False, type=int, default=128)
parser.add_argument('--iex', help='Initial exploration', required=False, type=float, default=0.1)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=100)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='one')
parser.add_argument('--noise', help='IC noise', required=False, type=float, default=0.)
parser.add_argument('--force', help='f(x)', required=False, type=str, default='zero')
parser.add_argument('--tf', help='Testing frequency in episodes', required=False, type=int, default=100)
parser.add_argument('--nt', help='Number of testing runs', required=False, type=int, default=10)
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)
parser.add_argument('--version', help='Version tag', required=False, type=int, default=0)
parser.add_argument('--test', action='store_true', help='Run tag', required=False)

args = parser.parse_args()
print(args)
T = args.dt*args.episodelength

### Import modules
import sys
sys.path.append('_model')
import laplace_environment as de

### Defining Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_result_laplace_{}/'.format(args.run)
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n")

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Agents Per Environment"] = args.numAgents
e["Problem"]["Policies Per Environment"] = 1
e["Problem"]["Custom Settings"]["Mode"] = "Testing" if args.test else "Training"
e["Problem"]["Environment Function"] = lambda s : de.environment( 
        s,
        N = args.N,
        dtSgs = args.dt, 
        episodeLength = args.episodelength, 
        ic = args.ic, 
        sforce = args.force, 
        noise = args.noise,
        numAgents = args.numAgents,
        version = args.version
        )

e["Problem"]["Testing Frequency"] = args.tf
e["Problem"]["Policy Testing Episodes"] = args.nt

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if args.test else "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.95
e["Solver"]["Mini Batch"]["Size"] = 256

### Defining Variables

nState = 4
assert args.N % args.numAgents == 0, f"[run-vracer-laplace] only works with N%numAgents==0 agents"
nAct = args.N // args.numAgents
# States (flow at sensor locations)
for i in range(nState):
	e["Variables"][i]["Name"] = "Field and Force Information " + str(i)
	e["Variables"][i]["Type"] = "State"

for i in range(3):
    e["Variables"][nState+i]["Name"] = "Forcing " + str(i)
    e["Variables"][nState+i]["Type"] = "Action"
    e["Variables"][nState+i]["Lower Bound"] = -3.
    e["Variables"][nState+i]["Upper Bound"] = +3.
    e["Variables"][nState+i]["Initial Exploration Noise"] = args.iex

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Start Size"] = 262144
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

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.exp
e["Solver"]["Termination Criteria"]["Max Generations"] = 1000000
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 50
e["File Output"]["Path"] = resultFolder

if args.test:
    e["Solver"]["Testing"]["Sample Ids"] = [0]

### Running Experiment
k.run(e)
print(args)
