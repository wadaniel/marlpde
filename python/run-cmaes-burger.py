import argparse
### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--NDNS', help='Discretization / number of grid points', required=False, type=int, default=512)
parser.add_argument('--gridSize', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--numgen', help='Number of generations', required=False, type=int, default=1000)
parser.add_argument('--pop', help='Population size', required=False, type=int, default=8)
parser.add_argument('--dt', help='Simulation timesteps', required=False, type=float, default=0.001)
parser.add_argument('--nu', help='Viscosity', required=False, type=float, default=0.02)
parser.add_argument('--noise', help='IC noise', required=False, type=float, default=0.0)
parser.add_argument('--specreward', help='Calculate spectral reward', required=False, action='store_true')
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='sinus')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)

args = parser.parse_args()

### Import modules

import sys
sys.path.append('_model')
import burger_cmaes as be

### Import modules
dns_default = be.setup_dns_default(args.NDNS, args.dt, args.nu, args.ic, args.seed)

### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_cmaes_burger_{}_{}_{}/'.format(args.ic, args.gridSize, args.run)
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n")

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = lambda s : be.fBurger( s , args.NDNS, args.gridSize, args.dt, args.nu, args.episodelength, args.ic, args.specreward, args.noise, args.seed, dns_default)

e["Variables"][0]["Name"] = "CS"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = args.pop
e["Solver"]["Mu Value"] = args.pop//4
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-16
e["Solver"]["Termination Criteria"]["Max Generations"] = args.numgen

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = resultFolder
e["File Output"]["Frequency"] = 1

### Running Experiment

k.run(e)
