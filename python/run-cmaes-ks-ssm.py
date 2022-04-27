import argparse
### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--NDNS', help='Discretization / numker of grid points', required=False, type=int, default=2048)
parser.add_argument('--gridSize', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--numgen', help='Number of generations', required=False, type=int, default=1000)
parser.add_argument('--pop', help='Population size', required=False, type=int, default=8)
parser.add_argument('--dt', help='Simulation timesteps', required=False, type=float, default=0.1)
parser.add_argument('--nu', help='Viscosity', required=False, type=float, default=1)
parser.add_argument('--noise', help='IC noise', required=False, type=float, default=0.0)
parser.add_argument('--specreward', help='Calculate spectral reward', required=False, action='store_true')
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='noise')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--run', help='Run tag', required=False, type=int, default=0)
parser.add_argument('--ssm', help='Closure type', required=False, type=int, default=1)
parser.add_argument('--dsm', help='Closure type', required=False, type=int, default=0)

args = parser.parse_args()

### Import modules

import sys
sys.path.append('_model')
import ks_cmaes_ssm as ke

### Import modules
dns_default = ke.setup_dns_default(N=args.NDNS, dt=args.dt, nu=args.nu, seed=args.seed)

### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_cmaes_ks_{}_{}_{}/'.format(args.ic, args.gridSize, args.run)
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n")

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = lambda s : ke.fKS(
        s,
        N = args.NDNS,
        gridSize = args.gridSize,
        dt = args.dt,
        nu = args.nu,
        episodeLength = args.episodelength,
		spectralReward = args.specreward,
		noise = args.noise,
        seed = args.seed,
		ssm = args.ssm,
		dsm = args.dsm,
        dns_default = dns_default )
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
