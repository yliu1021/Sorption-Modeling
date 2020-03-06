import os

# Geometric constants
GRID_SIZE = 40             # size of the lattice
THRESHOLD = 2              # boundary of the lattice
N_ITER = 80                # total number of DFT iterations
N_ADSORP = 40              # total number of adsorption iterations
STEP_SIZE = 0.025          # amount RH is changed with each timestep
# N_ITER = 125               # total number of DFT iterations
# N_ADSORP = 75              # total number of adsorption iterations
# STEP_SIZE = 0.01           # amount RH is changed with each timestep
N_SQUARES = GRID_SIZE**2   # total number of boxes

# Physical constants
T = 298.0                  # temperature K
Y = 1.5
TC = 647.0                 # K  # critical temperature
KB = 0.0019872041          # kcal/(mol?K)  # boltsman constant
BETA = 1/(KB*T)
MUSAT = -2.0 * KB * TC     # saturation chemical potential
C = 4.0                    # kcal/(mol?K)  # square is 4, triangle is 3.
WFF = -2.0 * MUSAT/C       # water-water interaction   # interaction energy
WMF = Y * WFF              # water-matrix interaction  # interaction energy

# Directory constants
FEATURE_DIR = 'output'
RESULT_DIR = os.path.join(FEATURE_DIR, 'results')
GRID_DIR = os.path.join(FEATURE_DIR, 'grids')
PLOT_DIR = os.path.join(FEATURE_DIR, 'plots')
CURVE_DIR = os.path.join(FEATURE_DIR, 'curves')
HYS_DIR = 'data'

PREDICT_DIR = 'predict_ri'
TRAIN_SET_DIR = os.path.join(PREDICT_DIR, 'grids')
RESULT_SET_DIR = os.path.join(PREDICT_DIR, 'results')
DATA_SET_DIR = os.path.join(PREDICT_DIR, 'data')

MODEL_DIR = 'training'
MODEL_RESULT_DIR = os.path.join(MODEL_DIR, 'results')
MODEL_BASE_DIR = os.path.join(MODEL_DIR, 'models')

MC_DIR = 'predict_mc'
MC_HYS_DIR = os.path.join(MC_DIR, 'data')
MC_SRC_GRID_DIR = os.path.join(MC_DIR, 'grids')
MC_SRC_RESULT_DIR = os.path.join(MC_DIR, 'results')
MC_SRC_PLOT_DIR = os.path.join(MC_DIR, 'plots')
MC_SRC_CURVE_DIR = os.path.join(MC_DIR, 'curves')

MC_MODEL_DIR = os.path.join(MC_DIR, 'training')
MCM_RESULT_DIR = os.path.join(MC_MODEL_DIR, 'results')
MCM_BASE_DIR = os.path.join(MC_MODEL_DIR, 'models')

MC_ITER_DIR = os.path.join(MC_DIR, 'iteration_d02')
MC_GRID_DIR = MC_ITER_DIR
MC_GRID_SET_DIR = os.path.join(MC_ITER_DIR, 'grids')
MC_RESULT_SET_DIR = os.path.join(MC_ITER_DIR, 'results')

SWARM_DIR = 'predict_swarm'
SWARM_POS_DIR = os.path.join(SWARM_DIR, 'positions')
SWARM_VEL_DIR = os.path.join(SWARM_DIR, 'velocities')
SWARM_BEST_DIR = os.path.join(SWARM_DIR, 'bests')

# Grid constants
N_FILLED = 120
THRESH = 2
N_VAR = 80

# Feature constants
DATASETS = [
    ('1', 1000), ('2', 1000), ('3', 9) #, ('4', 1000), ('5', 1000), ('6', 1000)
]
VERSIONS = [ 
    'l', 'm', 
]
PORE_CHOICES = [
    'random',
    'growing',
]
MODELS = ['cnn'] #['nn4', 'nn16', 'nn64', 'nn256', 'nn1024', 'nn4096', 'cnn', 'lin']

# Training constants
TRAIN_FRAC = 0.7
VERBOSE = True
N_GENERATED = 10000
N_OPTIMAL = 200
N_STEPS = 10

MC_ACCEPT = 0.5
MC_GROW_PROB = 0.5
INIT_ALPHA = 0.005
INIT_SIGMA = 0.025
MAX_ATTEMPTS = 100
N_MC_ITER = 2000
N_RES_PARAMS = 6

SWARM_PMIN = 0.0025
SWARM_PMAX = 0.9975
SWARM_COUNT = 20
N_SWARM_ITER = 100
P_WEIGHT = 0.5
G_WEIGHT = 0.5
