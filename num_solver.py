# import the required modules
import math
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd 
import argparse
import datetime
import importlib
import os 

# define parsers for the coefficients of the model and their default values; default values of the bve_model have been defined in 
# "A Mechanism for Atmospheric Regime Behavior" paper by D.T. Crommelin et al. (2003)
parser = argparse.ArgumentParser(description="Numerical solver for barotropic vorticity equation and shallow water model")
default_epsilon = 16*math.sqrt(2)/(5*math.pi)

parser.add_argument("--dir",     default='./',             type=str,   help="working directory")
parser.add_argument("--label",   default='soln',           type=str,   help="file label")
parser.add_argument("--model",   default='bve_model',      type=str,   help="choice of model: bve_model or shallow_model")
parser.add_argument("--t0",      default=0,                type=float, help="start time")
parser.add_argument("--tf",      default=4000,             type=float, help="end time")
# parameters bve_model
parser.add_argument("--epsilon", default=default_epsilon,  type=float, help="multiplication coefficient")
parser.add_argument("--C",       default=0.1,              type=float, help="multiplication coefficient")
parser.add_argument("--xstar1",  default=0.95,             type=float, help="forcing profile value")
parser.add_argument("--r",       default=-0.801,           type=float, help="ratio of the two forcing profile values")
# parameters shallow water model
parser.add_argument("--rho_0",   default=1,    type=float, help="plasma density")
parser.add_argument("--g",       default=1,    type=float, help="effective gravity")
parser.add_argument("--f0",      default=1,    type=float, help="equatorial rotation rate")
parser.add_argument("--s2",      default=1,    type=float, help="differential rotation coefficient")
parser.add_argument("--s4",      default=1,    type=float, help="differential rotation coefficient")
parser.add_argument("--pert",    default=1e-5, type=float, help="uniform initial value")
parser.add_argument("--diss",    default=10,   type=float, help="dissipation coefficient")
# parameters both models
parser.add_argument("--b",       default=0.5,  type=float, help="aspect ratio of beta-plane channel")
parser.add_argument("--beta",    default=1.25, type=float, help="beta coefficient (beta-plane approx)")
parser.add_argument("--gamma",   default=0.2,  type=float, help="multiplication coefficient")
# restart feature
parser.add_argument("--restart", action='store_true',      help="restart?")
parser.add_argument("--rfile",   default=' ',  type=str,   help="restart file name - optional")

args = parser.parse_args()

# reassign coefficients to their parsed values
directory = args.dir + '/'
label = args.label
model = args.model
t0 = args.t0
tf = args.tf
beta = args.beta
gamma = args.gamma
epsilon = args.epsilon
C = args.C
b = args.b
xstar1 = args.xstar1
r = args.r 
rho_0 = args.rho_0
d = args.diss
g = args.g
f0 = args.f0
s2 = args.s2
s4 = args.s4
pert = args.pert

# restart
rfile = args.rfile
restart = args.restart

# import the model specified in the variable "model"
import_model = importlib.import_module(model)

# log run parameters
f = open(directory + 'trial_values.args','a')
timestamp = str(datetime.datetime.now()).split('.')[0] # remove milliseconds
print(timestamp, args, file = f)
f.close()

# following enables initialising from simulation at different parameter values
if (rfile != ' '):
   if not os.path.isfile(directory + rfile):
      print("restart file " + directory + rfile + " not found.")
      print("Exiting.")
      exit()

   # reading t0 and initial data from last line of file 
   # for t0, this takes precedence over t0 from command line
   df = pd.read_csv(directory + rfile) 
   lastline = df.iloc[-1,:].values

# restart feature to append data to a file
if restart: 
   t0 = lastline[0]

   if (t0 > tf):
     raise ValueError("initial time larger than final time - check tf parameter")

   print('\nrestart from ', directory + rfile)
   print('at time ', t0)

   # append data to restart file 
   outfile = directory + rfile
   write_mode = 'a'
   header = False # do not write header when appending to file

else:
   print('newly initialised simulation')
   print('start at time ', t0)

   # write data to new file, to be specified for different models below 
   write_mode = 'w+'
   header = True # write header when writing new file
   
# define the time series
npoints = int((tf-t0) * 250) # output set to 250 points per time unit
T = np.linspace(t0, tf, npoints)

if model == 'bve_model':
    # initialise an empty array for the initial values
    initial = np.zeros(6)

    if (rfile != ' '):
        # define initial values
        initial = lastline[1:]

    else:
        # relationship between xstar1 and xstar4
        xstar4 = r*xstar1
        # define initial values
        initial = [xstar1, 0, 0, xstar4, 0, 0]
          
    if not restart:
        # write data to new file 
        outfile = directory + label + '-gamma_' + str(gamma) + '-xstar1_' + str(xstar1) + '-xstar4_' + str(xstar4) + '-C_' + str(C) + '-beta_' + str(beta) + '-b_' + str(b) + '-eps_' + str(epsilon) + '.csv'

    
    # set parameters, solve the model for the initial data
    params = (b, beta, gamma, C, epsilon, xstar1, xstar4)
    sol_ode = solve_ivp(import_model.ode, [t0, tf], initial, method ='RK45', t_eval = T, args=params)
    graph_data = pd.DataFrame(data={'time': sol_ode.t[:],
        'x_1_sol': sol_ode.y[0,:], 'x_2_sol': sol_ode.y[1,:], 'x_3_sol': sol_ode.y[2,:], 
        'x_4_sol': sol_ode.y[3,:], 'x_5_sol': sol_ode.y[4,:], 'x_6_sol': sol_ode.y[5,:]})
        

elif model == 'shallow_model':
    # define initial values
    initial = np.zeros(18)

    if (rfile != ' '):
        initial = lastline[1:]

    else:
        # initialise z to zero
        initial = np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        initial *= pert
        
    if not restart:
        # write data to new file 
        outfile = directory + label + '-diss_' + str(d) + '-s2_' + str(s2) + '-s4_' + str(s4) + '-g_' + str(g) + '-beta_' + str(beta) + '-b_' + str(b) + '.csv' 

    # set parameters, solve the model for the initial data
    params = (rho_0, g, f0, beta, b, s2, s4, d)
    sol_ode = solve_ivp(import_model.ode, [t0, tf], initial, method ='LSODA', t_eval = T, args=params, max_step=0.0001, rtol=1e-7, atol=1e-9)
    
    graph_data = pd.DataFrame(data={'time': sol_ode.t[:], 
        'x_1_sol': sol_ode.y[0,:],  'x_2_sol': sol_ode.y[1,:],  'x_3_sol': sol_ode.y[2,:],  
        'x_4_sol': sol_ode.y[3,:],  'x_5_sol': sol_ode.y[4,:],  'x_6_sol': sol_ode.y[5,:], 
        'y_1_sol': sol_ode.y[6,:],  'y_2_sol': sol_ode.y[7,:],  'y_3_sol': sol_ode.y[8,:], 
        'y_4_sol': sol_ode.y[9,:],  'y_5_sol': sol_ode.y[10,:], 'y_6_sol': sol_ode.y[11,:],
        'z_1_sol': sol_ode.y[12,:], 'z_2_sol': sol_ode.y[13,:], 'z_3_sol': sol_ode.y[14,:], 
        'z_4_sol': sol_ode.y[15,:], 'z_5_sol': sol_ode.y[16,:], 'z_6_sol': sol_ode.y[17,:]})

        
# save results in a .csv file
graph_data.to_csv(outfile, mode = write_mode, index=False, header = header)