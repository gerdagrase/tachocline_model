# import the required modules
import math
import numpy as np

# define the coefficients of the model
def coef(m, b, beta, gamma):
    alpha_m = 8*math.sqrt(2)/math.pi*m**2/(4*m**2-1)*(b**2+m**2-1)/(b**2+m**2)
    beta_m = beta*b**2/(b**2+m**2)
    delta_m = 64*math.sqrt(2)/(15*math.pi)*(b**2-m**2+1)/(b**2+m**2)
    gamma_t_m = gamma*4*m/(4*m**2-1)*math.sqrt(2)*b/math.pi
    gamma_m = gamma*4*m**3/(4*m**2-1)*math.sqrt(2)*b/(math.pi*(b**2+m**2))
    return np.array([alpha_m, beta_m, delta_m, gamma_t_m, gamma_m])
    
# define the 6-dimensional model
def ode(T, x, b, beta, gamma, C, epsilon, xstar1, xstar4):
    coef_1 = coef(1, b, beta, gamma)
    coef_2 = coef(2, b, beta, gamma)
    
    dx1_dt = coef_1[3]*x[2] - C*(x[0] - xstar1)
    dx2_dt = -(coef_1[0]*x[0] - coef_1[1])*x[2] - C*x[1] - coef_1[2]*x[3]*x[5] 
    dx3_dt = (coef_1[0]*x[0] - coef_1[1])*x[1] - coef_1[4]*x[0] - C*x[2] + coef_1[2]*x[3]*x[4] 
    dx4_dt = coef_2[3]*x[5] -C*(x[3] - xstar4) + epsilon*(x[1]*x[5] - x[2]*x[4])
    dx5_dt = -(coef_2[0]*x[0] - coef_2[1])*x[5] - C*x[4] - coef_2[2]*x[2]*x[3]
    dx6_dt = (coef_2[0]*x[0] - coef_2[1])*x[4] - coef_2[4]*x[3] - C*x[5] + coef_2[2]*x[3]*x[1]
    return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt])