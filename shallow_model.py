# import the required modules
import math
import numpy as np

# define the coefficients of the model
def coef_n(n, b, s_2, s_4, beta, f_0, g, rho_0, c): # n - dependent coefficients

    A_n = s_2*(math.pi**2/6 + 2/(math.pi*n**2)) + s_4*(math.pi**4/20 + math.pi/(4*n**2))
    C_n = -b * (s_2 * math.pi**3/12 + s_2 * math.pi/(4*n*n) + s_4 * math.pi**5/40 + s_4 * (math.pi**3/2 - 3*math.pi/(n*n))/(4*n*n) ) 
    
    E_n = (A_n*f_0 + C_n*beta)/(rho_0*g)

    alpha_n = 16*math.sqrt(2)/(3*math.pi)*n**2/(n**2 + 1)
    gamma_n = 32*math.sqrt(2)/(9*math.pi*b*n**2)

    c_n     = c * (1 + n*n/(b*b))

    return np.array([A_n, alpha_n, gamma_n, E_n, c_n])
    

def coef(b, s_2, s_4, beta, f_0, g, rho_0): # n - independent coefficients

    B   = 2/(math.pi*b)*(s_4*(320/(27*math.pi) + 8*math.pi/6) - 16*s_2/(9*math.pi))
    d   = -b * (s_2 * (12/math.pi - math.pi ) - (4/(3*math.pi) - math.pi)/9 + s_4 * (4*math.pi**3/9 - 12*math.pi*(1/81 -1) + 240/math.pi*(1/729 - 1) ) )

    D = 2 * d * beta/(rho_0*g)

    delta = math.sqrt(2)/b
    eta = beta*b/3
    mu = f_0 + beta*b*math.pi**2/2

    return np.array([B, D, delta, eta, mu])

    
# define the 18-dimensional model, separating the three spatial dimensions
def ode_x(T, x, y, z, coef_1, coef_2, coef_c, rho_0, g):

    dx1_dt =    coef_c[2]*(x[3]*y[0] - 0.5*x[0]*y[3] + 2*x[4]*y[1] - x[1]*y[4] + 2*x[5]*y[2] - x[2]*y[5]) - coef_1[4]*x[0]
    dx2_dt =  2*coef_1[0]*x[2] + 2*coef_c[0]*y[4] \
              - coef_1[1]*x[0]*x[2] \
              - coef_2[1]*x[3]*x[5] \
              - coef_c[2]*(x[0]*y[4] - 2*x[3]*y[1] - 2*x[4]*y[0] + y[3]*x[1]) \
              - 2*rho_0*g*z[2] - coef_c[4]*y[1] - coef_c[3]*y[4] - coef_1[4]*x[1]
    dx3_dt = -2*coef_1[0]*x[1] - 2*coef_c[0]*y[5] \
              + coef_1[1]*x[0]*x[1] \
              + coef_2[1]*x[3]*x[4] \
              + coef_c[2]*(x[0]*y[5] - 2*x[3]*y[2] - 2*x[5]*y[0] + y[3]*x[2]) \
              + 2*rho_0*g*z[1] - coef_c[4]*y[2] - coef_c[3]*y[5] - coef_1[4]*x[2]
    dx4_dt =  - coef_c[2]*(0.5*x[0]*y[0] + x[1]*y[1] + x[2]*y[2]) - coef_2[4]*x[3]
    dx5_dt =  2*coef_2[0]*x[5] + 2*coef_c[0]*y[1] \
              - coef_2[1]*(x[3]*x[2] + x[0]*x[5]) \
              - coef_c[2]*(x[0]*y[1] + x[1]*y[0]) \
              - 2*rho_0*g*z[5] - coef_c[4]*y[4] - coef_c[3]*y[1] - coef_2[4]*x[4]
    dx6_dt = -2*coef_2[0]*x[4] - 2*coef_c[0]*y[2] \
              + coef_2[1]*(x[3]*x[1] + x[0]*x[4]) \
              + coef_c[2]*(x[0]*y[2] + x[2]*y[0]) \
              + 2*rho_0*g*z[4] - coef_c[4]*y[5] - coef_c[3]*y[2] - coef_2[4]*x[5]

    return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt])
    
def ode_y(T, x, y, z, coef_1, coef_2, coef_c, rho_0, g):
    
    dy1_dt =  2*coef_1[1]*(x[2]*y[1] - x[1]*y[2]) \
            + 2*coef_2[1]*(x[5]*y[4] - x[4]*y[5]) \
            + 0.5*coef_c[2]*y[0]*y[3] + 2*coef_1[2]*rho_0*g*z[3] - coef_1[4]*y[0]
    dy2_dt =  2*coef_1[0]*y[2] \
              - coef_1[1]*x[0]*y[2] \
              - coef_2[1]*x[3]*y[5] \
              + coef_c[2]*(y[0]*y[4] + y[3]*y[1]) \
              - coef_2[2]*rho_0*g*z[4] + coef_c[4]*x[1] + coef_c[3]*x[4] - coef_1[4]*y[1]
    dy3_dt = -2*coef_1[0]*y[1] \
              + coef_1[1]*x[0]*y[1] \
              + coef_2[1]*x[3]*y[4] \
              - coef_c[2]*(y[0]*y[5] + y[3]*y[2]) \
              + coef_2[2]*rho_0*g*z[5] + coef_c[4]*x[2] + coef_c[3]*x[5] - coef_1[4]*y[2]
    dy4_dt =  2*coef_2[1]*(x[5]*y[1] - x[1]*y[5] + x[2]*y[4] - x[4]*y[2]) \
            - 0.5*(coef_c[2]*y[0]**2 + 2*(y[1]**2 + y[2]**2)) \
            - 2*coef_1[2]*rho_0*g*z[0] - coef_2[4]*y[3]
    dy5_dt =  2*coef_2[0]*y[5] \
              - coef_2[1]*(x[3]*y[2] + x[0]*y[5]) \
            - 2*coef_c[2]*y[0]*y[1] \
              - coef_1[2]*rho_0*g*z[5] + coef_c[4]*x[4] + coef_c[3]*x[1] - coef_2[4]*y[4]
    dy6_dt = -2*coef_2[0]*y[4] \
              + coef_2[1]*(x[3]*y[1] + x[0]*y[4]) \
            + 2*coef_c[2]*y[0]*y[2] \
              + coef_1[2]*rho_0*g*z[4] + coef_c[4]*x[5] + coef_c[3]*x[2] - coef_2[4]*y[5]

    return np.array([dy1_dt, dy2_dt, dy3_dt, dy4_dt, dy5_dt, dy6_dt])
    
def ode_z(T, x, y, z, coef_1, coef_2, coef_c, rho_0, g):

    dz1_dt =  coef_c[2]*(z[4]*y[1] + z[1]*y[4] + z[5]*y[1] + z[1]*y[5] - 0.5*z[0]*y[4] + z[3]*y[0] ) \
      - 0.5 * coef_1[3]*y[0] - 0.5 * coef_c[1]*y[3]  + 2 * coef_1[2] * y[3] \
            - coef_1[4]*z[0] # - dyv 


    dz2_dt =  2*coef_1[0]*z[2]                              \
              - coef_1[1]*(x[0]*z[2] + z[0]*x[2])           \
              - coef_2[1]*(x[3]*z[5] + z[3]*x[5])           \
              + coef_c[2]*(y[0]*z[4] + z[0]*y[4] + y[3]*z[1] + z[3]*y[1]) \
              - coef_1[3]*y[1] - coef_c[1]*y[4]     - 2 * x[2] - coef_2[2] * y[4] \
              - coef_1[4]*z[1] # -dxu - dyv


    dz3_dt = -2*coef_1[0]*z[1]                              \
              + coef_1[1]*(x[0]*z[1] + z[0]*x[1])           \
              + coef_2[1]*(x[3]*z[4] + z[3]*x[4])           \
              - coef_c[2]*(y[0]*z[5] + z[0]*y[5] + y[3]*z[2] + z[3]*y[2]) \
              + coef_1[3]*y[2] + coef_c[1]*y[5]    + 2 * x[1] + coef_2[2] * y[5] \
              - coef_1[4]*z[2] # -dxu - dyv 


    dz4_dt =  - coef_c[2]*(z[0]*y[0] + z[1]*y[1] + z[2]*y[2] ) \
              - 0.5 * coef_2[3]*y[3] - 0.5 * coef_c[0]*y[0]  - 2 * coef_1[2] * y[0] \
              - coef_2[4]*z[3] # -dyv  


    dz5_dt =  2*coef_2[0]*z[5]                              \
              - coef_2[1]*(x[3]*z[2] + z[3]*x[2] + x[0]*z[5] + z[0]*x[5]) \
            - 2*coef_c[2]*(y[0]*z[1] + z[0]*y[1])           \
              - coef_2[3]*y[4] - coef_c[1]*y[1]  - 2 * x[5] - coef_1[2]* y[5] \
              - coef_2[4]*z[4]  # -dxu - dyv  


    dz6_dt = -2*coef_2[0]*z[4]                              \
              + coef_2[1]*(x[3]*z[1] + z[3]*x[1] + x[0]*z[4] + z[0]*x[4]) \
            + 2*coef_c[2]*(y[0]*z[2] + z[0]*y[2])           \
              + coef_2[3]*y[5] - coef_c[1]*y[2]  + 2 * x[4] + coef_1[2]* y[4] \
              - coef_2[4]*z[5] # -dxu - dyv 

    return np.array([dz1_dt, dz2_dt, dz3_dt, dz4_dt, dz5_dt, dz6_dt])

# call the appropriate functions and concatenate the output
def ode(T, initial, rho_0, g, f_0, beta, b, s_2, s_4, c):
    x = initial[0:6]
    y = initial[6:12]
    z = initial[12:18]
    
    coef_1 = coef_n(1, b, s_2, s_4, beta, f_0, g, rho_0, c)
    coef_2 = coef_n(2, b, s_2, s_4, beta, f_0, g, rho_0, c)
    coef_c = coef(b, s_2, s_4, beta, f_0, g, rho_0)

    dx_dt = ode_x(T, x, y, z, coef_1, coef_2, coef_c, rho_0, g)
    dy_dt = ode_y(T, x, y, z, coef_1, coef_2, coef_c, rho_0, g)
    dz_dt = ode_z(T, x, y, z, coef_1, coef_2, coef_c, rho_0, g)
    
    result = np.concatenate([dx_dt, dy_dt, dz_dt])
    return result
