#OPTIMISATION WITH MMA

import numpy as np
import math
import MMA_functionsSS as MMAf

"""
Script for MMA optimisation of RHCSPs

Version June 2022

Luis Santos <pinhosl3@lsbu.ac.uk>
School of Buil Environment and Architecture
London South Bank University 
London, UK

*** Input:
    Design internal forces 
*** Output:
    Optimal value of Objective Function
    Value of panel variables
    Value of constraints functions
"""

# Input Design Internal Forces

Med= 407000 # von-Mises Moment in N.mm/mm
MedIC = ((390000, 286000),(337000, 321000)) # Moments for Intercelular buckling
Vedx= 456 # Design shear force in X in N/mm
Vedy= 450 # Design shear force in Y in N/mm
Ced= 2.02 # Design compression force in N/mm^2
Ded= 9.2e11/0.91 # Maximum bending deflection N.mm^2
Sed= 5.2e5 # Maximum shear deflection N
L = 5400 #in mm
W = 9000 #in mm
fy = 235 #in N/mm^2
modulus = 210000 #in N/mm^2
E = 210000
v = 0.3
shear_mod = modulus/(2*(1+v))
density = 7850

#print(MedIC)
#print("Vedx - %s" % Vedx)
#print("Vedy - %s" % Vedy)
#print("Ced - %s" % Ced)
#print("Ded - %s" % Ded)
#print("Sed - %s" % Sed)
#print("Med - %s" % Med)

ytop    = Med / fy # Constant for top plate yielding
ybot    = Med / fy # Constant for bot plate yielding
yshearx = Vedx * math.sqrt(3) / fy # Constant for shear yielding in X
ysheary = Vedy * math.sqrt(3) / fy # Constant for shear yielding in Y
ycomp   = Ced / fy # Constant for compressive yielding 
sbuckx  = Vedx * 12 * (1 - v**2) / (modulus * math.pi**2)
# Constant for shear buckling in X
sbucky  = Vedy * 12 * (1 - v**2) / (modulus * math.pi**2)
# Constant for shear buckling in Y
cbuck   = Ced  * 12 * (1 - v**2) / (modulus * math.pi**2)
# Constant for shear buckling

Medx_top1 = MedIC[0][0]
Medy_top1 = MedIC[0][1]
icbuck1 = Medx_top1 * 12 * (1 - v**2) / (modulus * math.pi**2)
# Constant for intercellular buckling
Medx_top2 = MedIC[1][0]
Medy_top2 = MedIC[1][1]
icbuck2 = Medx_top2 * 12 * (1 - v**2) / (modulus * math.pi**2)
# Constant for intercellular buckling

Kbending = Ded*0.91/modulus/(min(L,W)/360)
# Constant for bending deflection 
Kshear = Sed/shear_mod/(min(L,W)/360)
# Constant for shear deflection

#Boundaries 
bnds_x0 = ((0.1,8.0),(0.1,8.0),(25.0,500.0),(0.1,8.0),(25.0,500.0),(0.1,8.0),(25.0,500.0))
#bnds_x0 = ((3.0,15.0),(3.0,15.0),(25.0,700.0),(3.0,15.0),(25.0,500.0),(3.0,15.0),(25.0,500.0))

x0 = np.zeros(len(bnds_x0))
for i in range(len(x0)):
    x0[i] = np.random.uniform(1.1*bnds_x0[i][0], 0.9*bnds_x0[i][1])
    
    
#x0 = [0.10,0.10,25.0,0.10,25.0,0.10,25.0]
#x0 = [15.0,15.0,500.0,15.0,500.0,15.0,500.0]
#x0 = [8.0384985,9.37319026,454.55215247,3.88889136,137.17592847,1.97025152,226.25034263]
#x0 = [   5.78699861    ,1.24446403  ,195.77397615    ,6.77115691   ,41.83483535,1.45959861  ,193.99563098]
#x0 = [   4.32721102    ,4.32721104  ,400.23869154    ,1.76344881  ,163.39648663,    1.78203253  ,163.39735555]
#print(x0)

bounds = bnds_x0

Weight_max = density*(bnds_x0[0][1]+bnds_x0[1][1]+bnds_x0[2][1]*(bnds_x0[3][1]/bnds_x0[4][1]+bnds_x0[5][1]/bnds_x0[6][1]))

#Objective function
Weight = lambda x: density*(x[0]+x[1]+x[2]*(x[3]/x[4]+x[5]/x[6]))/Weight_max
    
#Objective function sensitivity
Diff_Weight = lambda x: np.array([density/Weight_max, density/Weight_max, (x[3]/x[4]+x[5]/x[6])*density/Weight_max, x[2]/x[4]*density/Weight_max, -x[2]*x[3]/x[4]**2*density/Weight_max, x[2]/x[6]*density/Weight_max, -x[2]*x[5]/x[6]**2*density/Weight_max])
  
#Top plate yielding with sensitivity  
con1 = {'type': 'ineq', 'fun': lambda x: np.array(ytop/(x[2]*x[0])-1), 
            'jac' : lambda x: np.array([-ytop/(x[2]*x[0]**2), 0, -ytop/(x[0]*x[2]**2),0,0,0,0])}

#Bot plate yielding with sensitivity  
con2 = {'type': 'ineq', 'fun': lambda x: np.array(ybot/(x[2]*x[1])-1), 
        'jac' : lambda x: np.array([0, -ybot/(x[2]*x[1]**2), -ybot/(x[1]*x[2]**2),0,0,0,0])}

#Core Shear yielding in X with sensitivity  
con3 = {'type': 'ineq', 'fun': lambda x: np.array(yshearx/(x[2]*x[3]/x[4])-1),  
        'jac' : lambda x: np.array([0,0, -x[4]*yshearx/(x[2]**2*x[3]), -x[4]*yshearx/(x[3]**2*x[2]),yshearx/(x[2]*x[3]),0,0])}

#Core Shear yielding in Y with sensitivity 
con4 = {'type': 'ineq', 'fun': lambda x: np.array(ysheary/(x[2]*x[5]/x[6])-1),  
        'jac' : lambda x: np.array([0,0, -x[6]*ysheary/(x[2]**2*x[5]),0,0, -x[6]*ysheary/(x[5]**2*x[2]), ysheary/(x[2]*x[5])])}

#Compressive yielding with sensitivity 
con5 = {'type': 'ineq', 'fun': lambda x: np.array(ycomp/(x[3]/x[4]+x[5]/x[6])-1),  
        'jac' : lambda x: np.array([0,0,0,-ycomp/(x[4]*(x[3]/x[4]+x[5]/x[6])**2), ycomp*x[3]/(x[4]**2*(x[3]/x[4]+x[5]/x[6])**2), -ycomp/(x[6]*(x[3]/x[4]+x[5]/x[6])**2), ycomp*x[5]/(x[6]**2*(x[3]/x[4]+x[5]/x[6])**2)])}


"INTERCELLULAR BUCKLING OF THE TOP PLATE - With sensitivity"
m_list = [1,2,3]
n_list = [1,2,3]
consIC_top = []
ratio1 = Medy_top1/Medx_top1
for m in m_list:
    conIC = {'type': 'ineq', 'fun': lambda x, m=m, icbuck1=icbuck1, ratio1=ratio1: np.array(icbuck1*(x[6]**2*x[4]**2)*(m**2*x[4]**2+ratio1*x[6]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2)-1),  
            'jac' : lambda x, m=m, icbuck1=icbuck1, ratio1=ratio1: np.array([-3*icbuck1*x[6]**2*x[4]**2*(ratio1*x[6]**2+m**2*x[4]**2)/(x[0]**4*x[2]*(m**2*x[4]**2+x[6]**2)**2), 0, 
                                        -icbuck1*x[6]**2*x[4]**2*(ratio1*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]**2*(m**2*x[4]**2+x[6]**2)**2), 0, 
                                        -4*icbuck1*x[6]**2*x[4]**3*m**2*(ratio1*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**3) + 2*icbuck1*x[4]*x[6]**2*(ratio1*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2) + 2*icbuck1*x[4]**3*x[6]**2*m**2/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2), 0, 
                                        -4*icbuck1*x[4]**2*x[6]**3*(ratio1*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**3) + 2*icbuck1*x[6]*x[4]**2*(ratio1*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2) + 2*icbuck1*x[6]**3*x[4]**2*ratio1/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2)])}
    consIC_top.append(conIC)

for n in n_list:
    conIC = {'type': 'ineq', 'fun': lambda x, n=n, icbuck1=icbuck1, ratio1=ratio1: np.array(icbuck1*(x[6]**2*x[4]**2)*(ratio1*n**2*x[6]**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2)-1),  
            'jac' : lambda x, n=n, icbuck1=icbuck1, ratio1=ratio1: np.array([-3*icbuck1*x[6]**2*x[4]**2*(ratio1*x[6]**2*n**2+x[4]**2)/(x[0]**4*x[2]*(n**2*x[6]**2+x[4]**2)**2), 0, 
                                        -icbuck1*x[6]**2*x[4]**2*(ratio1*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]**2*(n**2*x[6]**2+x[4]**2)**2), 0, 
                                        -4*icbuck1*x[6]**2*x[4]**3*(ratio1*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**3) + 2*icbuck1*x[4]*x[6]**2*(ratio1*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2) + 2*icbuck1*x[4]**3*x[6]**2/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2), 0, 
                                        -4*icbuck1*x[4]**2*x[6]**3*n**2*(ratio1*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**3) + 2*icbuck1*x[6]*x[4]**2*(ratio1*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2) + 2*icbuck1*x[6]**3*x[4]**2*ratio1*n**2/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2)])}
    consIC_top.append(conIC)    

ratio2 = Medy_top2/Medx_top2
for m in m_list:
    conIC = {'type': 'ineq', 'fun': lambda x, m=m, icbuck2=icbuck2, ratio2=ratio2: np.array(icbuck2*(x[6]**2*x[4]**2)*(m**2*x[4]**2+ratio2*x[6]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2)-1),  
            'jac' : lambda x, m=m, icbuck2=icbuck2, ratio2=ratio2: np.array([-3*icbuck2*x[6]**2*x[4]**2*(ratio2*x[6]**2+m**2*x[4]**2)/(x[0]**4*x[2]*(m**2*x[4]**2+x[6]**2)**2), 0, 
                                        -icbuck2*x[6]**2*x[4]**2*(ratio2*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]**2*(m**2*x[4]**2+x[6]**2)**2), 0, 
                                        -4*icbuck2*x[6]**2*x[4]**3*m**2*(ratio2*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**3) + 2*icbuck2*x[4]*x[6]**2*(ratio2*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2) + 2*icbuck2*x[4]**3*x[6]**2*m**2/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2), 0, 
                                        -4*icbuck2*x[4]**2*x[6]**3*(ratio2*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**3) + 2*icbuck2*x[6]*x[4]**2*(ratio2*x[6]**2+m**2*x[4]**2)/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2) + 2*icbuck2*x[6]**3*x[4]**2*ratio2/(x[0]**3*x[2]*(m**2*x[4]**2+x[6]**2)**2)])}
    consIC_top.append(conIC)

for n in n_list:
    conIC = {'type': 'ineq', 'fun': lambda x, n=n, icbuck2=icbuck2, ratio2=ratio2: np.array(icbuck2*(x[6]**2*x[4]**2)*(ratio2*n**2*x[6]**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2)-1),  
            'jac' : lambda x, n=n, icbuck2=icbuck2, ratio2=ratio2: np.array([-3*icbuck2*x[6]**2*x[4]**2*(ratio2*x[6]**2*n**2+x[4]**2)/(x[0]**4*x[2]*(n**2*x[6]**2+x[4]**2)**2), 0, 
                                        -icbuck2*x[6]**2*x[4]**2*(ratio2*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]**2*(n**2*x[6]**2+x[4]**2)**2), 0, 
                                        -4*icbuck2*x[6]**2*x[4]**3*(ratio2*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**3) + 2*icbuck2*x[4]*x[6]**2*(ratio2*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2) + 2*icbuck2*x[4]**3*x[6]**2/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2), 0, 
                                        -4*icbuck2*x[4]**2*x[6]**3*n**2*(ratio2*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**3) + 2*icbuck2*x[6]*x[4]**2*(ratio2*x[6]**2*n**2+x[4]**2)/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2) + 2*icbuck2*x[6]**3*x[4]**2*ratio2*n**2/(x[0]**3*x[2]*(n**2*x[6]**2+x[4]**2)**2)])}
    consIC_top.append(conIC)

    
"SHEAR BUCKLING - With sensitivity"                                 
"This one assumes h > ly or x[2] > x[6]"
con71 = {'type': 'ineq', 'fun': lambda x: np.array(sbuckx/(x[3]**3/(x[6]**2))/(5.35+4/(x[2]/x[6])**2)/(x[2]/x[4])-1),  
        'jac' : lambda x: np.array([0,0,8*sbuckx*x[6]**4*x[4]/(x[3]**3*x[2]**4*(5.35+4*x[6]**2/x[2]**2)**2)-sbuckx*x[4]*x[6]**2/(x[2]**2*x[3]**3*(5.35+4*x[6]**2/x[2]**2)),
                                    -3*x[6]**2*sbuckx*x[4]/(x[3]**4*x[2]*(5.35+4*x[6]**2/x[2]**2)),
                                    sbuckx*x[6]**2/(x[3]**3*x[2]*(5.35+4*x[6]**2/x[2]**2)),0,
                                    2*sbuckx*x[6]*x[4]/(x[3]**3*x[2]*(5.35+4*x[6]**2/x[2]**2))-8*sbuckx*x[6]**3*x[4]/(x[3]**3*x[2]**3*(5.35+4*x[6]**2/x[2]**2)**2)])}
    
"This one assumes h < ly or x[2] < x[6]"
con72 = {'type': 'ineq', 'fun': lambda x: np.array(sbuckx/(x[3]**3/(x[2]**2))/(5.35+4/(x[6]/x[2])**2)/(x[2]/x[4])-1),  
        'jac' : lambda x: np.array([0,0, sbuckx*x[4]/(x[3]**3*(5.35+4*x[2]**2/x[6]**2)) - 8*sbuckx*x[2]**2*x[4]/(x[3]**3*x[6]**2*(5.35+4*x[2]**2/x[6]**2)**2),
                                    -3*x[2]*x[4]*sbuckx/(x[3]**4*(5.35+4*x[2]**2/x[6]**2)),
                                    sbuckx*x[2]/(x[3]**3*(5.35+4*x[2]**2/x[6]**2)), 0,
                                    8*sbuckx*x[2]**3*x[4]/(x[3]**3*x[6]**3*(5.35+4*x[2]**2/x[6]**2)**2)])}  

"This one assumes h > lx or x[2] > x[4]"   
con81 = {'type': 'ineq', 'fun': lambda x: np.array(sbucky/(x[5]**3/(x[4]**2))/(5.35+4/(x[2]/x[4])**2)/(x[2]/x[6])-1),  
        'jac' : lambda x: np.array([0,0, 8*sbucky*x[4]**4*x[6]/(x[5]**3*(5.35+4*x[4]**2/x[2]**2)**2*x[2]**4)-sbucky*x[6]*x[4]**2/(x[2]**2*x[5]**3*(5.35+4*x[4]**2/x[2]**2)),
                                    0,2*sbucky*x[4]*x[6]/(x[5]**3*x[2]*(5.35+4*x[4]**2/x[2]**2))-8*sbucky*x[4]**3*x[6]/(x[5]**3*x[2]**3*(5.35+4*x[4]**2/x[2]**2)**2),
                                    -3*x[4]**2*sbucky*x[6]/(x[5]**4*x[2]*(5.35+4*x[4]**2/x[2]**2)),
                                    sbucky*x[4]**2/(x[5]**3*x[2]*(5.35+4*x[4]**2/x[2]**2))])}
    
"This one assumes h < lx or x[2] < x[4]"
con82 = {'type': 'ineq', 'fun': lambda x: np.array(sbucky/(x[5]**3/(x[2]**2))/(5.35+4/(x[4]/x[2])**2)/(x[2]/x[6])-1),   
        'jac' : lambda x: np.array([0,0, sbucky*x[6]/(x[5]**3*(5.35+4*x[2]**2/x[4]**2)) - 8*sbucky*x[2]**2*x[6]/(x[5]**3*x[4]**2*(5.35+4*x[2]**2/x[4]**2)**2),
                                    0,8*sbucky*x[2]**3*x[6]/(x[5]**3*x[4]**3*(5.35+4*x[2]**2/x[4]**2)**2),
                                    -3*x[2]*x[6]*sbucky/(x[5]**4*(5.35+4*x[2]**2/x[4]**2)),
                                    sbucky*x[2]/(x[5]**3*(5.35+4*x[2]**2/x[4]**2))])}
    
    
"COMPRESSIVE BUCKLING - With sensitivity"
m_limit = 8
m1_list = [i for i in range(1,m_limit+1)]
m2_list = [i for i in range(1,m_limit+1)]
consComp_buck = []
cbuck = Ced  * 12 * (1 - v**2) / (modulus * math.pi**2)
for m1 in m1_list:
    for m2 in m2_list:
        conCB = {'type': 'ineq', 'fun': lambda x, m1=m1, m2=m2, cbuck=cbuck: np.array(cbuck/((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6]))-1),  
                'jac' : lambda x, m1=m1, m2=m2, cbuck=cbuck: np.array([0,0,-cbuck*(2*(m1*x[6]/x[2]+x[2]/(m1*x[6]))*x[3]**3*(-m1*x[6]/x[2]**2+1/(m1*x[6]))/(x[6]**2*x[4]) + 2*(m2*x[4]/x[2]+x[2]/(m2*x[4]))*x[5]**3*(-m2*x[4]/x[2]**2+1/(m2*x[4]))/(x[4]**2*x[6])) / (((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6]))**2), 
                                            -3*cbuck*(m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**2 / ((((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6]))**2)*x[6]**2*x[4]),
                                            -cbuck*(-(m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]**2) + 2*(m2*x[4]/x[2]+x[2]/(m2*x[4]))*x[5]**3*(m2/x[2]-x[2]/(m2*x[4]**2))/(x[4]**2*x[6]) - 2*(m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**3*x[6])) / (((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6]))**2),
                                            -3*cbuck*(m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**2 / ((((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6]))**2)*x[4]**2*x[6]),
                                            -cbuck*(-(m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[6]**2*x[4]**2) + 2*(m1*x[6]/x[2]+x[2]/(m1*x[6]))*x[3]**3*(m1/x[2]-x[2]/(m1*x[6]**2))/(x[6]**2*x[4]) - 2*(m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**3*x[4])) / (((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6]))**2)])}
        consComp_buck.append(conCB)

#con11 = {'type': 'ineq', 'fun': lambda x: np.array(x[3]**3/(x[4]*x[2])/cbuck-1),  
#         'jac' : lambda x: np.array([0,0,-x[3]**3/(x[4]*x[2]**2)/cbuck, 3*x[3]**2/(x[4]*x[2])/cbuck,-x[3]**3/(x[4]**2*x[2])/cbuck])}


#con11 = {'type': 'ineq', 'fun': lambda x: np.array(x[3]**3/(x[4]*x[2])/cbuck-1),  
#         'jac' : lambda x: np.array([0,0,-x[3]**3/(x[4]*x[2]**2)/cbuck, 3*x[3]**2/(x[4]*x[2])/cbuck,-x[3]**3/(x[4]**2*x[2])/cbuck])}

"DEFLECTION Constraints - With sensitivity"
#con12_bending = {'type': 'ineq', 'fun': lambda x: np.array(Kbending/(x[0]*x[1]*x[2]**2/(x[0]+x[1]))-1),  
#        'jac' : lambda x: np.array([-Kbending*(x[0]+x[1])/(x[2]**2*x[0]**2*x[1]) + Kbending/(x[2]**2*x[1]*x[0]), -Kbending*(x[0]+x[1])/(x[2]**2*x[1]**2*x[0]) + Kbending/(x[2]**2*x[1]*x[0]), -2*Kbending*(x[0]+x[1])/(x[2]**3*x[1]*x[0]),0,0,0,0])}

con12 = {'type': 'ineq', 'fun': lambda x: np.array(Kbending/(x[0]*x[1]*x[2]**2/(x[0]+x[1]))+Kshear/(x[3]*x[2]/x[4])-1),  
        'jac' : lambda x: np.array([-Kbending*(x[0]+x[1])/(x[2]**2*x[0]**2*x[1]) + Kbending/(x[2]**2*x[1]*x[0]), -Kbending*(x[0]+x[1])/(x[2]**2*x[1]**2*x[0]) + Kbending/(x[2]**2*x[1]*x[0]), -2*Kbending*(x[0]+x[1])/(x[2]**3*x[1]*x[0])-Kshear*x[4]/(x[3]*x[2]**2),-Kshear*x[4]/(x[3]**2*x[2]),Kshear/(x[3]*x[2]),0,0])}

con13 = {'type': 'ineq', 'fun': lambda x: np.array(Kbending/(x[0]*x[1]*x[2]**2/(x[0]+x[1]))+Kshear/(x[5]*x[2]/x[6])-1),  
        'jac' : lambda x: np.array([-Kbending*(x[0]+x[1])/(x[2]**2*x[0]**2*x[1]) + Kbending/(x[2]**2*x[1]*x[0]), -Kbending*(x[0]+x[1])/(x[2]**2*x[1]**2*x[0]) + Kbending/(x[2]**2*x[1]*x[0]), -2*Kbending*(x[0]+x[1])/(x[2]**3*x[1]*x[0])-Kshear*x[6]/(x[5]*x[2]**2),0,0,-Kshear*x[6]/(x[5]**2*x[2]),Kshear/(x[5]*x[2])])}

#cons = [con1,con2,con3,con4,con5,con61,con62,con71,con72,con81,con82,con12,con13]
cons = [con1,con2,con3,con4,con5] + consIC_top + [con71,con72,con81,con82] + consComp_buck + [con12,con13]

#Solving
sol, outit, kktnorm = MMAf.MMA_init(Weight, x0, Diff_Weight, bounds, cons)

#Post processing
sol['f0val'] = sol['f0val']*Weight_max;
x = sol['xval']

tt,tb,h,twx,lx,twy,ly = x

ICSafFac_top1 = np.array([icbuck1/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((1**2*x[4]**2+x[6]**2)**2/(1**2*x[4]**2+ratio1*x[6]**2)),
                          icbuck1/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((2**2*x[4]**2+x[6]**2)**2/(2**2*x[4]**2+ratio1*x[6]**2)),
                          icbuck1/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((3**2*x[4]**2+x[6]**2)**2/(3**2*x[4]**2+ratio1*x[6]**2)),
                          icbuck1/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((1**2*x[6]**2+x[4]**2)**2/(ratio1*1**2*x[6]**2+x[4]**2)),
                          icbuck1/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((2**2*x[6]**2+x[4]**2)**2/(ratio1*2**2*x[6]**2+x[4]**2)),
                          icbuck1/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((3**2*x[6]**2+x[4]**2)**2/(ratio1*3**2*x[6]**2+x[4]**2)),
                          icbuck2/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((1**2*x[4]**2+x[6]**2)**2/(1**2*x[4]**2+ratio2*x[6]**2)),
                          icbuck2/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((2**2*x[4]**2+x[6]**2)**2/(2**2*x[4]**2+ratio2*x[6]**2)),
                          icbuck2/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((3**2*x[4]**2+x[6]**2)**2/(3**2*x[4]**2+ratio2*x[6]**2)),
                          icbuck2/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((1**2*x[6]**2+x[4]**2)**2/(ratio2*1**2*x[6]**2+x[4]**2)),
                          icbuck2/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((2**2*x[6]**2+x[4]**2)**2/(ratio2*2**2*x[6]**2+x[4]**2)),
                          icbuck2/(x[0]**3*x[2]/(x[6]**2*x[4]**2))/((3**2*x[6]**2+x[4]**2)**2/(ratio2*3**2*x[6]**2+x[4]**2))])**-1

"COMPRESSIVE BUCKLING Safety Factors"
CBSafFac = np.zeros(len(m1_list)*len(m2_list))
for m1 in m1_list:
    for m2 in m2_list:
        CBSafFac[m2-1+len(m1_list)*(m1-1)] = (cbuck/((m1*x[6]/x[2]+x[2]/(m1*x[6]))**2*x[3]**3/(x[6]**2*x[4]) + (m2*x[4]/x[2]+x[2]/(m2*x[4]))**2*x[5]**3/(x[4]**2*x[6])))**-1

#  Safety Factors
sol['SafFac'] = np.array([ytop/(x[2]*x[0]), 
                         ybot/(x[2]*x[1]),
                         yshearx/(x[2]*x[3]/x[4]), 
                         ysheary/(x[2]*x[5]/x[6]),
                         ycomp/(x[3]/x[4]+x[5]/x[6]),
                         min(ICSafFac_top1[0:6])**-1,    
                         min(ICSafFac_top1[6:12])**-1,
                         sbuckx/(x[3]**3/(x[6]**2))/(5.35+4/(x[2]/x[6])**2)/(x[2]/x[4]),
                         sbuckx/(x[3]**3/(x[2]**2))/(5.35+4/(x[6]/x[2])**2)/(x[2]/x[4]),
                         sbucky/(x[5]**3/(x[4]**2))/(5.35+4/(x[2]/x[4])**2)/(x[2]/x[6]),
                         sbucky/(x[5]**3/(x[2]**2))/(5.35+4/(x[4]/x[2])**2)/(x[2]/x[6]),
                         min(CBSafFac)**-1,
                         Kbending/(x[0]*x[1]*x[2]**2/(x[0]+x[1]))+Kshear/(x[3]*x[2]/x[4]),
                         Kbending/(x[0]*x[1]*x[2]**2/(x[0]+x[1]))+Kshear/(x[5]*x[2]/x[6])])**-1

#  Safety Factors calculated in standard form
sol['RealSafFac'] = np.array([Med/MMAf.YIELD_TOP(h,tt,fy), 
                   Med/MMAf.YIELD_BOT(h, tb, fy), 
                   Vedx/MMAf.YIELD_SHEAR_X(h, twx, lx, fy), 
                   Vedy/MMAf.YIELD_SHEAR_Y(h, twy, ly, fy),
                   Ced/MMAf.YIELD_COMPRESSION(twx, lx, twy, ly, fy),
                   Medx_top1/MMAf.BUCK_INTERCELL_v2(Medx_top1, Medy_top1, tt, lx, ly, h, E),
                   Medx_top2/MMAf.BUCK_INTERCELL_v2(Medx_top2, Medy_top2, tt, lx, ly, h, E),
                   Vedx/MMAf.BUCK_SHEAR_X(twx, lx, ly, h, E), 
                   Vedy/MMAf.BUCK_SHEAR_Y(twy, lx, ly, h, E),
                   Ced/MMAf.BUCK_COMPRESSION(twx,lx,twy,ly,h,E,m_limit),
                   (Ded/MMAf.D_DEFLECTION(h, tt, tb, E) + Sed/MMAf.S_DEFLECTION(h, twx, lx, twy, ly, shear_mod))/(min(L,W)/360)])**-1-1
    
print(sol['f0val'])
print(sol['xval'])   



print('--------------')

#Rerun the optimisation to accomodate splitwise functions
if h >= lx and h >= ly:
    print("h >= lx and h >= ly")
    cons = [con1,con2,con3,con4,con5] + consIC_top + [con71,con81] + consComp_buck + [con12,con13]
elif h < lx and h < ly:
    print("h < lx and h < ly")
    cons = [con1,con2,con3,con4,con5] + consIC_top + [con72,con82] + consComp_buck + [con12,con13]
elif h >= lx and h < ly:
    print("h >= lx and h < ly")
    cons = [con1,con2,con3,con4,con5] + consIC_top + [con72,con81] + consComp_buck + [con12,con13]
elif h < lx and h >= ly:
    print("h < lx and h >= ly")
    cons = [con1,con2,con3,con4,con5] + consIC_top + [con71,con82] + consComp_buck + [con12,con13]


sol2, outit, kktnorm = MMAf.MMA_init(Weight, x0, Diff_Weight, bounds, cons)

sol2['f0val'] = sol2['f0val']*Weight_max;
x2 = sol2['xval']


tt,tb,h,twx,lx,twy,ly = x2

ICSafFac_top1 = np.array([icbuck1/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((1**2*x2[4]**2+x2[6]**2)**2/(1**2*x2[4]**2+ratio1*x2[6]**2)),
            icbuck1/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((2**2*x2[4]**2+x2[6]**2)**2/(2**2*x2[4]**2+ratio1*x2[6]**2)),
            icbuck1/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((3**2*x2[4]**2+x2[6]**2)**2/(3**2*x2[4]**2+ratio1*x2[6]**2)),
            icbuck1/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((1**2*x2[6]**2+x2[4]**2)**2/(ratio1*1**2*x2[6]**2+x2[4]**2)),
            icbuck1/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((2**2*x2[6]**2+x2[4]**2)**2/(ratio1*2**2*x2[6]**2+x2[4]**2)),
            icbuck1/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((3**2*x2[6]**2+x2[4]**2)**2/(ratio1*3**2*x2[6]**2+x2[4]**2)),
            icbuck2/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((1**2*x2[4]**2+x2[6]**2)**2/(1**2*x2[4]**2+ratio2*x2[6]**2)),
            icbuck2/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((2**2*x2[4]**2+x2[6]**2)**2/(2**2*x2[4]**2+ratio2*x2[6]**2)),
            icbuck2/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((3**2*x2[4]**2+x2[6]**2)**2/(3**2*x2[4]**2+ratio2*x2[6]**2)),
            icbuck2/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((1**2*x2[6]**2+x2[4]**2)**2/(ratio2*1**2*x2[6]**2+x2[4]**2)),
            icbuck2/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((2**2*x2[6]**2+x2[4]**2)**2/(ratio2*2**2*x2[6]**2+x2[4]**2)),
            icbuck2/(x2[0]**3*x2[2]/(x2[6]**2*x2[4]**2))/((3**2*x2[6]**2+x2[4]**2)**2/(ratio2*3**2*x2[6]**2+x2[4]**2))])**-1
  
"COMPRESSIVE BUCKLING"
CBSafFac = np.zeros(len(m1_list)*len(m2_list))
for m1 in m1_list:
    for m2 in m2_list:
        CBSafFac[m2-1+len(m1_list)*(m1-1)] = (cbuck/((m1*x2[6]/x2[2]+x2[2]/(m1*x2[6]))**2*x2[3]**3/(x2[6]**2*x2[4]) + (m2*x2[4]/x2[2]+x2[2]/(m2*x2[4]))**2*x2[5]**3/(x2[4]**2*x2[6])))**-1
    
   
sol2['SafFac'] = np.array([ytop/(x2[2]*x2[0]), 
                         ybot/(x2[2]*x2[1]),
                         yshearx/(x2[2]*x2[3]/x2[4]), 
                         ysheary/(x2[2]*x2[5]/x2[6]),
                         ycomp/(x2[3]/x2[4]+x2[5]/x2[6]),
                         min(ICSafFac_top1[0:6])**-1,
                         min(ICSafFac_top1[6:12])**-1,
                         sbuckx/(x2[3]**3/(x2[6]**2))/(5.35+4/(x2[2]/x2[6])**2)/(x2[2]/x2[4]),
                         sbuckx/(x2[3]**3/(x2[2]**2))/(5.35+4/(x2[6]/x2[2])**2)/(x2[2]/x2[4]),
                         sbucky/(x2[5]**3/(x2[4]**2))/(5.35+4/(x2[2]/x2[4])**2)/(x2[2]/x2[6]),
                         sbucky/(x2[5]**3/(x2[2]**2))/(5.35+4/(x2[4]/x2[2])**2)/(x2[2]/x2[6]),
                         min(CBSafFac)**-1,
                         Kbending/(x2[0]*x2[1]*x2[2]**2/(x2[0]+x2[1]))+Kshear/(x2[3]*x2[2]/x2[4]),
                         Kbending/(x2[0]*x2[1]*x2[2]**2/(x2[0]+x2[1]))+Kshear/(x2[5]*x2[2]/x2[6])])**-1
    

sol2['RealSafFac'] = np.array([Med/MMAf.YIELD_TOP(h,tt,fy), 
                   Med/MMAf.YIELD_BOT(h, tb, fy), 
                   Vedx/MMAf.YIELD_SHEAR_X(h, twx, lx, fy), 
                   Vedy/MMAf.YIELD_SHEAR_Y(h, twy, ly, fy),
                   Ced/MMAf.YIELD_COMPRESSION(twx, lx, twy, ly, fy),
                   Medx_top1/MMAf.BUCK_INTERCELL_v2(Medx_top1, Medy_top1, tt, lx, ly, h, E),
                   Medx_top2/MMAf.BUCK_INTERCELL_v2(Medx_top2, Medy_top2, tt, lx, ly, h, E),
                   Vedx/MMAf.BUCK_SHEAR_X(twx, lx, ly, h, E), 
                   Vedy/MMAf.BUCK_SHEAR_Y(twy, lx, ly, h, E),
                   Ced/MMAf.BUCK_COMPRESSION(twx,lx,twy,ly,h,E,m_limit),
                   (Ded/MMAf.D_DEFLECTION(h, tt, tb, E) + Sed/MMAf.S_DEFLECTION(h, twx, lx, twy, ly, shear_mod))/(min(L,W)/360)])**-1
    
print(sol2['f0val'])
print(sol2['xval'])   

 
print("Delta max - %s" % (Ded/MMAf.D_DEFLECTION(h, tt, tb, E)+Sed/MMAf.S_DEFLECTION(h, twx, lx, twy, ly, shear_mod)))
print("Bending - %s" % (Ded/MMAf.D_DEFLECTION(h, tt, tb, E)))
print("Shear - %s" % (Sed/MMAf.S_DEFLECTION(h, twx, lx, twy, ly, shear_mod)))

print("KKT_norm: %s" % (kktnorm))
print("Number of iterations: %s" % (outit))

print(sol2['RealSafFac']**-1)

del m1,m2,n,m_list,n_list,m1_list,m2_list,m,m_limit
del Medx_top1,Medx_top2,Medy_top1,Medy_top2, Weight_max,bnds_x0,bounds, density,i

