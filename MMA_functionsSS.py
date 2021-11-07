import numpy as np
from scipy.sparse import spdiags
import math

"""
Script for MMA optimisation

For Python users, note that the constraints have to be of the type <=0, unlike 
what is usually done in scipy

MMA_init initializes the process:
    *** Input:
        Objective Function
        Jacobian of objective Function
        Initial value
        Boundaries
        Constraints
    *** Output:
        Optimal value of Objective Function
        Value of variables
        Value of constraints

"""


def MMA_init(ObjFun, x0, jac, bounds, cons):

    m = len(cons)
    n = len(x0)
    
    var_min = np.zeros(n)
    var_max = np.zeros(n)
    for i in range(len(bounds)):
        var_min[i] = bounds[i][0]
        var_max[i] = bounds[i][1]
    
    xmin  = var_min
    xmax  = var_max
    xval  = x0
    xold1 = xval
    xold2 = xval
    
    low   = xmin
    upp   = xmax
    c = 1000*np.ones(m)
    d = np.ones(m)
    a0 = 1
    a = np.zeros(m)
    
    outeriter = 0
    maxoutit  = 100
    kkttol  = 2e-7
    
    """
    If outeriter=0, the user should now calculate function values
    and gradients of the objective- and constraint functions at xval.
    The results should be put in f0val, df0dx, fval and dfdx:
    """
    
    fval = np.zeros((m))
    dfdx = np.zeros((m,n))
    outvector1 = np.zeros(n+1)
    outvector2 = np.zeros(m+1)
    
    if outeriter < 0.5:
        f0val = ObjFun(xval)
        df0dx = jac(xval)
        for i in range(m):
            fval[i]=cons[i]['fun'](xval)
            dfdx[i]=cons[i]['jac'](xval)
        outvector1[0] = outeriter
        outvector1[1:] = xval
        outvector2[0] = f0val
        outvector2[1:] = fval
    
                     
    "The iterations start"
    
    kktnorm = kkttol+10
    outit = 0
    while kktnorm > kkttol and outit < maxoutit:
        outit   = outit+1;
        outeriter = outeriter+1;
        
        "The MMA subproblem is solved at the point xval:"
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = MMASub(m,n,outeriter,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d)
                                
        "Some vectors are updated:"
        xold2 = xold1;
        xold1 = xval;
        xval  = xmma;
    
        """The user should now calculate function values and gradients
        of the objective- and constraint functions at xval.
        The results should be put in f0val, df0dx, fval and dfdx."""
      
        f0val = ObjFun(xval)
        df0dx = jac(xval)
        for i in range(m):
            fval[i]=cons[i]['fun'](xval)
            dfdx[i]=cons[i]['jac'](xval)
        "The residual vector of the KKT conditions is calculated:"
        residu,kktnorm,residumax = kktcheck(m,n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d)
        outvector1 = np.concatenate(([outeriter], xval.T))
        outvector2 = np.concatenate(([f0val], fval.T))
        
    sol = {'f0val': f0val, 'xval': xval, 'df0dx': df0dx, 'fval': fval, 'dfdx': dfdx}
        
    return sol, outit, kktnorm


def subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d):
    """
    This function subsolv solves the MMA subproblem:
            
    minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
             + SUM[ ci*yi + 0.5*di*(yi)^2 ],

    subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
               alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
           
    Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
    """
    een = np.ones(n)
    eem = np.ones(m)
    epsi = 1
    epsvecn = epsi*een
    epsvecm = epsi*eem
    x = 0.5*(alfa+beta)
    y = eem
    z = 1
    lam = eem
    xsi = een/(x-alfa)
    xsi = np.maximum(xsi,een)
    eta = een/(beta-x)
    eta = np.maximum(eta,een)
    mu  = np.maximum(eem,0.5*c)
    zet = 1
    s = eem
    itera = 0
    while epsi > epsimin:
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        uxinv1 = een/ux1
        xlinv1 = een/xl1
        plam = p0 + P.T.dot(lam)
        qlam = q0 + Q.T.dot(lam)
        gvec = P.dot(uxinv1) + Q.dot(xlinv1)
        dpsidx = plam/ux2 - qlam/xl2
        rex = dpsidx - xsi + eta
        rey = c + d*y - mu - lam
        rez = a0 - zet - a.T.dot(lam)
        relam = gvec - a*z - y + s - b
        rexsi = xsi*(x-alfa) - epsvecn
        reeta = eta*(beta-x) - epsvecn
        remu = mu*y - epsvecm
        rezet = zet*z - epsi
        res = lam*s - epsvecm
        residu1 = np.concatenate((rex, rey, [rez]))
        residu2 = np.concatenate((relam, rexsi, reeta, remu, [rezet], res))
        residu = np.concatenate((residu1, residu2))
        residunorm = np.sqrt(residu.T.dot(residu))
        residumax = max(abs(residu))
        ittt = 0
        while residumax > 0.9*epsi and ittt < 200:
            ittt=ittt + 1
            itera=itera + 1
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            plam = p0 + P.T.dot(lam)
            qlam = q0 + Q.T.dot(lam)
            gvec = P.dot(uxinv1) + Q.dot(xlinv1)
            GG = P*spdiags(uxinv2,0,n,n) - Q*spdiags(xlinv2,0,n,n)
            dpsidx = plam/ux2 - qlam/xl2
            delx = dpsidx - epsvecn/(x-alfa) + epsvecn/(beta-x)
            dely = c + d*y - lam - epsvecm/y
            delz = a0 - a.T.dot(lam) - epsi/z;
            dellam = gvec - a*z - y - b + epsvecm/lam
            diagx = plam/ux3 + qlam/xl3
            diagx = 2*diagx + xsi/(x-alfa) + eta/(beta-x)
            diagxinv = een/diagx
            diagy = d + mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam+diagyinv
            "THIS m < n: section was not properly tested but should work"
            if m < n:
                blam = dellam + dely/diagy - GG.dot(delx/diagx)
                bb = np.concatenate((blam, delz))
                Alam = spdiags(diaglamyi,0,m,m) + GG.dot(spdiags(diagxinv,0,n,n).dot(GG.T))
                AA = np.zeros([m+1,m+1])
                AA[0:m,0:m] = Alam
                AA[m,0:m] = a
                AA[0:m,m] = a
                AA[m,m] = -zet/z
                solut = np.linalg.inv(AA).dot(bb)
                dlam = solut[0:m]
                dz = solut[m]
                dx = -delx/diagx - GG.T.dot(dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam + dely/diagy
                Axx = spdiags(diagx,0,n,n) + GG.T.dot(spdiags(diaglamyiinv,0,m,m).dot(GG))
                azz = zet/z + a.dot(a/diaglamyi)
                axz = -GG.T.dot(a/diaglamyi)
                bx = delx + GG.T.dot(dellamyi/diaglamyi)
                bz  = delz - a.T.dot(dellamyi/diaglamyi)
                AA = np.zeros([n+1,n+1])
                AA[0:n,0:n] = Axx
                AA[n,0:n] = axz
                AA[0:n,n] = axz
                AA[n,n] = azz
                bb = np.concatenate((-bx, [-bz]))
                solut = np.linalg.inv(AA).dot(bb)
                dx  = solut[0:n]
                dz = solut[n]
                dlam = GG.dot(dx)/diaglamyi - dz*(a/diaglamyi) + dellamyi/diaglamyi

            dy = -dely/diagy + dlam/diagy;
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
            dmu  = -mu + epsvecm/y - (mu*dy)/y
            dzet = -zet + epsi/z - zet*dz/z
            ds   = -s + epsvecm/lam - (s*dlam)/lam
            xx  = np.concatenate((y,  [z],  lam,  xsi,  eta,  mu,  [zet],  s))
            dxx = np.concatenate((dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds))

            stepxx = -1.01*dxx/xx
            stmxx  = max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = max(stepbeta)
            stmalbe  = np.maximum(stmalfa,stmbeta)
            stmalbexx = np.maximum(stmalbe,stmxx)
            stminv = np.maximum(stmalbexx,1)
            steg = 1/stminv
    
            xold   =   x
            yold   =   y
            zold   =   z
            lamold =  lam
            xsiold =  xsi
            etaold =  eta
            muold  =  mu
            zetold =  zet
            sold   =   s

            itto = 0;
            resinew = 2*residunorm;
            while resinew > residunorm and itto < 50:
                itto = itto+1
                x   =   xold + steg*dx
                y   =   yold + steg*dy
                z   =   zold + steg*dz
                lam = lamold + steg*dlam
                xsi = xsiold + steg*dxsi
                eta = etaold + steg*deta
                mu  = muold  + steg*dmu
                zet = zetold + steg*dzet
                s   =   sold + steg*ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0 + P.T.dot(lam)
                qlam = q0 + Q.T.dot(lam) 
                gvec = P.dot(uxinv1) + Q.dot(xlinv1)
                dpsidx = plam/ux2 - qlam/xl2 
                rex = dpsidx - xsi + eta
                rey = c + d*y - mu - lam
                rez = a0 - zet - a.T.dot(lam)
                relam = gvec - a.dot(z) - y + s - b
                rexsi = xsi*(x-alfa) - epsvecn
                reeta = eta*(beta-x) - epsvecn
                remu = mu*y - epsvecm
                rezet = zet*z - epsi
                res = lam*s - epsvecm
                residu1 = np.concatenate((rex, rey, [rez]))
                residu2 = np.concatenate((relam, rexsi, reeta, remu, [rezet], res))
                residu = np.concatenate((residu1, residu2))
                resinew = np.sqrt(residu.T.dot(residu))
                steg = steg/2
            residunorm=resinew
            residumax = max(abs(residu))
            steg = 2*steg
#        if ittt > 198:
#            print(epsi)
#            print(ittt)
        epsi = 0.1*epsi
    xmma   =   x
    ymma   =   y
    zmma   =   z
    lamma =  lam
    xsimma =  xsi
    etamma =  eta
    mumma  =  mu
    zetmma =  zet
    smma   =   s

    return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma

def MMASub(m,n,itera,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d):
    
    """
    Version September 2007 (and a small change August 2008)
    
    Krister Svanberg <krille@math.kth.se>
    Department of Mathematics, SE-10044 Stockholm, Sweden.
    
    This function mmasub performs one MMA-iteration, aimed at
    solving the nonlinear programming problem:
         
      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmin_j <= x_j <= xmax_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m
    *** INPUT:
    
    m     = The number of general constraints.
    n     = The number of variables x_j.
    iter  = Current iteration number ( =1 the first time mmasub is called).
    xval  = Column vector with the current values of the variables x_j.
    xmin  = Column vector with the lower bounds for the variables x_j.
    xmax  = Column vector with the upper bounds for the variables x_j.
    xold1 = xval, one iteration ago (provided that iter>1).
    xold2 = xval, two iterations ago (provided that iter>2).
    f0val = The value of the objective function f_0 at xval.
    df0dx = Column vector with the derivatives of the objective function
            f_0 with respect to the variables x_j, calculated at xval.
    fval  = Column vector with the values of the constraint functions f_i,
            calculated at xval.
    dfdx  = (m x n)-matrix with the derivatives of the constraint functions
            f_i with respect to the variables x_j, calculated at xval.
            dfdx(i,j) = the derivative of f_i with respect to x_j.
    low   = Column vector with the lower asymptotes from the previous
            iteration (provided that iter>1).
    upp   = Column vector with the upper asymptotes from the previous
            iteration (provided that iter>1).
    a0    = The constants a_0 in the term a_0*z.
    a     = Column vector with the constants a_i in the terms a_i*z.
    c     = Column vector with the constants c_i in the terms c_i*y_i.
    d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
         
    *** OUTPUT:
    
    xmma  = Column vector with the optimal values of the variables x_j
            in the current MMA subproblem.
    ymma  = Column vector with the optimal values of the variables y_i
            in the current MMA subproblem.
    zmma  = Scalar with the optimal value of the variable z
            in the current MMA subproblem.
    lam   = Lagrange multipliers for the m general MMA constraints.
    xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
    mu    = Lagrange multipliers for the m constraints -y_i <= 0.
    zet   = Lagrange multiplier for the single constraint -z <= 0.
    s     = Slack variables for the m general MMA constraints.
    low   = Column vector with the lower asymptotes, calculated and used
            in the current MMA subproblem.
    upp   = Column vector with the upper asymptotes, calculated and used
            in the current MMA subproblem.
    
    epsimin = sqrt(m+n)*10^(-9);
    """
    epsimin = 10**(-7)
    raa0 = 0.00001
    move = 0.5
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.5
    eeen = np.ones(n)
#    eeem = np.ones(m)
#    zeron = np.zeros(n)
    
    "Calculation of the asymptotes low and upp :"
    if itera < 2.5:
        low = xval - asyinit*(xmax-xmin)
        upp = xval + asyinit*(xmax-xmin)
    else:
        zzz = (xval-xold1)*(xold1-xold2)
        factor = np.ones(n)
        factor[zzz>0] = asyincr
        factor[zzz<0] = asydecr
        low = xval - factor*(xold1 - low)
        upp = xval + factor*(upp - xold1)
        lowmin = xval - 10*(xmax-xmin)
        lowmax = xval - 0.01*(xmax-xmin)
        uppmin = xval + 0.01*(xmax-xmin)
        uppmax = xval + 10*(xmax-xmin)
        low = np.maximum(low,lowmin)
        low = np.minimum(low,lowmax)
        upp = np.minimum(upp,uppmax)
        upp = np.maximum(upp,uppmin)
    
    "Calculation of the bounds alfa and beta :"
    
    zzz1 = low + albefa*(xval-low)
    zzz2 = xval - move*(xmax-xmin)
    zzz  = np.maximum(zzz1,zzz2)
    alfa = np.maximum(zzz,xmin)
    zzz1 = upp - albefa*(upp-xval)
    zzz2 = xval + move*(xmax-xmin)
    zzz  = np.minimum(zzz1,zzz2)
    beta = np.minimum(zzz,xmax)
    
    "% Calculations of p0, q0, P, Q and b."
    
    xmami = xmax-xmin
    xmamieps = 0.00001*eeen
    xmami = np.maximum(xmami,xmamieps)
    xmamiinv = eeen/xmami
    ux1 = upp-xval
    ux2 = ux1*ux1
    xl1 = xval-low
    xl2 = xl1*xl1
    uxinv = eeen/ux1
    xlinv = eeen/xl1
    #p0 = zeron;
    #q0 = zeron;
    p0 = np.maximum(df0dx,0)
    q0 = np.maximum(-df0dx,0)
    #p0(find(df0dx > 0)) = df0dx(find(df0dx > 0))
    #q0(find(df0dx < 0)) = -df0dx(find(df0dx < 0))
    pq0 = 0.001*(p0 + q0) + raa0*xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0*ux2
    q0 = q0*xl2
    #
    #P = sparse(m,n);
    #Q = sparse(m,n);
    P = np.maximum(dfdx,0)
    Q = np.maximum(-dfdx,0)
    #P(find(dfdx > 0)) = dfdx(find(dfdx > 0));
    #Q(find(dfdx < 0)) = -dfdx(find(dfdx < 0));
    PQ = 0.001*(P + Q) + raa0*(np.outer(d,df0dx))
    P = P + PQ
    Q = Q + PQ
    P = P * spdiags(ux2,0,n,n)
    Q = Q * spdiags(xl2,0,n,n)
    b = P.dot(uxinv) + Q.dot(xlinv) - fval
#    print(b)
        
    "%%% Solving the subproblem by a primal-dual Newton method"
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d)    
        
    return xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp

    
def kktcheck(m,n,x,y,z,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d):

    """
    This is the file kktcheck.m
    Version Dec 2006.
    Krister Svanberg <krille@math.kth.se>




    The left hand sides of the KKT conditions for the following
    nonlinear programming problem are calculated.
         
    Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmax_j <= x_j <= xmin_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m
    *** INPUT:

    m     = The number of general constraints.
    n     = The number of variables x_j.
    x     = Current values of the n variables x_j.
    y     = Current values of the m variables y_i.
    z     = Current value of the single variable z.
    lam   = Lagrange multipliers for the m general constraints.
    xsi   = Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
    eta   = Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
    mu    = Lagrange multipliers for the m constraints -y_i <= 0.
    zet   = Lagrange multiplier for the single constraint -z <= 0.
    s     = Slack variables for the m general constraints.
    xmin  = Lower bounds for the variables x_j.
    xmax  = Upper bounds for the variables x_j.
    df0dx = Vector with the derivatives of the objective function f_0
            with respect to the variables x_j, calculated at x.
    fval  = Vector with the values of the constraint functions f_i,
            calculated at x.
    dfdx  = (m x n)-matrix with the derivatives of the constraint functions
            f_i with respect to the variables x_j, calculated at x.
            dfdx(i,j) = the derivative of f_i with respect to x_j.
    a0   =  The constants a_0 in the term a_0*z.%
    a    =  Vector with the constants a_i in the terms a_i*z.
    c    =  Vector with the constants c_i in the terms c_i*y_i.
    d    =  Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
     
    *** OUTPUT:
    
    residu     = the residual vector for the KKT conditions.
    residunorm = sqrt(residu'*residu).
    residumax  = max(abs(residu)).
    
    """

    rex   = df0dx + dfdx.T.dot(lam) - xsi + eta
    rey   = c + d*y - mu - lam
    rez   = a0 - zet - a.T.dot(lam)
    relam = fval - a.dot(z) - y + s
    rexsi = xsi*(x-xmin)
    reeta = eta*(xmax-x)
    remu  = mu*y
    rezet = zet*z
    res   = lam*s
    
    residu1 = np.concatenate((rex, rey, [rez]))
    residu2 = np.concatenate((relam, rexsi, reeta, remu, [rezet], res))
    residu = np.concatenate((residu1, residu2))
    residunorm = np.sqrt(residu.T.dot(residu))
    residumax = max(abs(residu));
    
    return residu,residunorm,residumax





def YIELD_TOP(h, tt, fy):
    Mrd = fy*h*tt
    return Mrd

def YIELD_BOT(h, tb, fy):
    Mrd = fy*h*tb
    return Mrd

def YIELD_SHEAR_X(h, twx, lx, fy):
    Vrdx = fy*h*twx/lx/(3**(1/2))
    return Vrdx

def YIELD_SHEAR_Y(h, twy, ly, fy):
    Vrdy = fy*h*twy/ly/(3**(1/2))
    return Vrdy

def YIELD_COMPRESSION(twx, lx, twy, ly, fy):
    Crd = fy*(twx/lx+twy/ly)
    return Crd


def BUCK_INTERCELL_v2(Medx, Medy, tt, lx, ly, h, E):
    Sratio = Medy/Medx
    a = ly
    b = lx
    if a >= b:
        Mrdx = []
        m_list = [1,2,3]
        for m in m_list:
            K = (m**2*b**2+a**2)**2/(m**2*b**2+Sratio*a**2)
            Ncr = K * E * math.pi**2 * tt**3 / (12*(1-0.3**2)*a**2*b**2)
            Mrdx.append(Ncr * h)
    else:
        Mrdx = []
        n_list = [1,2,3]
        for n in n_list:
            K = (n**2*a**2+b**2)**2/(Sratio*n**2*a**2+b**2)
            Ncr = K * E * math.pi**2 * tt**3 / (12*(1-0.3**2)*a**2*b**2)
            Mrdx.append(Ncr * h)
    MrdIC_x = min(Mrdx)
    return MrdIC_x

def BUCK_SHEAR_X(twx, lx, ly, h, E):
    if h > ly:
        fi = h / ly
        K = 5.35 + 4 / fi**2
        Ncrx = K * E * math.pi**2 * twx**3 / (12*(1-0.3**2)*ly**2)
        Vrdx = Ncrx * h/ lx
    else:
        fi = ly / h
        K = 5.35 + 4 / fi**2
        Ncrx = K * E * math.pi**2 * twx**3 / (12*(1-0.3**2)*h**2)
        Vrdx = Ncrx * h/ lx
    return Vrdx


def BUCK_SHEAR_Y(twy, lx, ly, h, E):
    if h > lx:
        fi = h / lx
        K = 5.35 + 4 / fi**2
        Ncry = K * E * math.pi**2 * twy**3 / (12*(1-0.3**2)*lx**2)
        Vrdy = Ncry * h/ly
    else:
        fi = lx / h
        K = 5.35 + 4 / fi**2
        Ncry = K * E * math.pi**2 * twy**3 / (12*(1-0.3**2)*h**2)
        Vrdy = Ncry * h/ly
    return Vrdy


def BUCK_COMPRESSION(twx, lx, twy, ly, h, E, m_limit):
    m_list = [i for i in range(1,m_limit+1)]
    Nrdx = []
    for m in m_list:
        Kx = (m*ly/h+h/(m*ly))**2
        Ncrx = (Kx * E * math.pi**2 * twx**3 / (12*(1-0.3**2)*ly**2))/lx
        Nrdx.append(Ncrx)
    Nrdy = []
    for m in m_list:
        Ky = (m*lx/h+h/(m*lx))**2
        Ncry = (Ky * E * math.pi**2 * twy**3 / (12*(1-0.3**2)*lx**2))/ly
        Nrdy.append(Ncry)
    
#    print(Nrdx, Nrdy)
    Crd = min(Nrdx) + min(Nrdy)
    return Crd


def D_DEFLECTION(h, tt, tb, E):
    Drd = E*tt*tb*(h)**2/(tt+tb)/0.91
    return Drd

def S_DEFLECTION(h, twx, lx, twy, ly, G):
    Srd = G*h*min(twx/lx,twy/ly)
    return Srd


def WEIGHT(h, tt, tb, twx, lx, twy, ly, fy):
    W = 7.85*(tt+tb)+7.85*h*(twx/lx + twy/ly)+1e-6*fy
    return W

