from scipy import stats
import numpy as np
from scipy.stats import iqr


def density(x_vec, n = 1000):
    
    # Calculating Density based on R's formula
    # documentation of the difference: https://stackoverflow.com/questions/55366188/why-do-stat-density-r-ggplot2-and-gaussian-kde-python-scipy-differ
    
    # KDE initialization
    bw_scalar = 0.9 * min(np.std(x_vec), iqr(x_vec)/1.34) * len(x_vec)**(-0.2)
    kde_initial = stats.gaussian_kde(x_vec, bw_method = bw_scalar)
    
    # Creating the X vec
    interval_x = np.abs((min(x_vec) - 10) - (max(x_vec) + 10)) / (n-1)
    final_x = np.append(np.arange((min(x_vec) - 10), (max(x_vec) + 10), interval_x), max(x_vec) + 10)
    
    # Evaluation
    final_y = kde_initial.evaluate(final_x)
    
    return final_x, final_y


def EstNull_LIST(z_vec, gamma=0.1):
         
    # x is a vector of z-values
    # gamma is a parameter, default is 0.1
    # output: the estimated mean and standard deviation    
    
    n = len(z_vec)
    t = np.arange(1,1001) / 200
    
    gan = n**(-gamma)
    that   = 0 
    shat   = 0
    uhat   = 0
    epshat = 0
    
    phiplus   = [1]*1000
    phiminus  = [1]*1000
    dphiplus  = [1]*1000
    dphiminus = [1]*1000
    phi       = [1]*1000
    dphi      = [1]*1000
    
    for i in range(1000):
        s = t[i]
        phiplus[i] = np.mean(np.cos(s*z_vec))
        phiminus[i]  = np.mean(np.sin(s*z_vec))
        dphiplus[i]  = - np.mean(z_vec * np.sin(s*z_vec))
        dphiminus[i] = np.mean(z_vec * np.cos(s*z_vec))
        phi[i]       = np.sqrt(phiplus[i]**2 + phiminus[i]**2)
        
    ind = min(np.where(np.array(phi) - gan <= 0)[0]) # in R this will be +1 index, in python because the index starts with 0, we're good
    tt = t[ind]
    a  = phiplus[ind]
    b  = phiminus[ind]
    da = dphiplus[ind]
    db = dphiminus[ind]
    c  = phi[ind]
    
    that   = tt
    shat   = -(a*da + b*db)/(tt*c*c)
    shat   = np.sqrt(shat)          # IT'S WEIRD BECAUSE WE JUST APPLIED A MINUS TO "SHAT" ... 
    uhat   = -(da*b - db*a)/(c*c)
    epshat = 1 - c*np.exp((tt*shat)**2/2)
    
    mu = uhat
    s = shat
    musigma = [mu, s]
    
    return musigma


def lin_itp_LIST(x, X, Y):
    
    #""" ODED: i have no idea what is this function about with its inputs and so"""
    #""" + plus THERE MIGHT BE A PROBLEM WITH INDECIES, I'M NOT SURE"""
    ## x: the coordinates of points where the density needs to be interpolated
    ## X: the coordinates of the estimated densities
    ## Y: the values of the estimated densities
    ## the output is the interpolated densities    
    
    x_N = len(x)
    X_N = len(X)
    y = [0]*x_N
    
    for k in range(x_N):
        i = max(np.where(x[k] - X >=0)[0])
        
        if i < X_N:
            y[k] = Y[i] + (Y[i+1]-Y[i]) / (X[i+1]-X[i]) * (x[k]-X[i])
        else: 
            y[k] = Y[i]
    return np.array(y)


def epsest(x, u, sigma):
    
    # x is a vector
    # u is the mean
    # sigma is the standard deviation    
    
    z  = (x - u) / sigma
    xi = np.arange(0,101) / 100
    tmax = np.sqrt(np.log(len(x)))
    tt = np.arange(0, tmax, 0.1)
    
    epsest_list = []
    
    for j in range(len(tt)):
        
        t = tt[j]
        f = t*xi
        f = np.exp(f**2/2)
        w  = (1 - np.abs(xi))   # WHY ITS DECLARED HERE ??? 
        co  = 0*xi
        
        for i in range(100):
            co[i] = np.mean(np.cos(t*xi[i]*z))
        
        epshat = 1 - np.sum(w*f*co) / np.sum(w)  # I HAVE NO IDEA WHICH DIMENSIONS THESE ARE
        epsest_list.append(epshat)
    
    max_epsest = max(np.array(epsest_list))
    
    return max_epsest


def adpt_cutz(lfdr, alpha):
    
    # the input
        # lfdr the vector of local fdr statistics - SHOULD BE NP.ARRAY
        # alpha the desired FDR level
    # the output is a list with
        # the first element (st.lfdr) the sorted local fdr values
        # the second element (k) the number of hypotheses to be rejected
        # the third element (lfdrk) the threshold for the local fdr values
        # the fourth element (reject) the set of indices of the rejected hypotheses
        # the fifth element (accept) the set of indices of the accepted hypotheses
        
    m = len(lfdr)
    st_lfdr = np.sort(lfdr)
    k = 1
    
    while (k < m+1) & ((1/k) * np.sum(st_lfdr[:k]) < alpha):
        k+=1
    k = k-2                                             ######### I THINK!!! THIS ONE SHOULD BE CHECKED - I checked it 95%, BUT we kinda DON'T CARE, because its not effecting out LOCFDR's calculations 
    lfdrk = st_lfdr[k]
    
    reject = np.where(lfdr <= lfdrk)[0]
    accept = np.where(lfdr > lfdrk)[0]
    
    sf = st_lfdr 
    nr = k 
    thr = lfdrk
    re = reject
    ac = accept
    
    return [sf, nr, thr, re, ac]


def adaptZ(zv, alpha):
    
    # the input
      # zv is the z-values transformed from m tests
      # alpha is the desired FDR level
    # the output is a list with
      # the first element (st.lfdr) the sorted local fdr values
      # the second element (k) the number of hypotheses to be rejected
      # the third element (lfdrk) the threshold for the local fdr values
      # the fourth element (reject) the set of indices of the rejected hypotheses
      # the fifth element (accept) the set of indices of the accepted hypotheses    

    ## the estimates for the local fdr statistics
    # density estimates 
    X,Y = density(zv)

    # linear interpolation
    zv_ds = lin_itp_LIST(zv, X, Y)
    
    # estimating the null distribution
    zv_MuSigma = EstNull_LIST(zv)
    mu = zv_MuSigma[0]
    s = zv_MuSigma[1]
        
    # mu<-0; s<-1 ## oded - it was like this in the original function
    
    zv_p0 = 1 - epsest(zv, mu, s)
    dist = stats.norm(mu, s)
    
    zv_lfdr = zv_p0 * dist.pdf(zv) / zv_ds

    #y = adpt_cutz(zv_lfdr, alpha)
    
    return zv_lfdr, mu, s #y    