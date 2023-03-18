import numpy as np
import math
import scipy.stats as stats
from scipy.linalg import sqrtm
import scipy
import matplotlib.pyplot as plt
import time
import datetime
import random
import statistics

import numba
from numba import jit, njit
from numba.typed import List


### 1.) BACKGROUND COMMON FUNCTIONS

def my_rbeta(num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1, with_h = False): 
    
    # Z & vector h
    Z_vec = np.random.normal(0,1,num_hypo)
    vec_h = np.random.binomial(1, p=prob_to_1, size=num_hypo)

    # mu's
    mu_vec_1 = vec_h * mu1 
    mu_vec_1_0 = np.where(mu_vec_1 == 0, mu0, mu_vec_1)

    # variance & std err
    var_vec_1 = vec_h * variance_1
    var_vec_1_0 = np.where(var_vec_1 == 0, variance_0, var_vec_1)
    sqrt_var_vec_1_0 = np.sqrt(var_vec_1_0)
    
    if not with_h:
        return sqrt_var_vec_1_0 * Z_vec + mu_vec_1_0
    else:
        return sqrt_var_vec_1_0 * Z_vec + mu_vec_1_0, vec_h


def SLFDR_decision_rule(locfdr, alpha):
    # recieves: SORTED np.array - locfdr of some group "k"  +  alpha
    # returns: number of rejections + olocfdr of the rejected ones
    
    locfdr_cumsum = np.cumsum(locfdr)
    rule_sums = locfdr_cumsum / np.array(list(range(1, len(locfdr_cumsum)+1))) 
    
    for statistic in rule_sums[::-1]:
        if statistic <= alpha:
            break
            
    if (statistic == rule_sums[0]) & (statistic > alpha):        
        num_rejections = 0
        rejections_olocfdr = 0        
        
    else:
        num_rejections = list(rule_sums).index(statistic) + 1
        rejections_olocfdr = locfdr[:num_rejections]
        
    return num_rejections, rejections_olocfdr
    
### 2.) PROCEDURES

## 2.1.) SLFDR
def my_SLFDR (alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1): 
    locfdr_agg = []
    final_R = [] 
    final_V = []
    
    for i in range(len(mu0)):    
        ###################################################################### creating RBETA instance 
        block_beta, vec_h = my_rbeta(num_hypo[i], 
                                     prob_to_1[i], 
                                     mu0[i], 
                                     mu1[i], 
                                     variance_0[i], 
                                     variance_1[i],
                                     True)
        ###################################################################### creating marginal_locfdr 
        dist_0 = stats.norm(mu0[i], np.sqrt(variance_0[i]))
        dist_1 = stats.norm(mu1[i], np.sqrt(variance_1[i]))
        margprob_ = (1-prob_to_1[i]) * dist_0.pdf(block_beta)  +  prob_to_1[i] * dist_1.pdf(block_beta)
        marglocfdr = (1-prob_to_1[i]) * dist_0.pdf(block_beta) / margprob_                        
        omarglocfdr = np.sort(marglocfdr)
  
        ###################################################################### Decision rule
        num_rejections, rejections_olocfdr = SLFDR_decision_rule(omarglocfdr, alpha)
        
        ###################################################################### CREATING "V" (for each group):
        
        #1. take the "marglocfdr" & "vec_h" , enumerate it as a dictionary 
        marglocfdr_d = dict(enumerate(marglocfdr))
        vec_h_d = dict(enumerate(vec_h))
        
        #2. sort the locfdr (values), so that their index (keys) is shuffeled
        omarglocfdr_d = dict(sorted(marglocfdr_d.items(), key=lambda x: x[1]))
    
        #3. take only the "num_rejections" first SHUFFELED indexes from "omarglocfdr_d"
        first_indexes = list(omarglocfdr_d.keys())[:num_rejections]

        #4. take only the "first_indexes" indexes from the enumerated vec_h dictionary, the vector of their values: "values_h"
        final_h = dict([(k, vec_h_d[k]) for k in first_indexes])

        #5. V = num_rejections - sum(values_h)
        V = num_rejections - sum(final_h.values())
        
        final_R.append(num_rejections)
        locfdr_agg.append(rejections_olocfdr)
        final_V.append(V)
    
    minprob = 0
    if sum(final_R) > 0:
        minprob = 1

    return locfdr_agg ,final_R, final_V, minprob 



## 2.2.) CLFDR
def my_CLFDR (alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1): 
    vec_h_agg = []
    locfdr_agg = []
    
    for i in range(len(mu0)):    
        ###################################################################### creating RBETA instance 
        block_beta, vec_h = my_rbeta(num_hypo[i], 
                                     prob_to_1[i], 
                                     mu0[i], 
                                     mu1[i], 
                                     variance_0[i], 
                                     variance_1[i],
                                     True)
        ###################################################################### creating marginal_locfdr 
        dist_0 = stats.norm(mu0[i], np.sqrt(variance_0[i]))
        dist_1 = stats.norm(mu1[i], np.sqrt(variance_1[i]))
        margprob_ = (1-prob_to_1[i]) * dist_0.pdf(block_beta)  +  prob_to_1[i] * dist_1.pdf(block_beta)
        marglocfdr = (1-prob_to_1[i]) * dist_0.pdf(block_beta) / margprob_                        
  
        ### Add it to the overall lists
        vec_h_agg.append(vec_h)
        locfdr_agg.append(marglocfdr)

    ###################################################################### h_stacking & Decision rule - using the SLFDR decision rule for the aggregated locfdrs
    vec_h_agg_stacked = np.hstack(vec_h_agg)
    locfdr_agg_stacked = np.hstack(locfdr_agg)
    omarglocfdr = np.sort(locfdr_agg_stacked)
    
    num_rejections, rejections_olocfdr = SLFDR_decision_rule(omarglocfdr, alpha)
        
    ###################################################################### CREATING "V" (for each group):
        
    #1. take the "marglocfdr" & "vec_h" , enumerate it as a dictionary 
    marglocfdr_d = dict(enumerate(locfdr_agg_stacked))
    vec_h_d = dict(enumerate(vec_h_agg_stacked))
        
    #2. sort the locfdr (values), so that their index (keys) is shuffeled
    omarglocfdr_d = dict(sorted(marglocfdr_d.items(), key=lambda x: x[1]))
    
    #3. take only the "num_rejections" first SHUFFELED indexes from "omarglocfdr_d"
    first_indexes = list(omarglocfdr_d.keys())[:num_rejections]

    #4. take only the "first_indexes" indexes from the enumerated vec_h dictionary, the vector of their values: "values_h"
    final_h = dict([(k, vec_h_d[k]) for k in first_indexes])

    #5. V = num_rejections - sum(values_h)
    V = num_rejections - sum(final_h.values())
    
    # adding minprob (=1 if R>0, =0 else)
    minprob = 0
    if num_rejections > 0:
        minprob = 1

    # return: 1.) locfdrs  2.) final_R  3.) final_V  4.) minprob
    return rejections_olocfdr, num_rejections, V, minprob 

## 2.3.) Ruth & Saharon

def DCpp_numba(R):
    K=len(R)
    D=[0]*K
    cR=0

    i=0
    while i<K:
        cR=0
        j=i
        while j<K:
            cR+=R[j]
            if cR>0:
                D[i]=1
                j+=1    # I think that this one is useless
                break
            j+=1
        if D[i]==0:
            i+=1
            break    
        i+=1
    return D
DCpp_numba_jit = jit()(DCpp_numba)

def BZCpp_numba(Tz):
    K= len(Tz)
    S = np.zeros((K,K))
    b = [0.0]*K
    
    S[0,0] = 1
    i = 1
    while i<K:
        S[i,0] = S[i-1,0]*(1-Tz[i-1])
        v=1
        while v<=i:
            S[i,v] = S[i-1,v]*(1-Tz[i-1]) + S[i-1,v-1]*Tz[i-1]
            v+=1
        i+=1
    b[0] = S[0,0]*Tz[0]
    i = 1
    while i<K:
        j = 0
        while j<=i:
            b[i] += S[i,j]*(((i-j)*Tz[i] - j*(1-Tz[i]))/(i*(i+1)))
            j+=1
        i+=1
    return b 
BZCpp_numba_jit = jit()(BZCpp_numba)

def FDR_Generic_structure (mus, a, b_1, b_2, ind, lev_mat, pow_mat, minprob_mat, ev_mat, er_mat, olocfdr_function, alpha = 0.05):
    for mui in range(len(mus)):                ##### we use "mui" as an INDEX
        mu = mus[mui]
        Rz = a-mu*b_1
        if (ind == 1) | (ind == 4):
            Rz[0] = Rz[0] + mu*alpha
            
        Rz_numba = numba.typed.List(Rz)
        Dz = list(DCpp_numba_jit(Rz_numba))
        
        indices_ab = [i for i, x in enumerate(Dz) if x == 1]
        sum_b = sum([b_2[j] for j in indices_ab])
        ### if sum_b <0: # it's here because u had a problem with that
        lev_mat[mui, ind] = lev_mat[mui,ind] + sum_b 
        sum_a = sum([a[j] for j in indices_ab])  
        pow_mat[mui, ind] = pow_mat[mui,ind] + sum_a 
        minprob_mat[mui, ind] = minprob_mat[mui,ind] + Dz[0] 
        sum_oloc = sum([olocfdr_function[j] for j in indices_ab])
        ev_mat[mui, ind] = ev_mat[mui, ind] + sum_oloc 
        er_mat[mui, ind] = er_mat[mui, ind] + sum(Dz) 
    return lev_mat, pow_mat, minprob_mat, ev_mat, er_mat

def fdep_marginals_groups(alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1, musMargFDR, musMargPFDR, musMargMFDR):

    marglocfdr_agg = []
    
    #  columns order: marglocfdrFDR, marglocfdrpFDR, marglocfdrmFDR
    minprob_mat = np.zeros((1,6),float)
    lev_mat = np.zeros((1,6),float) 
    pow_mat = np.zeros((1,6),float) 
    ev_mat = np.zeros((1,6),float) 
    er_mat = np.zeros((1,6),float) 
    
    
    # RUTH AND SAHARON LOGIC IS MORE LIKE "CLFDR" - WE TAKE EVERYTHING FROM ALL THE GROUPS, AND THEN COMBINE
    for i in range(len(mu0)):
        # - - - BLOCK BETA - - - 
        block_beta = my_rbeta(num_hypo[i], 
                                prob_to_1[i], 
                                mu0[i], 
                                mu1[i], 
                                variance_0[i], 
                                variance_1[i],
                                False)

        # - - - for marginal rule - - -
        dist_0 = stats.norm(mu0[i], np.sqrt(variance_0[i]))
        dist_1 = stats.norm(mu1[i], np.sqrt(variance_1[i]))
        margprob_ = (1-prob_to_1[i]) * dist_0.pdf(block_beta)  +  prob_to_1[i] * dist_1.pdf(block_beta)
        marglocfdr = (1-prob_to_1[i]) * dist_0.pdf(block_beta) / margprob_    
        marglocfdr_agg.append(marglocfdr)
    
    marglocfdr_agg_stacked = np.hstack(marglocfdr_agg)
    omarglocfdr = np.sort(marglocfdr_agg_stacked)
    amarg = 1 - omarglocfdr
    omarglocfdr_numba = numba.typed.List(omarglocfdr)    
    bmarg = np.array(BZCpp_numba_jit(omarglocfdr_numba)) # it WAS np.array but now it's a LIST of np.arrays so that when you take an index from it, it will be fine
    
    # marglocfdrFDR -v-     
    lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = FDR_Generic_structure (mus=musMargFDR, a=amarg, b_1=bmarg, 
                                                                               b_2=bmarg, ind=3, 
                                                                               lev_mat=lev_mat, 
                                                                               pow_mat=pow_mat, 
                                                                               minprob_mat=minprob_mat, 
                                                                               ev_mat=ev_mat, 
                                                                               er_mat=er_mat,  
                                                                               olocfdr_function=omarglocfdr, 
                                                                               alpha = alpha)
    # marglocfdrpFDR -v-
    lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = FDR_Generic_structure (mus=musMargPFDR, a=amarg, b_1=bmarg, 
                                                                               b_2=bmarg, ind=4, 
                                                                               lev_mat=lev_mat, 
                                                                               pow_mat=pow_mat, 
                                                                               minprob_mat=minprob_mat, 
                                                                               ev_mat=ev_mat, 
                                                                               er_mat=er_mat,  
                                                                               olocfdr_function=omarglocfdr,  
                                                                               alpha = alpha)
    # marglocfdrmFDR 
    lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = FDR_Generic_structure (mus=musMargMFDR, a=amarg, b_1=(omarglocfdr - alpha),
                                                                               b_2=bmarg, ind=5, 
                                                                               lev_mat=lev_mat, 
                                                                               pow_mat=pow_mat, 
                                                                               minprob_mat=minprob_mat, 
                                                                               ev_mat=ev_mat, 
                                                                               er_mat=er_mat,  
                                                                               olocfdr_function=omarglocfdr, 
                                                                               alpha = alpha)
    return lev_mat, pow_mat, minprob_mat, ev_mat, er_mat



### 3.) Mother function (and FNR before)

def FNR(num_hypo, prob_to_1, R_mean, V_mean):
    
    # calculating number of non-nulls + total number of hypotheses
    non_nulls = 0
    total_num_hypo = 0
    for i in range(len(num_hypo)):
        non_nulls += num_hypo[i]*prob_to_1[i]
        total_num_hypo += num_hypo[i]

    return (non_nulls - R_mean + V_mean) / (total_num_hypo - R_mean)

def mother_procedure(iter_num, alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1, musMargFDR_scalar, musMargPFDR_scalar, musMargMFDR_scalar):
    
    start = time.time()    

    # SLFDR
    slfdr_power, slfdr_fdr, slfdr_mfdr_V, slfdr_mfdr_R, slfdr_pfdr_minprob = [], [], [], [], []
    
    # CLFDR
    clfdr_power, clfdr_fdr, clfdr_mfdr_V, clfdr_mfdr_R, clfdr_pfdr_minprob = [], [], [], [], []
    
    # r & s
    #rs_power, rs_fdr, rs_mfdr_V, rs_mfdr_R, rs_pfdr_minprob = [], [], [], [], []
    lev_mat_agg = pow_mat_agg = minprob_mat_agg = ev_mat_agg = er_mat_agg = 0
    
    for i in range(iter_num):
        #if i == 4999:
        #    print(i)
        # SLFDR
        locfdr_agg, R_sl, V_sl, minprob_sl = my_SLFDR (alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1)
        
        slfdr_power.append(sum(R_sl) - sum(V_sl))
        if sum(R_sl) > 0:
            slfdr_fdr.append(sum(V_sl) / sum(R_sl))
        else: 
            slfdr_fdr.append(0)
        slfdr_mfdr_V.append(sum(V_sl))
        slfdr_mfdr_R.append(sum(R_sl))
        slfdr_pfdr_minprob.append(minprob_sl)
        
        # CLFDR
        rejections_olocfdr ,R_cl , V_cl, minprob_cl = my_CLFDR (alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1)

        clfdr_power.append(R_cl - V_cl)
        if R_cl > 0:
            clfdr_fdr.append(V_cl / R_cl)
        else: 
            clfdr_fdr.append(0)
        clfdr_mfdr_V.append(V_cl)
        clfdr_mfdr_R.append(R_cl)
        clfdr_pfdr_minprob.append(minprob_cl)        
        
        # r & s
        lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = fdep_marginals_groups (alpha, num_hypo, prob_to_1, mu0, mu1, variance_0, variance_1, musMargFDR_scalar, musMargPFDR_scalar, musMargMFDR_scalar)
        
        lev_mat_agg += lev_mat[0]
        pow_mat_agg += pow_mat[0]
        minprob_mat_agg += minprob_mat[0]
        ev_mat_agg += ev_mat[0]
        er_mat_agg += er_mat[0]
        
    # SLFDR RETURN
    slfdr_power_r = statistics.mean(slfdr_power)
    slfdr_fdr_r = statistics.mean(slfdr_fdr)
    if statistics.mean(slfdr_mfdr_R) != 0:
        slfdr_mfdr_r = statistics.mean(slfdr_mfdr_V) / statistics.mean(slfdr_mfdr_R)
    else:
        slfdr_mfdr_r = 0
    slfdr_pfdr_r = slfdr_fdr_r / statistics.mean(slfdr_pfdr_minprob)
    
    # CLFDR RETURN
    clfdr_power_r = statistics.mean(clfdr_power)
    clfdr_fdr_r = statistics.mean(clfdr_fdr)
    if statistics.mean(clfdr_mfdr_R) != 0:
        clfdr_mfdr_r = statistics.mean(clfdr_mfdr_V) / statistics.mean(clfdr_mfdr_R)
    else:
        clfdr_mfdr_r = 0
    clfdr_pfdr_r = clfdr_fdr_r / statistics.mean(clfdr_pfdr_minprob)

    # R & S
    # (IF I UNDERSTAND CORRECTRLY, WITH R & S WE WILL HAVE 3 X 4 = 12 NEW METRICS: EACH OF THE 4 METRIC FOR EACH Err Optimizer)
    pow_mat_r = pow_mat_agg / iter_num
    lev_mat_r = lev_mat_agg / iter_num
    ev_mat_r = ev_mat_agg / iter_num
    er_mat_r = er_mat_agg / iter_num
    minprob_mat_r = minprob_mat_agg / iter_num
    
    ## marginallocfdr FDR control
    rs_margFDR_power_r = pow_mat_r[3]
    rs_margFDR_fdr_r = lev_mat_r[3]
    rs_margFDR_mfdr_r = ev_mat_r[3] / er_mat_r[3]
    rs_margFDR_pfdr_r = lev_mat_r[3] / minprob_mat_r[3]

    ## marginallocfdr pFDR control
    rs_margpFDR_power_r = pow_mat_r[4]
    rs_margpFDR_fdr_r = lev_mat_r[4]
    rs_margpFDR_mfdr_r = ev_mat_r[4] / er_mat_r[4]
    rs_margpFDR_pfdr_r = lev_mat_r[4] / minprob_mat_r[4]
    
    ## marginallocfdr mFDR control
    rs_margmFDR_power_r = pow_mat_r[5]
    rs_margmFDR_fdr_r = lev_mat_r[5]
    rs_margmFDR_mfdr_r = ev_mat_r[5] / er_mat_r[5]
    rs_margmFDR_pfdr_r = lev_mat_r[5] / minprob_mat_r[5]
    
    # FNR for each one of the 5 procedures
    slfdr_fnr_r = FNR(num_hypo, prob_to_1, statistics.mean(slfdr_mfdr_R), statistics.mean(slfdr_mfdr_V))
    clfdr_fnr_r = FNR(num_hypo, prob_to_1, statistics.mean(clfdr_mfdr_R), statistics.mean(clfdr_mfdr_V))    
    rs_margFDR_fnr_r = FNR(num_hypo, prob_to_1, er_mat_r[3], ev_mat_r[3]) 
    rs_margpFDR_fnr_r = FNR(num_hypo, prob_to_1, er_mat_r[4], ev_mat_r[4])
    rs_margmFDR_fnr_r = FNR(num_hypo, prob_to_1, er_mat_r[5], ev_mat_r[5])
    
    # Arranging the final output
    slfdr_list = [slfdr_power_r, slfdr_fdr_r, slfdr_mfdr_r, slfdr_pfdr_r, slfdr_fnr_r]
    clfdr_list = [clfdr_power_r, clfdr_fdr_r, clfdr_mfdr_r, clfdr_pfdr_r, clfdr_fnr_r]
    rs_margFDR_list = [rs_margFDR_power_r, rs_margFDR_fdr_r, rs_margFDR_mfdr_r, rs_margFDR_pfdr_r, rs_margFDR_fnr_r]
    rs_margpFDR_list = [rs_margpFDR_power_r, rs_margpFDR_fdr_r, rs_margpFDR_mfdr_r, rs_margpFDR_pfdr_r, rs_margpFDR_fnr_r]
    rs_margmFDR_list = [rs_margmFDR_power_r, rs_margmFDR_fdr_r, rs_margmFDR_mfdr_r, rs_margmFDR_pfdr_r, rs_margmFDR_fnr_r]

    stop = time.time()
    duration = stop-start
    print("TIME IT TOOK FOR THE REST OF MOTHER TO RUN, AFTER ALL THE ITERATIONS WERE DONE: " + str(duration))
    return  slfdr_list, clfdr_list, rs_margFDR_list, rs_margpFDR_list, rs_margmFDR_list


### 4.) graphs

def plot_metric(metric_index, metric_tite_str, metric_realizations_x_axis, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR):
    # this function needs as an input a series of NESTED LISTS per policy procedure
    
    slfdr_metric_list = [policy_realization[metric_index] for policy_realization in slfdr]
    clfdr_metric_list = [policy_realization[metric_index] for policy_realization in clfdr]
    rs_margFDR_metric_list = [policy_realization[metric_index] for policy_realization in rs_margFDR]
    rs_margpFDR_metric_list = [policy_realization[metric_index] for policy_realization in rs_margpFDR]
    rs_margmFDR_metric_list = [policy_realization[metric_index] for policy_realization in rs_margmFDR]
        
    plt.plot(metric_realizations_x_axis, slfdr_metric_list, 'b-' , label = "SLFDR")
    plt.plot(metric_realizations_x_axis, clfdr_metric_list , 'g--' , label = "CLFDR")
    plt.plot(metric_realizations_x_axis, rs_margFDR_metric_list , 'r-.' , label = "margFDR") # was rs_ before
    plt.plot(metric_realizations_x_axis, rs_margpFDR_metric_list , 'k:' , label = "margpFDR") # was rs_ before
    plt.plot(metric_realizations_x_axis, rs_margmFDR_metric_list , 'y' , label = "margmFDR") # was rs_ before

    plt.title(metric_tite_str)
    plt.legend()
    plt.show()

def plot_4_metrics(p1_simulation, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR):
    # this function needs: the output of the 5 procedures + the X axis simulations change
    
    plot_metric(0, "Power - E(TP)", p1_simulation, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR)
    plot_metric(4, "mFNR", p1_simulation, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR)
    plot_metric(1, "FDR", p1_simulation, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR)
    plot_metric(2, "mFDR", p1_simulation, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR)
    plot_metric(3, "pFDR", p1_simulation, slfdr, clfdr, rs_margFDR, rs_margpFDR, rs_margmFDR)
