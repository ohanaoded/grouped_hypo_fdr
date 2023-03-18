import numpy as np
import math
import scipy.stats as stats
from scipy.linalg import sqrtm
import scipy
import matplotlib.pyplot as plt
import time
import datetime
import random

import numba
from numba import jit, njit
from numba.typed import List


### 1.) blocks

def blocks_generator(mu0, variance_0, mu1, variance_1, number_of_blocks, number_block_members, Rho):

    KMax = number_block_members

    block_matrices = list()
    block_sum_matrices = list()
    block_nones = list()
    block_detfacs = list()
    block_deltas = list()
    block_invdeltas = list()

    for block in range(number_of_blocks):
        covmat_base = np.array([[Rho]*(KMax)]*(KMax)) + np.identity(KMax)*variance_0 - np.identity(KMax)*Rho
        matnow = np.array([]).reshape(0,KMax)   
        nonenow = np.array([]).reshape(1,0)  
        detfacnow = np.array([]).reshape(1,0) 
        deltnow = np.array([]).reshape(1,0)  
        invdeltnow = np.array([]).reshape(1,0)
        
        bitmask = [2**(x+1) for x in range(KMax)] # This is the bitmask saharon was asking for.
        for i in range(2**(KMax)):                # WE'RE USING i ONLY AS A NUMBER, NOT AS AN INDEX - i is perfect 100%.
            modulu_i = [i%x for x in bitmask]
            divide_2 = [x/2 for x in bitmask]
            fn = np.fromiter((modulu_i[j] >= divide_2[j] for j in range(len(bitmask))), dtype=float)
            added_diag = np.zeros(((KMax),(KMax)),float)
            np.fill_diagonal(added_diag,list(fn*variance_1))
            covmat = covmat_base + added_diag
            invmat = np.linalg.inv(covmat)            
            matnow = np.concatenate((matnow, invmat), axis=0)                          
            nonenow = np.concatenate((nonenow, np.array([fn.sum()]).reshape(1,1)), axis=1) 
            detfacnow = np.concatenate((detfacnow, np.array([np.linalg.det(covmat)**(-0.5)]).reshape(1,1)), axis=1) 
            fnmu = np.where(fn == 0, mu0, mu1)
            deltnow = np.concatenate((deltnow, np.array(fnmu).reshape(1,KMax)), axis=1) 
            invdeltnow = np.concatenate((invdeltnow, np.array(np.matmul(invmat,fnmu)).reshape(1,KMax)), axis=1)

        sum_matrix = np.array([], dtype=np.int64).reshape(0,2**(KMax))   
        
        for i in range(KMax):                       # HERE WERE USING i AS A NUMBER, AND NOT AS AN INDEX
            modulu_2i = [x%(2**(i+1)) for x in list(range(2**(KMax)))]                      
            cols = np.array([modulu_2i[k] >= (2**(i+1))/2 for k in range(len(modulu_2i))]) 
            sum_matrix = np.concatenate((sum_matrix,np.matrix(cols*1)), axis=0)        # THIS SUM_MATRIX IS ADDING CLOSED LISTS AND -NOT- CONCATING

        block_matrices.append(matnow)    
        block_sum_matrices.append(sum_matrix)
        block_nones.append(nonenow)
        block_detfacs.append(detfacnow)
        block_deltas.append(deltnow)
        block_invdeltas.append(invdeltnow)  

    block_matrices = np.array(block_matrices)
    block_sum_matrices = np.array(block_sum_matrices)
    block_nones = np.array(block_nones)
    block_detfacs = np.array(block_detfacs)
    block_deltas = np.array(block_deltas)
    block_invdeltas = np.array(block_invdeltas)

    return block_matrices, block_sum_matrices, block_nones, block_detfacs, block_deltas, block_invdeltas


### 2.) rbeta

def rbeta_generator (mu0, variance_0, mu1, variance_1, number_of_blocks, number_block_members, Rho, prob_to_1, maxiter):
    
    K = number_block_members
    
    a = np.full((maxiter, number_of_blocks,1,K,K), (np.array([Rho]*number_of_blocks)*variance_0)[0])
    print("A")
    b = np.full((maxiter, number_of_blocks,1,K,K), np.identity(K)*(1 - np.array(Rho)))
    print("b")
    new_h = np.array([np.array([np.full((1,1,K),np.random.binomial(1, p=prob_to_1, size=K)) for i in range(number_of_blocks)]) for i in range(maxiter)])
    print("new_h")
    new_c = np.array([np.array([np.full((1,K,K), np.identity(K)) for i in range(number_of_blocks)]) for i in range(maxiter)])
    print("new_c")
    c = new_c * new_h * variance_1
    covmat = (a+b)*variance_0 + c
    sqrtmat = np.real(np.array([np.array([sqrtm(covmat[j][i][0]) for i in range(covmat.shape[1])]).reshape(number_of_blocks,1,K,K) for j in range(covmat.shape[0])]).reshape(maxiter,number_of_blocks,1,K,K))
    print("sqrtmat")
    new_h_mus = np.where(new_h == 0, mu0, mu1)
    beta = np.array([np.array([np.matmul(sqrtmat[j][i][0],np.random.normal(0,1,K)) for i in range(number_of_blocks)]) for j in range(maxiter)]) + (new_h_mus).reshape(maxiter,number_of_blocks,K)
    print("beta")
    
    return list(beta), new_h.reshape(maxiter,number_of_blocks,K)


### 3.) locfdr

def AandB_locfdr_generator (block_beta, number_of_blocks, number_block_members, prob_to_1, maxiter, 
                            block_matrices, block_deltas, block_invdeltas, block_sum_matrices, block_nones, block_detfacs):
    
    vec_len = number_of_blocks
    K = number_block_members
    
    # S1_1
    block_matrices_vec = np.full((maxiter, vec_len, block_matrices.shape[1], K), block_matrices)
    block_beta_vec = np.asarray(block_beta).reshape(maxiter, vec_len, K, 1)
    S1_1 = np.matmul(block_matrices_vec, block_beta_vec).reshape(maxiter, vec_len, 1, block_matrices.shape[1])

    # S1_2
    deltas_bet_vec = np.full((maxiter, vec_len, 1, block_deltas.shape[2]), 2 * block_deltas)
    deltas_bet_reshape = deltas_bet_vec.reshape(maxiter, vec_len,int(block_deltas.shape[2]/K), K)
    block_beta_vec = np.array(block_beta).reshape(maxiter,vec_len,1,K)                     
    S1_2 = (block_beta_vec - deltas_bet_reshape).reshape(maxiter, vec_len, 1, block_deltas.shape[2])

    # S1_3
    S1_3 = np.full((maxiter,block_deltas.shape[0],block_deltas.shape[1],block_deltas.shape[2]), block_deltas*block_invdeltas)
    S1 = S1_1 * S1_2 + S1_3
    M1_before_T = S1.reshape(maxiter, vec_len, int(block_deltas.shape[2]/K), K)
    M1 = np.transpose(M1_before_T, axes=(0,1,3,2))    
    
    # pzh
    block_detfacs_vec = np.full((maxiter, block_detfacs.shape[0], block_detfacs.shape[1], block_detfacs.shape[2]),block_detfacs)
    pzh_1 = block_detfacs_vec * np.exp(M1.sum(axis=2)*(-0.5)).reshape(maxiter, vec_len, 1, int(block_deltas.shape[2]/K))
    block_nones_vec = np.full((maxiter, block_nones.shape[0], block_nones.shape[1], block_nones.shape[2]), block_nones)
    pzh_2 = (prob_to_1**block_nones_vec) * ((1-prob_to_1)**(K - block_nones_vec))
    pzh = pzh_1*pzh_2
    
    # locfdr
    block_sum_matrices_vec = np.full((maxiter, block_sum_matrices.shape[0], block_sum_matrices.shape[1], block_sum_matrices.shape[2]), block_sum_matrices)
    numerator =  np.matmul(block_sum_matrices_vec, pzh.reshape(maxiter, vec_len,pzh.shape[3],1))      #mone
    denominator = np.sum(pzh,axis=3)[:,:,None]                                                        #mechane  #.shape i need it to be (50,10,1,1)
    locfdr = np.array([np.concatenate((1 - (numerator / denominator))[i]) for i in range(maxiter)])
    
    return locfdr

def AandB_locfdr_generator_optimizer(block_beta, number_of_blocks, number_block_members, prob_to_1, maxiter, 
                                     block_matrices, block_deltas, block_invdeltas, block_sum_matrices, block_nones, block_detfacs):
    
    # dividing the process into loops of 625 iterations (it was 1250 (5 places), checking if it's possible)
    
    locfdr_optimizer = []
    
    loop_times = maxiter//1250
    if maxiter%1250 > 0:
        loop_times += 1
    for i in range(loop_times):
        print("loop no.: " + str(i))
        iterations = 1250
        iter_block_beta = block_beta[i*iterations:(i+1)*iterations]
        
        if (i == loop_times-1) and (maxiter%1250 > 0):
            iterations = maxiter%1250
            iter_block_beta = block_beta[-iterations:]
        
        locfdr = AandB_locfdr_generator (block_beta = iter_block_beta, 
                                         number_of_blocks = number_of_blocks, 
                                         number_block_members = number_block_members, 
                                         prob_to_1 = prob_to_1, 
                                         maxiter = iterations, 
                                         block_matrices = block_matrices, 
                                         block_deltas = block_deltas, 
                                         block_invdeltas = block_invdeltas, 
                                         block_sum_matrices = block_sum_matrices, 
                                         block_nones = block_nones, 
                                         block_detfacs = block_detfacs)        
        locfdr_optimizer.append(locfdr)

    return np.concatenate(locfdr_optimizer,axis=0)
 
### 4.) locfdr ALL GROUPS process + 2 concatenators

def locfdr_generator_all_groups(mu0, variance_0, mu1, variance_1, number_of_blocks, number_block_members, Rho, prob_to_1, maxiter):
    
    locfdr_holder = []
    h_holder = []
    for i in range(len(mu0)):
            
        # 1.) BLOCKS
        print(" * * * BLOCKS START: ")
        start = time.time()   
        block_matrices, block_sum_matrices, block_nones, block_detfacs, block_deltas, block_invdeltas = blocks_generator (mu0[i], 
                                                                                                                    variance_0[i], 
                                                                                                                    mu1[i], 
                                                                                                                    variance_1[i], 
                                                                                                                    number_of_blocks[i], 
                                                                                                                    number_block_members[i],
                                                                                                                    Rho[i])
        stop = time.time()
        duration = stop-start
        print(" * * * BLOCKS time: " + str(duration))
        
        # 2.) RBETA 
        print(" * * * BLOCK BETA START: ")
        start = time.time()   
        block_beta, new_h = rbeta_generator (mu0[i], variance_0[i], mu1[i], variance_1[i], number_of_blocks[i], number_block_members[i], Rho[i], prob_to_1[i], maxiter)
        stop = time.time()
        duration = stop-start
        print(" * * * BLOCK BETA time: " + str(duration))
        
        # 3.) loc fdr: Optimizer(AandB)
        print(" * * * AANDB LOCFDR GENERATOR START: ")
        start = time.time()   
        locfdr = AandB_locfdr_generator_optimizer (block_beta, number_of_blocks[i], number_block_members[i], prob_to_1[i], maxiter, block_matrices, block_deltas, block_invdeltas, block_sum_matrices, block_nones, block_detfacs)
        stop = time.time()
        duration = stop-start
        print(" * * * AANDB LOCFDR GENERATOR time: " + str(duration))
        
        locfdr_holder.append(locfdr)
        h_holder.append(new_h)
        
    return locfdr_holder, h_holder

def locfdr_concator(locfdr_master):
    list_concat_locfdr = []
    for i in range(len(locfdr_master)):
        list_concat_locfdr.append(locfdr_master[i])
    return np.concatenate(list_concat_locfdr, axis=1)

def vec_h_concator(vec_h):
    list_vec_h = []
    for i in range(len(vec_h)):
        block_to_group_array = np.array([np.concatenate(vec_h[i][j]) for j in range(len(vec_h[i]))])
        list_vec_h.append(block_to_group_array)
    return np.concatenate(list_vec_h, axis=1)


### 5.) slfdr decision rule & V & FNR

def SLFDR_decision_vectorized_rule(olocfdr, alpha):
    
    locfdr_cumsum = np.cumsum(olocfdr, axis=1).reshape(len(olocfdr),len(olocfdr[0]))
    ranks_vec = np.full((len(olocfdr),len(olocfdr[0])),np.array(list(range(1, len(olocfdr[0])+1))))
    rule_sums = locfdr_cumsum / ranks_vec
    
    num_rejections_list = []
    rejections_olocfdr_list = []
    
    for iter in range(len(olocfdr)):
        num_rejections = olocfdr[iter][olocfdr[iter] < alpha].shape[0]
        rejections_olocfdr = olocfdr[iter][:num_rejections]

        num_rejections_list.append(num_rejections)
        rejections_olocfdr_list.append(rejections_olocfdr)
        
    return num_rejections_list, rejections_olocfdr_list

def V_generator(locfdr_, vec_h_, num_rejections_list):
    
    # INPUT: original locfdr, original vec_h, number of rejections
    # OUTPUT: A list of "V" per iteration for maxiter
    
    #1. take the "marglocfdr" & "vec_h" , enumerate it as a dictionary 
    marglocfdr_d = [dict(enumerate(locfdr_[i])) for i in range(len(locfdr_))]
    vec_h_d = [dict(enumerate(vec_h_[i])) for i in range(len(vec_h_))]
    
    #2. sort the locfdr (values), so that their index (keys) is shuffeled
    omarglocfdr_d = [dict(sorted(marglocfdr_d[i].items(), key=lambda x: x[1])) for i in range(len(marglocfdr_d))]

    #3. take only the "num_rejections" first SHUFFELED indexes from "omarglocfdr_d"
    first_indexes = [list(omarglocfdr_d[i].keys())[:num_rejections_list[i]] for i in range(len(omarglocfdr_d))]

    #4. take only the "first_indexes" indexes from the enumerated vec_h dictionary, the vector of their values: "values_h"
    final_h = [dict([(k, vec_h_d[i][k]) for k in first_indexes[i]]) for i in range(len(vec_h_d))]

    #5. V = num_rejections - sum(values_h)
    V = [num_rejections_list[i] - sum(final_h[i].values()) for i in range(len(final_h))]

    return V

def FNR(num_hypo, prob_to_1, R_mean, V_mean):
    
    # calculating number of non-nulls + total number of hypotheses
    non_nulls = 0
    total_num_hypo = 0
    for i in range(len(num_hypo)):
        non_nulls += num_hypo[i]*prob_to_1[i]
        total_num_hypo += num_hypo[i]

    return (non_nulls - R_mean + V_mean) / (total_num_hypo - R_mean)


### 6.) S & R

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

def AandB_a_and_b_generator (locfdr, maxiter):
    
    K = locfdr.shape[1]
    locFDR = np.sort(locfdr, axis=1).reshape(locfdr.shape[0], locfdr.shape[1])
    
    clocFDR = np.cumsum(locFDR, axis=1)
    a = 1-locFDR

    b_1 = locFDR / np.array(range(1,K+1))

    zeros = np.full((maxiter,1),np.array([0]))
    clocFDR_b2 = np.delete(clocFDR,-1,1)
    b_2 = np.append(zeros, clocFDR_b2 / np.array(range(1,K)) / np.array(range(2,K+1)), axis=1)

    b = b_1 - b_2

    return a, b, locFDR

def fdep_ORACLE(musFDR, musPFDR, musMFDR, olocfdr_packed, maxiter, alpha):

    # "columns order: OMTFDR, OMTpFDR, OMTmFDR ... "
    minprob_mat = np.zeros((1,3),float)
    lev_mat = np.zeros((1,3),float)
    pow_mat = np.zeros((1,3),float)
    ev_mat = np.zeros((1,3),float)
    er_mat = np.zeros((1,3),float)  
    
    # - - - for oracle rule - - -
    a, b, locFDR = AandB_a_and_b_generator (olocfdr_packed, maxiter)
    
    for iter in range(maxiter): 

        # OMTFDR 
        lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = FDR_Generic_structure (mus=musFDR, 
                                                                               a=a[iter], 
                                                                               b_1=b[iter], 
                                                                               b_2=b[iter], 
                                                                               ind=0, 
                                                                               lev_mat=lev_mat, 
                                                                               pow_mat=pow_mat, 
                                                                               minprob_mat=minprob_mat, 
                                                                               ev_mat=ev_mat, 
                                                                               er_mat=er_mat, 
                                                                               olocfdr_function=locFDR[iter],  
                                                                               alpha = alpha)      
         # OMTpFDR 
        lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = FDR_Generic_structure (mus=musPFDR, 
                                                                               a=a[iter], 
                                                                               b_1=b[iter], 
                                                                               b_2=b[iter], 
                                                                               ind=1, 
                                                                               lev_mat=lev_mat, 
                                                                               pow_mat=pow_mat, 
                                                                               minprob_mat=minprob_mat, 
                                                                               ev_mat=ev_mat, 
                                                                               er_mat=er_mat, 
                                                                               olocfdr_function=locFDR[iter],  
                                                                               alpha = alpha)    
        # OMTMFDR 
        lev_mat, pow_mat, minprob_mat, ev_mat, er_mat = FDR_Generic_structure (mus=musMFDR, 
                                                                               a=a[iter-1], 
                                                                               b_1=locFDR[iter] - alpha, 
                                                                               b_2=b[iter], 
                                                                               ind=2, 
                                                                               lev_mat=lev_mat, 
                                                                               pow_mat=pow_mat, 
                                                                               minprob_mat=minprob_mat, 
                                                                               ev_mat=ev_mat, 
                                                                               er_mat=er_mat, 
                                                                               olocfdr_function=locFDR[iter],  
                                                                               alpha = alpha)                  

    return lev_mat/maxiter, pow_mat/maxiter, minprob_mat/maxiter, ev_mat/maxiter, er_mat/maxiter


### 7.) MOTHER FUNCTIONS (one without db, another with db)

def the_general_mother(musFDR, musPFDR, musMFDR, alpha, mu0, variance_0, mu1, variance_1, number_of_blocks, number_block_members, Rho, prob_to_1, maxiter):
    
    # initialize
    num_hypo = list(np.array(number_of_blocks) * np.array(number_block_members))
     
    ############################################################ 1.) Generating locfdr & vec_h per group, as a List
    locfdr_unpacked, vec_h_unpacked = locfdr_generator_all_groups(mu0, variance_0, mu1, variance_1, number_of_blocks, number_block_members, Rho, prob_to_1, maxiter)
    
    ############################################################ 2.) SLFDR 
    SLFDR_R = []
    SLFDR_V = []
    for group in range(len(locfdr_unpacked)):
        SLFDR_olocfdr = np.sort(locfdr_unpacked[group], axis=1)
        num_rejections, rejections_olocfdr = SLFDR_decision_vectorized_rule(SLFDR_olocfdr, alpha)
        vec_h_per_group = np.array([np.concatenate(vec_h_unpacked[group][j]) for j in range(len(vec_h_unpacked[group]))])
        v = V_generator(locfdr_unpacked[group], vec_h_per_group, num_rejections)
        SLFDR_R.append(num_rejections)
        SLFDR_V.append(v)
    SLFDR_MINPROB = np.where(sum((np.array(SLFDR_R) > 0) + 0) > 1, 1, 0)        
    
    # 2.1.) Calculating the procedure's metrics
    SLFDR_POWER = np.mean(np.sum(SLFDR_R, axis = 0) - np.sum(SLFDR_V, axis = 0))
    a = np.sum(SLFDR_V, axis = 0)
    b = np.sum(SLFDR_R, axis = 0)
    SLFDR_FDR_DIVIDE = np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b!=0)
    SLFDR_FDR = np.mean(SLFDR_FDR_DIVIDE) 
    if np.mean(np.sum(SLFDR_R, axis = 0)) != 0:
        SLFDR_MFDR = np.mean(np.sum(SLFDR_V, axis = 0)) / np.mean(np.sum(SLFDR_R, axis = 0))
    else:
        SLFDR_MFDR = 0  
    if np.mean(SLFDR_MINPROB) != 0:
        SLFDR_PFDR = SLFDR_FDR / np.mean(SLFDR_MINPROB)
    else:
        SLFDR_PFDR = 0
    SLFDR_FNR = FNR(num_hypo, prob_to_1, np.mean(np.sum(SLFDR_R, axis = 0)), np.mean(np.sum(SLFDR_V, axis = 0)))
    
    slfdr_list = [SLFDR_POWER, SLFDR_FNR, SLFDR_FDR, SLFDR_MFDR, SLFDR_PFDR]
    ############################################################ 3.) packing locfdr & vec_h together for the rest 4 procedures 
    locfdr_packed = locfdr_concator(locfdr_unpacked)
    vec_h_packed = vec_h_concator(vec_h_unpacked)
    olocfdr_packed = np.sort(locfdr_packed, axis=1)
    
    ############################################################ 4.) CLFDR
    CLFDR_R, rejections_olocfdr_list = SLFDR_decision_vectorized_rule(olocfdr_packed, alpha)
    CLFDR_V = V_generator(locfdr_packed, vec_h_packed, CLFDR_R)
    CLFDR_MINPROB = (np.array(CLFDR_R) > 0) + 0
    
    # 4.1.) Calculating the procedure's metrics
    CLFDR_POWER = np.mean(np.array(CLFDR_R) - np.array(CLFDR_V))
    aa = np.array(CLFDR_V)
    bb = np.array(CLFDR_R)
    CLFDR_FDR_DIVIDE = np.divide(aa, bb, out=np.zeros(aa.shape, dtype=float), where=bb!=0)
    CLFDR_FDR = np.mean(CLFDR_FDR_DIVIDE)
    if np.mean(CLFDR_R) != 0:
        CLFDR_MFDR = np.mean(CLFDR_V) / np.mean(CLFDR_R)
    else:
        CLFDR_MFDR = 0
    if np.mean(CLFDR_MINPROB) != 0:
        CLFDR_PFDR = CLFDR_FDR / np.mean(CLFDR_MINPROB)
    else:
        CLFDR_PFDR = 0
    CLFDR_FNR = FNR(num_hypo, prob_to_1, np.mean(CLFDR_R), np.mean(CLFDR_V))
    
    clfdr_list = [CLFDR_POWER, CLFDR_FNR, CLFDR_FDR, CLFDR_MFDR, CLFDR_PFDR]
    
    ############################################################ 5.) R&S
    lev_mat_r, pow_mat_r, minprob_mat_r, ev_mat_r, er_mat_r = fdep_ORACLE(musFDR, musPFDR, musMFDR, olocfdr_packed, maxiter, alpha)
    
    # 5.1.) OMTFDR control
    OMTFDR_power = pow_mat_r[0][0]
    OMTFDR_fdr = lev_mat_r[0][0]
    OMTFDR_mfdr = ev_mat_r[0][0] / er_mat_r[0][0]
    OMTFDR_pfdr = lev_mat_r[0][0] / minprob_mat_r[0][0]
    OMTFDR_fnr = FNR(num_hypo, prob_to_1, er_mat_r[0][0], ev_mat_r[0][0]) 
    
    OMTFDR_list = [OMTFDR_power, OMTFDR_fnr, OMTFDR_fdr, OMTFDR_mfdr, OMTFDR_pfdr]
    
    # 5.2.) OMTpFDR control
    OMTpFDR_power = pow_mat_r[0][1]
    OMTpFDR_fdr = lev_mat_r[0][1]
    OMTpFDR_mfdr = ev_mat_r[0][1] / er_mat_r[0][1]
    OMTpFDR_pfdr = lev_mat_r[0][1] / minprob_mat_r[0][1]
    OMTpFDR_fnr = FNR(num_hypo, prob_to_1, er_mat_r[0][1], ev_mat_r[0][1]) 
    
    OMTpFDR_list = [OMTpFDR_power, OMTpFDR_fnr, OMTpFDR_fdr, OMTpFDR_mfdr, OMTpFDR_pfdr]
    
    # 5.3.) OMTmFDR control
    OMTmFDR_power = pow_mat_r[0][2]
    OMTmFDR_fdr = lev_mat_r[0][2]
    OMTmFDR_mfdr = ev_mat_r[0][2] / er_mat_r[0][2]
    OMTmFDR_pfdr = lev_mat_r[0][2] / minprob_mat_r[0][2]
    OMTmFDR_fnr = FNR(num_hypo, prob_to_1, er_mat_r[0][2], ev_mat_r[0][2]) 
    
    OMTmFDR_list = [OMTmFDR_power, OMTmFDR_fnr, OMTmFDR_fdr, OMTmFDR_mfdr, OMTmFDR_pfdr]

    return slfdr_list, clfdr_list, OMTFDR_list, OMTpFDR_list, OMTmFDR_list


def the_general_mother_database(locfdr_unpacked, vec_h_unpacked, musFDR, musPFDR, musMFDR, alpha, mu0, variance_0, mu1, variance_1, number_of_blocks, number_block_members, Rho, prob_to_1, maxiter):
    
    # initialize
    num_hypo = list(np.array(number_of_blocks) * np.array(number_block_members))
     
    ############################################################ 1.) Generating locfdr & vec_h per group, as a List
    # no longer needed because of db :)
    
    ############################################################ 2.) SLFDR 
    SLFDR_R = []
    SLFDR_V = []
    for group in range(len(locfdr_unpacked)):
        SLFDR_olocfdr = np.sort(locfdr_unpacked[group], axis=1)
        num_rejections, rejections_olocfdr = SLFDR_decision_vectorized_rule(SLFDR_olocfdr, alpha)
        vec_h_per_group = np.array([np.concatenate(vec_h_unpacked[group][j]) for j in range(len(vec_h_unpacked[group]))])
        v = V_generator(locfdr_unpacked[group], vec_h_per_group, num_rejections)
        SLFDR_R.append(num_rejections)
        SLFDR_V.append(v)
    SLFDR_MINPROB = np.where(sum((np.array(SLFDR_R) > 0) + 0) > 1, 1, 0)        
    
    # 2.1.) Calculating the procedure's metrics
    SLFDR_POWER = np.mean(np.sum(SLFDR_R, axis = 0) - np.sum(SLFDR_V, axis = 0))
    a = np.sum(SLFDR_V, axis = 0)
    b = np.sum(SLFDR_R, axis = 0)
    SLFDR_FDR_DIVIDE = np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b!=0)
    SLFDR_FDR = np.mean(SLFDR_FDR_DIVIDE) 
    if np.mean(np.sum(SLFDR_R, axis = 0)) != 0:
        SLFDR_MFDR = np.mean(np.sum(SLFDR_V, axis = 0)) / np.mean(np.sum(SLFDR_R, axis = 0))
    else:
        SLFDR_MFDR = 0  
    if np.mean(SLFDR_MINPROB) != 0:
        SLFDR_PFDR = SLFDR_FDR / np.mean(SLFDR_MINPROB)
    else:
        SLFDR_PFDR = 0
    SLFDR_FNR = FNR(num_hypo, prob_to_1, np.mean(np.sum(SLFDR_R, axis = 0)), np.mean(np.sum(SLFDR_V, axis = 0)))
    
    slfdr_list = [SLFDR_POWER, SLFDR_FNR, SLFDR_FDR, SLFDR_MFDR, SLFDR_PFDR]
    ############################################################ 3.) packing locfdr & vec_h together for the rest 4 procedures 
    locfdr_packed = locfdr_concator(locfdr_unpacked)
    vec_h_packed = vec_h_concator(vec_h_unpacked)
    olocfdr_packed = np.sort(locfdr_packed, axis=1)
    
    ############################################################ 4.) CLFDR
    CLFDR_R, rejections_olocfdr_list = SLFDR_decision_vectorized_rule(olocfdr_packed, alpha)
    CLFDR_V = V_generator(locfdr_packed, vec_h_packed, CLFDR_R)
    CLFDR_MINPROB = (np.array(CLFDR_R) > 0) + 0
    
    # 4.1.) Calculating the procedure's metrics
    CLFDR_POWER = np.mean(np.array(CLFDR_R) - np.array(CLFDR_V))
    aa = np.array(CLFDR_V)
    bb = np.array(CLFDR_R)
    CLFDR_FDR_DIVIDE = np.divide(aa, bb, out=np.zeros(aa.shape, dtype=float), where=bb!=0)
    CLFDR_FDR = np.mean(CLFDR_FDR_DIVIDE)
    if np.mean(CLFDR_R) != 0:
        CLFDR_MFDR = np.mean(CLFDR_V) / np.mean(CLFDR_R)
    else:
        CLFDR_MFDR = 0
    if np.mean(CLFDR_MINPROB) != 0:
        CLFDR_PFDR = CLFDR_FDR / np.mean(CLFDR_MINPROB)
    else:
        CLFDR_PFDR = 0
    CLFDR_FNR = FNR(num_hypo, prob_to_1, np.mean(CLFDR_R), np.mean(CLFDR_V))
    
    clfdr_list = [CLFDR_POWER, CLFDR_FNR, CLFDR_FDR, CLFDR_MFDR, CLFDR_PFDR]
    
    ############################################################ 5.) R&S
    lev_mat_r, pow_mat_r, minprob_mat_r, ev_mat_r, er_mat_r = fdep_ORACLE(musFDR, musPFDR, musMFDR, olocfdr_packed, maxiter, alpha)
    
    # 5.1.) OMTFDR control
    OMTFDR_power = pow_mat_r[0][0]
    OMTFDR_fdr = lev_mat_r[0][0]
    OMTFDR_mfdr = ev_mat_r[0][0] / er_mat_r[0][0]
    OMTFDR_pfdr = lev_mat_r[0][0] / minprob_mat_r[0][0]
    OMTFDR_fnr = FNR(num_hypo, prob_to_1, er_mat_r[0][0], ev_mat_r[0][0]) 
    
    OMTFDR_list = [OMTFDR_power, OMTFDR_fnr, OMTFDR_fdr, OMTFDR_mfdr, OMTFDR_pfdr]
    
    # 5.2.) OMTpFDR control
    OMTpFDR_power = pow_mat_r[0][1]
    OMTpFDR_fdr = lev_mat_r[0][1]
    OMTpFDR_mfdr = ev_mat_r[0][1] / er_mat_r[0][1]
    OMTpFDR_pfdr = lev_mat_r[0][1] / minprob_mat_r[0][1]
    OMTpFDR_fnr = FNR(num_hypo, prob_to_1, er_mat_r[0][1], ev_mat_r[0][1]) 
    
    OMTpFDR_list = [OMTpFDR_power, OMTpFDR_fnr, OMTpFDR_fdr, OMTpFDR_mfdr, OMTpFDR_pfdr]
    
    # 5.3.) OMTmFDR control
    OMTmFDR_power = pow_mat_r[0][2]
    OMTmFDR_fdr = lev_mat_r[0][2]
    OMTmFDR_mfdr = ev_mat_r[0][2] / er_mat_r[0][2]
    OMTmFDR_pfdr = lev_mat_r[0][2] / minprob_mat_r[0][2]
    OMTmFDR_fnr = FNR(num_hypo, prob_to_1, er_mat_r[0][2], ev_mat_r[0][2]) 
    
    OMTmFDR_list = [OMTmFDR_power, OMTmFDR_fnr, OMTmFDR_fdr, OMTmFDR_mfdr, OMTmFDR_pfdr]

    return slfdr_list, clfdr_list, OMTFDR_list, OMTpFDR_list, OMTmFDR_list


### 7.) Visualizing

def plot_metric(metric_index, metric_tite_str, metric_realizations_x_axis, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR):
    # this function needs as an input a series of NESTED LISTS per policy procedure
    
    slfdr_metric_list = [policy_realization[metric_index] for policy_realization in slfdr]
    clfdr_metric_list = [policy_realization[metric_index] for policy_realization in clfdr]
    rs_OMTFDR_metric_list = [policy_realization[metric_index] for policy_realization in rs_OMTFDR]
    rs_OMTpFDR_metric_list = [policy_realization[metric_index] for policy_realization in rs_OMTpFDR]
    rs_OMTmFDR_metric_list = [policy_realization[metric_index] for policy_realization in rs_OMTmFDR]
        
    plt.plot(metric_realizations_x_axis, slfdr_metric_list , label = "slfdr")
    plt.plot(metric_realizations_x_axis, clfdr_metric_list , label = "clfdr")
    plt.plot(metric_realizations_x_axis, rs_OMTFDR_metric_list , label = "OMTFDR")
    plt.plot(metric_realizations_x_axis, rs_OMTpFDR_metric_list , label = "OMTpFDR")
    plt.plot(metric_realizations_x_axis, rs_OMTmFDR_metric_list , label = "OMTmFDR")


    plt.title(metric_tite_str)
    plt.legend()
    plt.show()

def plot_4_metrics(simulation, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR):
    # this function needs: the output of the 5 procedures + the X axis simulations change
    
    plot_metric(0, "Power - E(TP)", simulation, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR)
    plot_metric(1, "FNR", simulation, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR)
    plot_metric(2, "FDR", simulation, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR)
    plot_metric(3, "mFDR", simulation, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR)
    plot_metric(4, "pFDR", simulation, slfdr, clfdr, rs_OMTFDR, rs_OMTpFDR, rs_OMTmFDR)