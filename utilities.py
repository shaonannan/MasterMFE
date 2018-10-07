"""
Master Equation to solve all these problem
"""

import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import scipy.integrate as spi
import bisect
from scipy.optimize import minimize
from scipy import special
import time


def fraction_overlap(a1, a2, b1, b2):
    """
    Calculate the fractional overlap between range (a1,a2) and (b1,b2).
    
    Used to compute a reallocation of probability mass from one set of bins to
    another, assuming linear interpolation.
    """
    if a1 >= b1:    # range of A starts after B starts
        if a2 <= b2:    
            return 1       # A is within B
        if a1 >= b2:
            return 0       # A is after B
        # overlap is from a1 to b2
        return (b2 - a1) / (a2 - a1)
    else:            # start of A is before start of B
        if a2 <= b1:
            return 0       # A is completely before B
        if a2 >= b2:
            # B is subsumed in A, but fraction relative to |A|
            return (b2 - b1) / (a2 - a1)
        # overlap is from b1 to a2
        return (a2 - b1) / (a2 - a1) 
    

def redistribute_probability_mass(A, B):
    '''Takes two 'edge' vectors and returns a 2D matrix mapping each 'bin' in B
    to overlapping bins in A. Assumes that A and B contain monotonically increasing edge values.
    '''
    
    mapping = np.zeros((len(A)-1, len(B)-1))
    newL = 0
    newR = newL
    
    # Matrix is mostly zeros -- concentrate on overlapping sections
    for L in range(len(A)-1):
        
        # Advance to the start of the overlap
        while newL < len(B) and B[newL] < A[L]:
            newL = newL + 1
        if newL > 0:
            newL = newL - 1
        newR = newL
        
        # Find end of overlap
        while newR < len(B) and B[newR] < A[L+1]:
            newR = newR + 1
        if newR >= len(B):
            newR = len(B) - 1

        # Calculate and store remapping weights
        for j in range(newL, newR):
            mapping[L][j] = fraction_overlap(A[L], A[L+1], B[j], B[j+1])

    return mapping

    
def flux_matrix(v, w, lam, p=1):
    'Compute a flux matrix for voltage bins v, weight w, firing rate lam, and probability p.'
    
    zero_bin_ind_list = get_zero_bin_list(v)
    
    # Flow back into zero bin:
    if w > 0:
        
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p

        # Threshold:
        flux_to_zero_vector = -A.sum(axis=0)
        for curr_zero_ind in zero_bin_ind_list:
            A[curr_zero_ind,:] += flux_to_zero_vector/len(zero_bin_ind_list)
    else:
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p
        
        
        missing_flux = -A.sum(axis=0)
        A[0,:] += missing_flux
        
        flux_to_zero_vector = np.zeros_like(A.sum(axis=0))

    return flux_to_zero_vector, A

def flux_matrix_with_NMDA(v, w, lam, p=1):
    'Compute a flux matrix for voltage bins v, weight w, firing rate lam, and probability p.'
    
    zero_bin_ind_list = get_zero_bin_list(v)
    
    # Flow back into zero bin:
    A = -np.eye(len(v)-1)*lam*p
    A += redistribute_probability_mass(v+w, v).T*lam*p
    ind_neg = np.arange(max(zero_bin_ind_list)+1,len(v)-1)
    neg     = np.squeeze(A[:,ind_neg])
    flux_to_zero_vector_neg = -neg.sum(axis=0)
    for curr_zero_ind in zero_bin_ind_list:
            A[curr_zero_ind,ind_neg] += flux_to_zero_vector_neg/len(zero_bin_ind_list)
    ind_pos = np.arange(0,min(zero_bin_ind_list))
    pos     = np.squeeze(A[:,ind_pos])
    missing_flux = -pos.sum(axis=0)
    A[0,ind_pos] += missing_flux
    
    # flux to zero total
    flux_to_zero_vector            = np.zeros_like(A.sum(axis=0))
    # print('neg: ',np.shape(flux_to_zero_vector_neg))
    # print('total: ',np.shape(flux_to_zero_vector))
    flux_to_zero_vector[ind_neg] =  flux_to_zero_vector_neg
    return  flux_to_zero_vector,A


def exact_update_method(J,rhov,dt = 1e-4):
    rhov = np.dot(spla.expm(J*dt),rhov)
    assert_probability_mass_conserved(rhov)
    return rhov


def approx_update_method_tol(J, pv, tol=2.2e-16, dt=.0001, norm='inf'):
    'Approximate the effect of a matrix exponential, with residual smaller than tol.'
    
    # No order specified:
    J *= dt
    curr_err = np.inf
    counter = 0.
    curr_del = pv
    pv_new = pv
    
    while curr_err > tol:
        counter += 1
        curr_del = J.dot(curr_del)/counter
        pv_new += curr_del
        curr_err = spla.norm(curr_del, norm)

    
#    try:
#        assert_probability_mass_conserved(pv)
#    except:                                                                                                                                                     # pragma: no cover
#        raise Exception("Probabiltiy mass error (p_sum=%s) at tol=%s; consider higher order, decrease dt, or increase dv" % (np.abs(pv).sum(), tol))            # pragma: no cover
#    
    return pv_new

def approx_update_method_order(J, pv, dt=.0001, approx_order=2):
    'Approximate the effect of a matrix exponential, truncating Taylor series at order \'approx_order\'.'
    
    # Iterate to a specific order:
    coeff = 1.
    curr_del = pv
    pv_new = pv
    for curr_order in range(approx_order):
        coeff *= curr_order+1
        curr_del = J.dot(curr_del*dt)
        pv_new += (1./coeff)*curr_del
    
#    try:
#        assert_probability_mass_conserved(pv_new)
#    except:                                                                                                                                                             # pragma: no cover
#        raise Exception("Probabiltiy mass error (p_sum=%s) at approx_order=%s; consider higher order, decrease dt, or increase dv" % (np.abs(pv).sum(), approx_order))  # pragma: no cover

    return pv_new
def approx_update_kn(leak_input,short_range_input, pv,tol=2.2e-16,  dt=.0001,norm='norm'):    
    # initiation
    curr_del = pv
    counter  = 0
    curr_err = np.inf
    pv_new   = pv
    leak_curr = 0

    for key,val in short_range_input.items():
        key.J_exp = 0.0
    # approximate
    while curr_err > tol:
        counter += 1
        curr_del_new = np.zeros_like(curr_del)
        if counter <2: # start
            leak_curr = leak_input.copy()
            leak_curr = np.dot(leak_curr*dt,pv)
            curr_del_new += leak_curr
            for key,val in short_range_input.items():
                key.J_exp = np.dot(key.flux_matrix * dt * val,pv) / counter
                curr_del_new += key.J_exp
        else:
            leak_curr = np.dot(leak_input*dt,leak_curr) / counter
            curr_del_new += leak_curr
            for key,val in short_range_input.items():
                key.J_exp = np.dot(key.flux_matrix * dt,key.J_exp) / counter
                curr_del_new += key.J_exp
        curr_del = curr_del_new
        pv_new  += curr_del
        #print('curr_del: ',curr_del)
        curr_err = spla.norm(curr_del,norm)
    try:
        assert_probability_mass_conserved(pv)
    except:                                                                                                                                                     # pragma: no cover
        raise Exception("Probabiltiy mass error (p_sum=%s) at tol=%s; consider higher order, decrease dt, or increase dv" % (np.abs(pv).sum(), tol))            # pragma: no cover
    
    return pv_new


def get_v_edges(v_min,v_max,dv):
    # Used for voltage-distribution and discretization
    edges = np.concatenate((np.arange(v_min,v_max,dv),[v_max]))
    edges[np.abs(edges) < np.finfo(np.float).eps] = 0
    return edges
def get_zero_bin_list(v):
    # find low boundary for vreset
    v = np.array(v) # cast to avoid mistake
    
    if(len(np.where(v==0)[0])>0):
        zero_edge_ind = np.where(v==0)[0][0]
        if zero_edge_ind == 0:
            return [0]
        else:
            return [zero_edge_ind-1,zero_edge_ind]
    else:
        return [bisect.bisect_right(v,0)-1]
    
def leak_matrix(v,tau): # original version (v,tau)
    # knew voltage-edge and design leaky-integrate-and-fire mode

    zero_bin_ind_list = get_zero_bin_list(v)   
    # initialize A transmit function for leaky integrate and fire
    A = np.zeros((len(v)-1,len(v)-1))
    
    # if dv/dt = right , right --> positive leak:
    delta_w_ind = -1
    for pre_ind in np.arange(max(zero_bin_ind_list)+1,len(v)-1):
        post_ind = pre_ind + delta_w_ind
        dv = v[pre_ind+1]-v[pre_ind]
        bump_rate = v[pre_ind+1]/(tau*dv)
        A[pre_ind,pre_ind] -= bump_rate
        A[post_ind,pre_ind] += bump_rate
        
    # if dv/dt = right, right --> negative leak:
    delta_w_ind = 1
    for pre_ind in np.arange(0,min(zero_bin_ind_list)):
        post_ind = pre_ind + delta_w_ind
        dv = v[pre_ind]-v[pre_ind+1]
        # reverse
        bump_rate = v[pre_ind]/(tau*dv)  # always choose the smaller one
        A[pre_ind,pre_ind] -= bump_rate
        A[post_ind,pre_ind] += bump_rate
        
    return A

def leak_matrix_with_NMDA(v,v_min_NMDA,tau):
    zero_bin_ind_list = get_zero_bin_list(v_min_NMDA)
    A = np.zeros((len(v)-1,len(v)-1))
    delta_w_ind = -1
    for pre_ind in np.arange(max(zero_bin_ind_list)+1,len(v)-1):
        post_ind = pre_ind + delta_w_ind
        dv = v[pre_ind+1]-v[pre_ind]
        # v_depend = v[pre_ind+1] - Inmda * tau
        v_depend  = v_min_NMDA[pre_ind]
        # bump_rate = v[pre_ind+1]/(tau*dv)
        bump_rate  = v_depend / (tau*dv)
        A[pre_ind,pre_ind] -= bump_rate
        A[post_ind,pre_ind] += bump_rate
        
    # if dv/dt = right, right --> negative leak:
    delta_w_ind = 1
    for pre_ind in np.arange(0,min(zero_bin_ind_list)):
        post_ind = pre_ind + delta_w_ind
        dv = v[pre_ind]-v[pre_ind+1]
        # reverse
        # v_depend = [v_pre_ind] - Inmda * tau
        v_depend  = v_min_NMDA[pre_ind]
        # bump_rate = v[pre_ind]/(tau*dv)  # always choose the smaller one
        bump_rate = v_depend / (tau * dv)
        A[pre_ind,pre_ind] -= bump_rate
        A[post_ind,pre_ind] += bump_rate
        
    return A

def assert_probability_mass_conserved(pv):
    'Assert that probability mass in control nodes sums to 1.'
    
    try:
        assert np.abs(np.abs(pv).sum() - 1) < 1e-12
    except:                                                                                 # pragma: no cover
        raise Exception('Probability mass below threshold: %s' % (np.abs(pv).sum() - 1))    # pragma: no cover

