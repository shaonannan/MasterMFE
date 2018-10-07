"""
Module to describe  dynamics of internal excitatory/ inhibitory population
There exist 2 different approachs (methods) to quantify the effect of long-range NMDA type synaptic inputs
on subthreshold voltage
1_st Algorithm looks much similar to that of processing short-range synaptic input, function called flux_matrix
2_nd Algorithm looks more like that of processing leakage voltage, function called leak_matrix

Last modified version 2018/04/07
"""

import numpy as np
import utilities as util
from functools import reduce
'''
To tell the difference between 1st and 2nd approach!
For the 1st approach, the algorithm is much more similar to that in processing short-range synaptic input, which we could 
extract 2 matrixs from the result, the first represents threshold flux due to long-range synaptic inputs, and the second
item represents transmission(flux) matrix of subthreshold voltage distribution

And, for the 2nd approach, the algorithm processes long-range synaptic inputs the same way as that for leakage current, thus in
this algorithm, flux through threshold is neglected, large long-range synaptic inputs may trigger exception and error! 
'''

class RecurrentPopulation(object):
    """
    Parameters:
    tau_m: time constant for membrane potential
    v_min: minimum voltage(default = -1.0)
    v_th/max: maximum/threshold 
    dv   : voltage domain discritization 
    record: flag(True/False)
    curr_firing_rate: firing rate of the corresponding recurrent population
    update_mode: str'approx' or 'exact'(default=')
    
    """
    
    def __init__(self,tau_m = 20.0,dt = 1e-1,v_min = -1.0,v_max = 1.0,dv = 1e-3,record = True,
                firing_rate = 0.0,update_method = 'exact',approx_order = None,tol = 1e-12,celltype = 'e',norm = np.inf,**kwargs):
        # transmit parameters
        self.tau_m = tau_m
        self.dt    = dt
        (self.v_min,self.v_max) = (v_min,v_max)
        self.dv = dv
        self.record        = record
        self.firing_rate   = 0.0 # firing_rate
        self.update_method = update_method
        self.approx_order  = approx_order
        self.tol   = tol
        self.norm  = norm
        self.celltype = celltype
        self.type  = 'Recurrent'
        # additional parameters
        self.metadata = kwargs
        
        # before real initialization, voltage-edge and voltage-distribution
        # are all None, these setting should be initialized later by specific command
        
        self.edges = None
        self.rhov  = None
        self.firing_rate_record = None # used for recording corresponding spike train
        self.t_record           = None # time series
        # This item is dynamical variable -- NMDA synaptic input
        self.leak_flux_matrix   = None
        
        self.threshold_flux_vector_self = None

        # simulation in identical platform
        self.simulation         = None

        # if we use active release NMDA synaptic signal
        # once the sender(pre) population generated firing rate
        # it had ability to automatically release NMDA-type slow conductance
        # so it naturally has this property(without self.weights)
        self.hnmda = 0.0
        self.inmda = 0.0

        # for long-range connections
        self.tau_r = 2.0
        self.tau_d = 128.0

        # for approximation 
        self.J_exp = 0.0
        # MFE
        self.MFE_firing_rate = 0
        self.MFEflag = 0
        self.NE,self.NI = 100,100
        self.potential_MFE = 0.0
        self.total_longrange_NMDA = 0.0
        
        
    def initialize(self):
        """
        initialize some specific parameters and variables by hand
        with 
            1)voltage-edge/bin
            2)connection dictionary
            3)all about recorder
        """
        self.initialize_edges()
        self.initialize_prob()
        self.initialize_total_input_dict()
        
    """
    Code below is designed for some basic matrix or elements which might be initialized
    at the beginning and maintained unchanged during the whole data analysis, but if U 
    need update some connections or strurtures, U could still start the 'Update' function 
    to regenerate a new structure(connections)
    """   
    def initialize_edges(self):
        # initialize discreted voltage bins
        self.edges = util.get_v_edges(self.v_min,self.v_max,self.dv)
        leak_flux_matrix = util.leak_matrix(self.edges,self.tau_m)
        self.leak_flux_matrix = leak_flux_matrix
        # this should be dynamically changed!



    def initialize_prob(self):
        # initialize voltage-distribution
        self.rhov = np.zeros_like(self.edges[:-1])
        zero_bin_list = util.get_zero_bin_list(self.edges)
        for ii in zero_bin_list:
            self.rhov[ii] = 1./len(zero_bin_list)



    # also combine short-range and long-range input
    # short-range input relates to real firing rate of pre-population
    # while long-range input relates to slow-changed NMDA-type synaptic input
    # both have no relationship with weights!
    def initialize_total_input_dict(self):
        
        # Feedforward only contributes to short-range inputs
        self.total_inputsr_dict = {}
        for c in self.source_connection_list:
            if (c.conn_type == 'ShortRangeFF'):
                try:
                    curr_input = self.total_inputsr_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    curr_input = self.total_inputsr_dict.setdefault(c.connection_distribution,0)
                self.total_inputsr_dict[c.connection_distribution] = curr_input + c.curr_firing_rate * c.nsyn
        
        # Recurrent could contribute to short-range inputs as well as MFE-event
        self.total_MFEsr_dict   = {}
        for c in self.source_connection_list:
            if(c.conn_type == 'ShortRangeRC'):
                # if already initialize connection_distribution or not
                try:
                    curr_input = self.total_inputsr_dict.setdefault(c.connection_distribution,0)
                    MFE_input  = self.total_MFEsr_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    curr_input = self.total_inputsr_dict.setdefault(c.connection_distribution,0)
                    MFE_input = self.total_MFEsr_dict.setdefault(c.connection_distribution,0)
                self.total_inputsr_dict[c.connection_distribution] = curr_input + c.curr_firing_rate * c.nsyn
                self.total_MFEsr_dict[c.connection_distribution]   = MFE_input + c.curr_firing_rate * c.nsyn
                
        self.total_inputlr_dict = {}
        for c in self.source_connection_list:
            if(c.conn_type == 'LongRange'):
                # if already initialize connection_distribution or not
                try:
                    curr_input = self.total_inputlr_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    curr_input = self.total_inputlr_dict.setdefault(c.connection_distribution,0)
                self.total_inputlr_dict[c.connection_distribution] = curr_input +  c.nsyn * c.curr_Inmda

            
    def update_total_flux_matrix(self):
        """
        long-range connections lift base-voltage, but didn't generate flux!!!
        """
        # Then, the NMDA-type input contributes to leak_flux_matrix_with_NMDA
        # at first, we should calculate total long-range input

        """
        1_st Algorithm to calculate the effect of long-range synaptic inputs onto the dynamics of voltage evolution
        """
        
        # This approach uses flux matrix algorithm like that in short range synaptic input
        self.total_longrange_NMDA = 0.0
        for key,val in self.total_inputlr_dict.items():
            self.total_longrange_NMDA += key.weights * val
        
#        self.leak_flux_matrix = util.leak_matrix(v = self.edges,tau = self.tau_m)
        total_flux_matrix = self.leak_flux_matrix.copy()
        # short range
        for key,val in self.total_inputsr_dict.items():
            try:
                total_flux_matrix += key.flux_matrix * val
            except:
                key.initialize()
                total_flux_matrix += key.flux_matrix * val
                
        # long range
        for key,val in self.total_inputlr_dict.items():
            try:
                total_flux_matrix += key.flux_matrix * val
            except:
                key.initialize()
                total_flux_matrix += key.flux_matrix * val

        return total_flux_matrix


    
    
    # short range and long rage input
    def update_total_input_dict(self):
        # for curr_CD in self.source_connection_list:
        # have already exist
        # short range input
        for curr_CD in self.total_inputsr_dict.keys():
            self.total_inputsr_dict[curr_CD] = 0.0
        # ONLY FF SHORTRANGE -- INPUTSR
        for c in self.source_connection_list:
            if ( c.conn_type =='ShortRangeFF'):
                self.total_inputsr_dict[c.connection_distribution] += c.pre_population.curr_firing_rate * c.nsyn # c.curr_firing_rate * c.nsyn # 
#                print('>>>>pre:',c.pre_population.curr_firing_rate,'; c:',c.curr_firing_rate)
        # FFSHORTRANGE AND MFE
        for c in self.source_connection_list:
            if ( c.conn_type =='ShortRangeRC'):
                self.total_inputsr_dict[c.connection_distribution] += c.pre_population.curr_firing_rate * c.nsyn # c.curr_firing_rate * c.nsyn # 
#                print('>>>>pre:',c.pre_population.curr_firing_rate,'; c:',c.curr_firing_rate)

        # for long range connections, the input stimuli will change step-by-step as well(base on firing rate of pre-population)
        # but will not immediately change Inmda (as well as hNMDA), because this function only care about total INPUT so, could reset and 
        # refresh step-by-step
        for curr_CD in self.total_inputlr_dict.keys():
            self.total_inputlr_dict[curr_CD] = 0.0
        for c in self.source_connection_list:
            if (c.conn_type == 'LongRange'):
                self.total_inputlr_dict[c.connection_distribution] +=  c.pre_population.curr_Inmda * c.nsyn #  c.nsyn * c.curr_Inmda #

    def update_prob(self):
        J = self.update_total_flux_matrix()
        self.rhov = self.rhov/np.sum(self.rhov)
#        print('>>>>>>>sum rhov: ;',np.sum(self.rhov))
        if self.update_method == 'exact':
            self.rhov = util.exact_update_method(J,self.rhov,self.simulation.dt) 
            self.rhov = self.rhov/np.sum(self.rhov)
        elif self.update_method == 'approx_kn':
            self.rhov = util.approx_update_kn(self.leak_flux_matrix,self.total_inputsr_dict,self.rhov,tol=self.tol, dt=self.simulation.dt, norm=self.norm)       
            self.rhov = self.rhov/np.sum(self.rhov)
        elif self.update_method == 'approx':

            if self.approx_order == None:
                self.rhov = util.approx_update_method_tol(J, self.rhov, tol=self.tol, dt=self.simulation.dt, norm=self.norm)
                self.rhov = self.rhov/np.sum(self.rhov)

            else:
                self.rhov = util.approx_update_method_order(J, self.rhov, approx_order=self.approx_order, dt=self.simulation.dt)
                self.rhov = self.rhov/np.sum(self.rhov)
        
        else:
            raise Exception('Unrecognized population update method: "%s"' % self.update_method)  # pragma: no cover

    def update_firing_rate(self):
        flux_vector = reduce(np.add,[key.threshold_flux_vector * val for key,val in
                                    self.total_inputsr_dict.items()])   
        flux_vector += reduce(np.add,[key.threshold_flux_vector * val for key,val in
                                    self.total_inputlr_dict.items()])          
        """   
        """
        self.firing_rate = np.dot(flux_vector,self.rhov)
        # print 'firing rate: ',self.firing_rate

    # update own hNMDA and iNMDA, which only depends on curr_firing_rate 
    # in another words, a-subpopulation's hNMDA & iNMDA only depend on itself
    def update_NMDA_midvar_syncurr(self):
        ownfr = self.curr_firing_rate
        # parameters
        deltat = self.dt
        trise  = self.tau_r
        tdamp  = self.tau_d

        tr   = deltat/trise
        etr  = np.exp(-tr)
        td   = deltat/tdamp
        etd  = np.exp(-td)
        cst  = 1.0/(tdamp - trise)*(etd - etr) # trise/(tdamp - trise)*(etd - etr)

        self.inmda = self.inmda * etd + self.hnmda * cst
        self.hnmda = self.hnmda * etr + ownfr #* self.simulation.dt
        # print 'release NMDA: ',self.inmda
        # print 'ownfr:  ',ownfr
    
    def update_MFE(self):
        # THE 1ST STEP, TODO SELF FLUX AND CHECK THE MFE
        for curr_CD in self.total_MFEsr_dict.keys():
            self.total_MFEsr_dict[curr_CD] = 0.0
        # ok, already be zero
        for c in self.source_connection_list:
            if ( c.conn_type =='ShortRangeRC'):
                pre_p = c.pre_population
                self.total_MFEsr_dict[c.connection_distribution] += c.curr_firing_rate * c.nsyn * (1-pre_p.MFEflag)+pre_p.MFEflag * 2.0/self.simulation.dt#pre_p.curr_firing_rate * c.nsyn * (1-pre_p.MFEflag)+pre_p.MFEflag * 2.0/self.simulation.dt
                print('>>>>>total_MFEsr:',self.total_MFEsr_dict[c.connection_distribution])
        # UPDATE JMAT AND TOZERO
        MFE_flux_matrix = np.zeros_like(self.leak_flux_matrix.copy())
        for key,val in self.total_MFEsr_dict.items():
            try:
                MFE_flux_matrix += key.flux_matrix * val
            except:
                key.initialize()
                MFE_flux_matrix += key.flux_matrix * val
#        print('MFE_flux_matrix:',MFE_flux_matrix)
        # CALCULATE TOZERO 
        MFE_flux_vector = reduce(np.add,[key.threshold_flux_vector * val for key,val in self.total_MFEsr_dict.items()])
        # UPDATE PROB
        if self.update_method == 'approx':
            self.rhov = util.approx_update_method_tol(MFE_flux_matrix,self.rhov,tol=self.tol,dt=self.simulation.dt,norm = self.norm)
#            print('self.rhov:',self.rhov)
        # UPDATE FIRING RATE
        self.firing_rate = np.dot(MFE_flux_vector,self.rhov)
        # when doing MFE, initializing MFE_firing_rate ==> LE/I here ~
        self.MFE_firing_rate = self.MFEflag * 2.0/self.simulation.dt
        if self.celltype == 'e':
            self.MFE_firing_rate += self.firing_rate * self.NE
        else:
            self.MFE_firing_rate += self.firing_rate * self.NI
        return self.MFE_firing_rate, self.firing_rate
     
    def update_iteration_MFE(self):
        # THE 1ST STEP, TODO SELF FLUX AND CHECK THE MFE
        for curr_CD in self.total_MFEsr_dict.keys():
            self.total_MFEsr_dict[curr_CD] = 0.0
        # ok, already be zero
        for c in self.source_connection_list:
            if ( c.conn_type =='ShortRangeRC'):
                pre_p = c.pre_population
                self.total_MFEsr_dict[c.connection_distribution] += c.curr_firing_rate * c.nsyn * (1-pre_p.MFEflag)# pre_p.curr_firing_rate * c.nsyn * (1-pre_p.MFEflag)#+pre_p.MFEflag * 2.0/self.simulation.dt
#                print('total_MFEsr:',self.total_MFEsr_dict[c.connection_distribution])
        # UPDATE JMAT AND TOZERO
        MFE_flux_matrix = np.zeros_like(self.leak_flux_matrix.copy())
        for key,val in self.total_MFEsr_dict.items():
            try:
                MFE_flux_matrix += key.flux_matrix * val
            except:
                key.initialize()
                MFE_flux_matrix += key.flux_matrix * val
#        print('MFE_flux_matrix:',MFE_flux_matrix)
        # CALCULATE TOZERO 
        MFE_flux_vector = reduce(np.add,[key.threshold_flux_vector * val for key,val in self.total_MFEsr_dict.items()])
        # UPDATE FIRING RATE
        self.firing_rate = np.dot(MFE_flux_vector,self.rhov)

        if self.celltype == 'e':
            self.MFE_firing_rate += self.firing_rate * self.NE
        else:
            self.MFE_firing_rate += self.firing_rate * self.NI
        # UPDATE PROB
        if self.update_method == 'approx':
            self.rhov = util.approx_update_method_tol(MFE_flux_matrix,self.rhov,tol=self.tol,dt=self.simulation.dt,norm = self.norm)
#            print('self.rhov:',self.rhov)

        return self.MFE_firing_rate,self.firing_rate

    def update_potential_MFE(self):
        # THE 1ST STEP, TODO SELF FLUX AND CHECK THE MFE
        for curr_CD in self.total_MFEsr_dict.keys():
            self.total_MFEsr_dict[curr_CD] = 0.0
        # ok, already be zero
        for c in self.source_connection_list:
            if ( c.conn_type =='ShortRangeRC'):
                pre_p = c.pre_population
                self.total_MFEsr_dict[c.connection_distribution] += c.curr_firing_rate * c.nsyn * (1-pre_p.MFEflag)# pre_p.curr_firing_rate * c.nsyn * (1-pre_p.MFEflag)#+pre_p.MFEflag * 1.0/self.simulation.dt
        # calculate S*m(E) - S*m(I)
        self.potential_MFE = 0.0
        for key,val in self.total_MFEsr_dict.items():
            try:
                self.potential_MFE += key.weights * val
            except:
                key.initialize()
                self.potential_MFE += key.weights * val
        return self.potential_MFE
    # updating function set
    def update(self):
        self.update_total_input_dict()
        self.update_NMDA_midvar_syncurr()
        self.update_total_flux_matrix()
        self.update_firing_rate()
        self.update_prob()   
              

    @property
    def source_connection_list(self):
        return [c for c in self.simulation.connection_list if c.post_population == self]
        
    @property
    def curr_firing_rate(self):
        return self.firing_rate

    @property
    def curr_Inmda(self):
        return self.inmda

        

        
