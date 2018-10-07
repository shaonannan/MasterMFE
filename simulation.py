"""
Last modified 201/04/07
"""

from connectiondistributioncollection import ConnectionDistributionCollection
import time
import numpy as np
import matplotlib.pyplot as plt
import utilities as util

from scipy import interpolate

class Simulation(object):
    """
    Parameters:
    list :
        All sub-population (cluster)
        All connection (cluster)
        [type of both is 'List', which is changable variable, and could be changed]
        
    generate after initiate(by hand)
        connection_distribution
        connection_distribution_list
        [the differences between connection, connection_distribution and connection_distribution_list are
        connection ---> the component of 'connection_list', record all information and related information and object,like source and others
        connection_distribution --> this variable is a preparation variable for further processing, each 'connection' could generate a 
        class 'connecton_distribution' and then, using weight,syn,prob, could calculate flux_matrix and threshold
        each 'connection_distribution' item is defined by 'weight''syn ''prob', items with identical symbol will be classified to the same
        distribution
        connection_distribution_list --> this is a 'basket', store all unique connections(definition of unique: unique symbol
        'weight','syn','prob' no matter the target/source population)
    """
    def __init__(self,population_list,connection_list,Net_settings,Cell_type_num,DEE,DIE,DEI,DII,verbose=True):
        
        self.verbose = verbose
        self.population_list = population_list
        self.connection_list = [c for c in connection_list if c.nsyn!=0.0]
        self.Net_settings    = Net_settings
        tfinal = Net_settings['Final_time']
        dt     = Net_settings['dt']
        self.NPATCH = Net_settings['nmax']
        self.ntt    = int(tfinal/dt)
        self.tau_m  = 20.
        
        # recording for  MFE
        # Each tiny step for testing
        self.rhovE = np.zeros((self.NPATCH,200))
        self.rhovI = np.zeros((self.NPATCH,200))
        
        self.INMDAE = np.zeros(self.NPATCH)
        self.INMDAI = np.zeros(self.NPATCH)
        
        self.HNMDAE = np.zeros(self.NPATCH)
        self.HNMDAI = np.zeros(self.NPATCH)
        
        self.newrhovE = np.zeros_like(self.rhovE)
        self.newrhovI = np.zeros_like(self.rhovI)
        
        self.MFE_frE = np.zeros(self.NPATCH)
        self.MFE_frI = np.zeros(self.NPATCH)
        
        self.FR_casE = np.zeros(self.NPATCH)
        self.FR_casI = np.zeros(self.NPATCH)
        
        self.epsMFE  = 1E-6
        
        # >>>>>>>>>>>>> For MFE
        self.Vedges, self.Vbins = None,None
        self.DEE,self.DIE,self.DEI,self.DII = DEE,DIE,DEI,DII
        self.NE,self.NI = Cell_type_num['e'],Cell_type_num['i']
        self.MFE_num,self.MFE_flag = 0,0
        
    def initialize(self,t0=0.0):
        """
        initialize by hand, first put all sub-population and connection-pair
        !!! put them on the same platform!!! simulationBridge
        """
        
        # An connection_distribution_list (store unique connection(defined by weight,syn,prob))
        self.connection_distribution_collection = ConnectionDistributionCollection() # this is 
        self.t = t0

        # Matrix to record 
        numCGPatch = self.Net_settings['nmax'] * 2 # excitatory and inhibitory
        # 2 * numCGPatch = External Population and Recurrent Population
        # set Matrix to record only Internal Population
        self.m_record = np.zeros((numCGPatch+1, self.ntt + 10))  
        
        # put all subpopulation and all connections into the same platform
        for subpop in self.population_list:
            subpop.simulation = self    # .simulation = self(self is what we called 'simulation')
        for connpair in self.connection_list:
            connpair.simulation = self
            
        # initialize population_list, calculate         
        for p in self.population_list:
            p.initialize()      # 2   
        
        for c in self.connection_list:
            #print 'initialize population'
            c.initialize()      # 1
            
        # Calculate MFE-probability
        self.iteration_max = self.ntt + 100
        iteration_max = self.iteration_max
        self.tbin_tmp = 0
        self.tbinsize = 1.0
        dtperbin = int(self.tbinsize/self.dt)
        
        iteration_bin = int(iteration_max/dtperbin)
        NPATCH,NE,NI = self.Net_settings['hyp_num'],self.NE,self.NI
        
        # Parameters and Variables recorded
        self.mEbin_ra = np.zeros((iteration_bin,NPATCH))
        self.mIbin_ra = np.zeros((iteration_bin,NPATCH))
#        self.HNMDAEbin_ra = np.zeros((iteration_bin,NPATCH))
#        self.HNMDAIbin_ra = np.zeros((iteration_bin,NPATCH))
        self.NMDAEbin_ra  = np.zeros((iteration_bin,NPATCH))
        self.NMDAIbin_ra  = np.zeros((iteration_bin,NPATCH))
        self.P_MFEbin_ra  = np.zeros((iteration_bin,NPATCH))
        # MFE or Not
        self.P_MFE_eff    = np.zeros((iteration_max,2)) # ,0] Prob and ,1] index
        # rhov 
        self.rEbin_ra     = np.zeros((NPATCH,200,iteration_bin))
        self.rIbin_ra     = np.zeros((NPATCH,200,iteration_bin))
        # STILL NEED _RA 
        self.LE_ra = np.zeros((iteration_max,NPATCH))
        self.LI_ra = np.zeros((iteration_max,NPATCH))
        
        # Prepare for MFE
        DEE,DIE,DEI,DII = self.DEE,self.DIE,self.DEI,self.DII 
        vT = 1.0
        dv = self.Net_settings['dv']
        self.Vedges = util.get_v_edges(-1.0,1.0,dv)
        # Nodes(Vbins) = Nodes(Vedges)-1
        self.Vbins  = 0.5*(self.Vedges[:-1] + self.Vedges[1:])
        # >>>>>>>>>>>>>>>>>>> splin
        self.Vedgesintp = util.get_v_edges(-1.0,1.0,1e-3)
        self.Vbinsintp  = 0.5*(self.Vedgesintp[:-1] + self.Vedgesintp[1:])
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        Vedges      = self.Vedges.copy()
        Vbins       = self.Vbins.copy()
        
        Vedgesintp  = self.Vedgesintp.copy()
        Vbinsintp   = self.Vbinsintp.copy()
        
        idx_vT      = len(Vedgesintp) - 1
        idx_kickE,idx_kickI = np.zeros((NPATCH,NPATCH),dtype= int), np.zeros((NPATCH,NPATCH),dtype= int)
        for it in range(self.NPATCH):
            for js in range(self.NPATCH):
                value_kickE = vT - DEE[it,js]
                value_kickI = vT - DIE[it,js]
                
                Ind_k1      = np.where(Vedgesintp>value_kickE)
                IndI_k1     = np.where(Vedgesintp>value_kickI)
                if np.shape(Ind_k1)[1]>0:
                    idx_kickE[it,js] = Ind_k1[0][0]
                else:
                    idx_kickE[it,js] = idx_vT
                if np.shape(IndI_k1)[1]>0:
                    idx_kickI[it,js] = IndI_k1[0][0]
                else:
                    idx_kickI[it,js] = idx_vT
        self.idx_kickE,self.idx_kickI = idx_kickE,idx_kickI
        print('kick!>>>',self.idx_kickE)
        self.idx_vT = idx_vT
        self.MFE_pevent = np.zeros(self.NPATCH)
        self.p_single   = np.zeros(self.NPATCH)
        
        
        
        
            
    def update(self,t0 = 0.0,dt = 1e-1,tf = 20.0):
        self.dt = dt#Variable(torch.Tensor([dt]))
        self.tf = tf#Variable(torch.Tensor([tf]))   
        NE,NI = self.NE,self.NI
        # initialize:
        start_time = time.time()
        self.initialize(t0)
        self.initialize_time_period = time.time()-start_time
        
        # start_running
        start_time = time.time()
        counter = 0
        numCGPatch = self.Net_settings['nmax'] * 2
        print('Number of both Excitatory and Inhibitory populations (double of NPATCH):',self.Net_settings['nmax']*2)
        while self.t < self.tf:
            rhoVrec = np.zeros((4,200))
            self.t+=self.dt
            # >>>>>>>>> Time bin size >>>>>>>>>>>>>>
            self.tbin_tmp = int(np.floor(self.t/self.tbinsize))
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Normal Process
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            ind_rec = 0
            # >>>>>>>>>>>>> Update (Normal) >>>>>>>>>>>>>>>
            for p in self.population_list:
                p.update()
                ind_rec += 1
                if(np.mod(counter,100)==0):
                    if(ind_rec<=numCGPatch):
                        print('External firing_rate: %.5f'%p.curr_firing_rate) 
                 
                    else:
                        rhoVrec[ind_rec-numCGPatch-1,:] = p.rhov
                        print('recurrent firing_rate: %.5f'%p.curr_firing_rate)  
            # >>>>>>>>>>>>>>>>> Plot Figure >>>>>>>>>>>>>>>>
            if np.mod(counter,100)==0:
                plt.figure(1)
                for ipop in range(4):
                    if np.mod(ipop,2)==0:
                        plt.plot(rhoVrec[ipop,:],'r')
                        plt.pause(0.1)
                    else:
                        plt.plot(rhoVrec[ipop,:],'b')
                        plt.pause(0.1)                        
            ind_rec,idx_E,idx_I = 0,0,0
            for p in self.population_list:
                ind_rec += 1
                # recurrent~
                if(ind_rec>numCGPatch):
                    if p.celltype =='e':
                        self.rhovE[idx_E,:] = p.rhov
                        self.INMDAE[idx_E]  = p.inmda
                        self.HNMDAE[idx_E]  = p.hnmda
                        self.p_single[idx_E] = p.curr_firing_rate * self.dt * NE # This step is for ...
                        idx_E += 1
                    if p.celltype =='i':
                        self.rhovI[idx_I,:] = p.rhov
                        self.INMDAI[idx_I]  = p.inmda
                        self.HNMDAI[idx_I]  = p.hnmda
                        idx_I += 1    
            # >>>>>>>>>>>>>>> CHECK >>>>>>>>>>>>>>>
            # >>>>>>>>>>>>>>> MFE >>>>>>>>>>>>>>>>>
            # >>>>>>>>>>>>>>> CHECK >>>>>>>>>>>>>>>
            h = (self.Vedgesintp[1] -self.Vedgesintp[0])/(self.Vedges[1] - self.Vedges[0]) # bin
            local_pevent = 1.0 # None co-spike neuron
            NPATCH = self.NPATCH
            # Calculate MFE-prob
            for isource in range(self.NPATCH): #  only excitatory source neuron(population) could trigger MFE
                local_pevent = 1.0
                for jt in range(NPATCH):
                    kickE,kickI = self.idx_kickE[jt,isource],self.idx_kickI[jt,isource]
                    idx_vT      = self.idx_vT
                    
                    '''
                    Only excitatory source population could trigger MFE
                    '''
                    # >>>>>>> Excitatory target population 
                    trhovE = np.squeeze(self.rhovE[jt,:])
                    # interp1d
                    x = self.Vbins
                    y = trhovE
                    f = interpolate.interp1d(x,y,fill_value='extrapolate')
                    xnew = self.Vbinsintp
                    ynew = f(xnew)
                    Nup = NE
                    prob_event = (1.0-np.sum(np.squeeze(ynew[kickE:idx_vT]))*h) ** Nup
                    
                    
                    local_pevent *= prob_event
                    # >>>>>> Inhibitory target population 
                    trhovI = np.squeeze(self.rhovI[jt,:])
                    # interp1d
                    x = self.Vbins
                    y = trhovI
                    f = interpolate.interp1d(x,y,fill_value='extrapolate')
                    xnew = self.Vbinsintp
                    ynew = f(xnew)
                    Nup = NI
                    prob_event = (1.0-np.sum(np.squeeze(ynew[kickI:idx_vT]))*h) ** Nup
                    local_pevent *= prob_event
                # so for a particular isource EXCITATORY population, we could calculate MFE-prob
                self.MFE_pevent[isource] = self.p_single[isource] * (1-local_pevent)
                
                
            MFE_pevent_max = max(self.MFE_pevent)
            idx_pevent_max = np.argmax(self.MFE_pevent) # which excitatory population was chosen to trigger MFE
            self.P_MFEbin_ra[self.tbin_tmp,idx_pevent_max] += MFE_pevent_max * dt
            # record that population
            # but not effective
            
            # >>>>>>> choose random number to decide whether do MFE-process
            local_pevent_d = np.random.random()
            if local_pevent_d < MFE_pevent_max:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>> MFE !!! >>>>>>>>>>>>>>>>>>')
                print('PROB>>>>>>>>:',MFE_pevent_max,'; Rand>>>',local_pevent_d)
                # effective MFE
                self.MFE_flag = 1
                self.MFE_num += 1
                self.P_MFE_eff[self.MFE_num,1] = idx_pevent_max
                self.P_MFE_eff[self.MFE_num,0] = MFE_pevent_max
                # start MFE
                
                ind_rec,idx_E,idx_I = 0,0,0
                for p in self.population_list:
                    ind_rec += 1
                    # recurrent~
                    if(ind_rec>numCGPatch):
#                        print('>>>Type:',p.type)
                        if p.celltype =='e':
                            if idx_E == idx_pevent_max:
                                # MFEflag = 1
                                p.MFEflag = self.MFE_flag # 1
                            p.firing_rate = 0.0
#                            self.rhovE[idx_E,:] = p.rhov
#                            print('idx:',ind_rec,' celltype:',p.celltype,' MFE-flag:',p.MFEflag)
                            idx_E += 1
                        if p.celltype =='i':
                            p.firing_rate = 0.0
#                            self.rhovI[idx_I,:] = p.rhov
#                            print('idx:',ind_rec,' celltype:',p.celltype,' MFE-flag:',p.MFEflag)
                            idx_I += 1


                # update information
                for c in self.connection_list:
                    c.update()
                # do first step in MFE
                ind_rec,idx_E,idx_I = 0,0,0
                for p in self.population_list:
                    ind_rec += 1
                    if (ind_rec > numCGPatch):
                        fstMFE,cascadFR = p.update_MFE()
                        if p.celltype == 'e':
                            self.MFE_frE[idx_E] = fstMFE
                            self.FR_casE[idx_E] = cascadFR
#                            print('idx-e:',idx_E,',',p.celltype,', cascade FR:',self.FR_casE[idx_E],'/',p.curr_firing_rate)
                            idx_E   += 1
                            
                        if p.celltype == 'i':
                            self.MFE_frI[idx_I] = fstMFE
                            self.FR_casI[idx_I] = cascadFR
#                            print('idx-i:',idx_I,',',p.celltype,', cascade FR:',self.FR_casI[idx_I],'/',p.curr_firing_rate)
                            idx_I   += 1

                ind_rec,idx_E,idx_I = 0,0,0
                for p in self.population_list:
                    ind_rec += 1
                    # recurrent~
                    if(ind_rec>numCGPatch):
#                        print('>>>accumulated:')
                        if p.celltype == 'e':
#                            print('idx-e:',idx_E,'accumulated FR:',p.MFE_firing_rate)
                            idx_E   += 1
                            
                        if p.celltype == 'i':
#                            print('idx-i:',idx_I,'accumulated FR:',p.MFE_firing_rate)
                            idx_I   += 1

                # update information
                for c in self.connection_list:
                    c.update()
                # make sure that MFEflag was resetted back to zero
                ind_rec,idx_E,idx_I = 0,0,0
                for p in self.population_list:
                    ind_rec += 1
                    # recurrent~
                    if(ind_rec>numCGPatch):
                        if p.celltype =='e':
                            p.MFEflag = 0
                            idx_E += 1
                        if p.celltype =='i':
                            p.MFEflag = 0
                            idx_I += 1
                # update information
                for c in self.connection_list:
                    c.update()                
                # Calculate Potential MFE
                ind_rec,idx_E,idx_I = 0,0,0
                MAX_Potential_MFE = -np.inf
                for p in self.population_list:
                    ind_rec += 1
                    # recurrent~
                    if(ind_rec>numCGPatch):
                        if p.celltype =='e':
                            pot_MFE = p.update_potential_MFE()
                            if pot_MFE > MAX_Potential_MFE:
                                MAX_Potential_MFE = pot_MFE
                            idx_E += 1
                        if p.celltype =='i':
                            pot_MFE = p.update_potential_MFE()
                            if pot_MFE > MAX_Potential_MFE:
                                MAX_Potential_MFE = pot_MFE
                            idx_I += 1                
                # ITERATION mfe
                countMFE = 0
                while MAX_Potential_MFE > self.epsMFE:
                    ind_rec,idx_E,idx_I = 0,0,0
                    for p in self.population_list:
                        ind_rec += 1
                        if (ind_rec > numCGPatch):
                            
                            fstMFE,cascadFR = p.update_iteration_MFE()
                            if p.celltype == 'e':
                                self.newrhovE[idx_E,:] = p.rhov
                                self.MFE_frE[idx_E] = fstMFE
                                self.FR_casE[idx_E] = cascadFR
#                                print('idx-e:',idx_E,',',p.celltype,', cascade FR:',self.FR_casE[idx_E],'/',p.curr_firing_rate)
                                idx_E   += 1
                                
                            if p.celltype == 'i':
                                self.newrhovI[idx_I,:] = p.rhov
                                self.MFE_frI[idx_I] = fstMFE
                                self.FR_casI[idx_I] = cascadFR
#                                print('idx-i:',idx_I,',',p.celltype,', cascade FR:',self.FR_casI[idx_I],'/',p.curr_firing_rate)
                                idx_I   += 1
                    # update information
                    for c in self.connection_list:
                        c.update()
                    # RE-CALCULATE mfe-prob
                    ind_rec,idx_E,idx_I = 0,0,0
                    MAX_Potential_MFE = -np.inf
                    for p in self.population_list:
                        ind_rec += 1
                        # recurrent~
                        if(ind_rec>numCGPatch):
                            if p.celltype =='e':
                                pot_MFE = p.update_potential_MFE()
                                if pot_MFE > MAX_Potential_MFE:
                                    MAX_Potential_MFE = pot_MFE
                                idx_E += 1
                            if p.celltype =='i':
                                pot_MFE = p.update_potential_MFE()
                                if pot_MFE > MAX_Potential_MFE:
                                    MAX_Potential_MFE = pot_MFE
                                idx_I += 1   
#                    print('ITER:',countMFE,' === MAXpotential:',MAX_Potential_MFE)
                    countMFE += 1
                
                # update firing rate and HNMDA
                # 1st update firing rate
                ind_rec,idx_E,idx_I = 0,0,0
                for p in self.population_list:
                    ind_rec += 1
                    # recurrent~
                    if(ind_rec>numCGPatch):
                        if p.celltype =='e':
                            p.firing_rate = 0.0
                            p.hnmda += p.MFE_firing_rate
                            # >>>>> record LE
                            self.LE_ra[counter,idx_E] = p.MFE_firing_rate
                            idx_E += 1
                        if p.celltype =='i':
                            p.firing_rate = 0.0
                            p.hnmda += p.MFE_firing_rate
                            # >>>>> record LI
                            self.LI_ra[counter,idx_I] = p.MFE_firing_rate
                            idx_I += 1   
#                print('ITER:',countMFE,' === MAXpotential:',MAX_Potential_MFE)
                countMFE += 1
                
            # at first ! counter +1
            for c in self.connection_list:
                c.update()                
            counter +=1
            
#                return self.MFE_frE,self.MFE_frI,self.rhovE,self.rhovI,self.newrhovE,self.newrhovI,self.FR_casE,self.FR_casI
                        
            #>>>>>>>>>>>>>>>>>>>> Recording >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # >>>>>>>>>>>>>>>>>>> rhov, m >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            self.tbin_ra[counter] = np.floor(self.t/self.tbinsize)
            tbin = int(np.floor(self.t/self.tbinsize))
            ind_rec,idxE,idxI = 0,0,0
            for p in self.population_list:
                ind_rec += 1
                if(ind_rec > numCGPatch):
                    if p.celltype =='e':
                        self.mEbin_ra[tbin,idxE] += (1-self.MFE_flag) * p.curr_firing_rate * NE * dt * dt + self.MFE_flag * self.LE_ra[counter-1,idxE] * dt
                        self.rEbin_ra[idxE,:,tbin] += p.rhov
                        # >>>>> LR Variables
                        self.NMDAEbin_ra[tbin,idxE] += p.total_longrange_NMDA * dt * dt
                        idxE += 1
                    if p.celltype =='i':
                        self.mIbin_ra[tbin,idxI] += (1-self.MFE_flag) * p.curr_firing_rate * NI * dt * dt + self.MFE_flag * self.LI_ra[counter-1,idxI] * dt
                        self.rIbin_ra[idxI,:,tbin] += p.rhov
                        # >>>>> LR Variables
                        self.NMDAIbin_ra[tbin,idxI] += p.total_longrange_NMDA * dt * dt
                        idxI += 1
            # >>>>>>>>>>>>>>>>>>>>> PLOTTING !!!>>>>>>>>>>>>>>>>>>>>>>>>>>
            # >>>>>>>>>>>>>>>>>>>>> VISUALIZE >>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if np.mod(counter,100) < 1:
                if np.mod(counter,100) == 0:
                    print('time point: ',counter * self.dt)
                for i in range(NPATCH):
                    idshown = np.floor(tbin/5000)
                    idstart = np.int(idshown * 5000)
                    if idstart == tbin:
                        idstart = idstart -1
                        
                    ttt = np.arange(idstart,tbin) * 1.0
                    plt.figure(11)
                    plt.subplot(2,1,int(i)+1)
                    plt.plot(ttt,self.mEbin_ra[idstart:tbin,i],'r')
                    plt.xlim(ttt[0],ttt[0] + 5000)
                    plt.ylim([0,40])
                    plt.pause(0.1)
                    plt.figure(12)
                    plt.subplot(2,1,int(i)+1)
                    plt.plot(ttt,self.mIbin_ra[idstart:tbin,i],'r')
                    plt.xlim(ttt[0],ttt[0] + 5000)
                    plt.ylim([0,40])
                    plt.pause(0.1)
                    

        return self.MFE_frE,self.MFE_frI,self.rhovE,self.rhovI,self.newrhovE,self.newrhovI,self.FR_casE,self.FR_casI,self.HNMDAE,self.HNMDAI,self.INMDAE,self.INMDAI










