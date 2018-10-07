"""Module containing ConnectionDistribution class, organizing connection details."""

# Copyright 2013 Allen Institute
# This file is part of dipde
# dipde is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dipde is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dipde.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import utilities as util

class ConnectionDistribution(object):
    """
    Parameters:
    which could define unique connection,
    like weight, nsyn and prob
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,edges,weights,probs,sparse = True):
        self.edges   = edges
        self.weights = weights
        self.probs   = probs
        """
        be remained!
        1) flux_matrix and threshold_flux_matrix,
        if connection has identical weight syn and prob, then the clux  matrix
        should be identical, this could be reuse --> connection_distribution
        """
        # mentioned above could be solved
        self.flux_matrix = None
        self.threshold_flux_vector = None
        self.fluxn_matrix = None
        self.threshold_fluxn_vector = None
        self.fluxk_matrix = None
        self.threshold_fluxk_matrix = None
        self.nmdajump = None
        # notice that threshold_flux_vector is !!! vector which 
        # could map to each voltage-center
        # self.simulation = None
        
        # reversal potential could be used in conductance based model
        self.reversal_potential = None
        if self.reversal_potential != None:
            assert NotImplementedError 
    def initialize(self):
        """
        if we already have those connectional properties, we could 
        calculate the flux_matrix and threshold_flux_matrix
        these matrix could be reused at identical connection_cluster as well as time steps
        """
        nv = len(self.edges)-1
        self.flux_matrix = np.zeros((nv,nv))
        self.threshold_flux_vector = np.zeros(nv)
        curr_threshold_flux_vector,curr_flux_matrix = util.flux_matrix(self.edges,self.weights,self.probs)
        self.flux_matrix = curr_flux_matrix
        self.threshold_flux_vector = curr_threshold_flux_vector
        self.fluxn_matrix = np.eye(nv)
        self.threshold_fluxn_vector = np.zeros(nv)

        self.fluxk_matrix = np.eye(nv)
        self.threshold_fluxk_vector = np.zeros(nv)
        curr_threshold_fluxk_vector,curr_fluxk_matrix = util.flux_matrix(self.edges,self.weights,self.probs)
        self.fluxk_matrix = curr_fluxk_matrix
        self.threshold_fluxk_vector = curr_threshold_fluxk_vector



    @property    
    def signature(self):
        """
        unique signature
        """
        return (tuple(self.edges),tuple([self.weights]),tuple([self.probs]))           

    
 
    

