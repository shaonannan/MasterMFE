"""Module containing Connection class, connections between source and target populations."""

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
from connectiondistribution import ConnectionDistribution
import collections
import matplotlib.pyplot as plt

# Recurrent Connection
class Connection(object):
    """
    Parameters:
    pre-population
    post-population
    nsyn-population
    connection weight
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,pre,post,nsyn,weights,probs,conn_type):
        self.pre_population = pre
        self.post_population = post
        self.nsyn = nsyn  # Number of Pre(sender) population
        self.weights = weights
        self.probs = probs
        self.conn_type = conn_type
        # multiply probability of connection
        
        """
        1) connection_list should be classified into some unique population(cluster)
        which means, if 'weight''syn''prob' is identical,should be classified into identical 
        connection_distribution
        2) curr_firing_rate could be replace by ...
        3) simulation could be used to find original platform
        """
        # initialize None and Initialize when simulation
        self.firing_rate = 0.0
        self.simulation = None
        # long range
        self.inmda = 0.0
        """
        be remained!
        1) flux_matrix and threshold_flux_matrix,
        if connection has identical weight syn and prob, then the clux  matrix
        should be identical, this could be reuse --> connection_distribution
        """
    # initialize by hand! when start simulation
    def initialize(self):
        self.initialize_connection_distribution()
        self.initialize_firing_rate()
        self.initialize_I_nmda()
    
    def initialize_connection_distribution(self):
        CD = ConnectionDistribution(self.post_population.edges,self.weights,self.probs)
        CD.simulation = self.simulation
        self.simulation.connection_distribution_collection.add_unique_connection(CD)
        self.connection_distribution = self.simulation.connection_distribution_collection[CD.signature]
        
        
    def initialize_firing_rate(self):
        self.firing_rate = self.pre_population.curr_firing_rate
    # LONG RANGE 
    def initialize_I_nmda(self):
        self.inmda = self.pre_population.curr_Inmda
        
    def update(self):
        self.firing_rate = self.pre_population.curr_firing_rate
        self.inmda       = self.pre_population.curr_Inmda
        # initialize_firing_rate
    def update_flux_matrix(self,flux_matrix,threshold_flux_matrix):
        self.flux_matrix = flux_matrix
        self.threshold_flux_matrix = threshold_flux_matrix
    def update_connection(self,npre,npost,nsyn,**nkwargs):
        self.pre_population = [],
        self.pre_population = npre,
        self.post_population = [],
        self.post_population = npost,
        self.syn_population = [],
        self.syn_population = nsyn
        
    @property
    def curr_firing_rate(self):
        return self.firing_rate
    @property
    def curr_Inmda(self):
        return self.inmda
            
