# -*- coding: utf-8 -*-
"""Model for GeoMx counts"""

import sys, ast, os
import time
import itertools
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import theano.tensor as tt
import pymc3 as pm
import pickle
import theano

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

from pycell2location.models.pymc3_model import Pymc3Model 

# defining the model itself
class GeoMxCounts(Pymc3Model):
    r"""Single cell expression programme model with M_g in the likelihood & no proportions.
    :param n_fact: Number of expression programmes to find
    :param X_data: Numpy array of gene expression (cols) in cells (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    """

    def __init__(
        self,
        n_fact: int,
        X_data: np.ndarray,
        data_type: str = 'float32',
        n_iter = 200000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None, fact_names=None, sample_id=None
    ):

        ############# Initialise parameters ################
        super().__init__(X_data,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint)
        
        self.cell_factors_df = None

        ############# Define the model ################
        self.model = pm.Model()

        with self.model:
            
            # =====================Gene factors======================= #
            # Gene factor prior:
            self.gene_factors = pm.Dirichlet('gene_factors', a=np.ones(self.n_fact) * 1.,
                                     shape=(self.n_genes, self.n_fact))
            
            # =====================Gene expression level scaling======================= #
            # Explains difference in expression between genes and 
            # how it differs in single cell and spatial technology
            self.gene_level_alpha_hyp = pm.Gamma('gene_level_alpha_hyp', 1, 1, shape=(1, 1))
            self.gene_level_beta_hyp = pm.Gamma('gene_level_beta_hyp', 1, 1, shape=(1, 1))
        
            self.gene_level = pm.Gamma('gene_level', self.gene_level_alpha_hyp,
                                       self.gene_level_beta_hyp, shape=(self.n_genes, 1))
    
            # =====================Cell factors======================= #
            # Cell factor mean:
            self.cell_fact_mu_hyp = pm.Gamma('cell_fact_mu_hyp', 1, 0.1, shape=self.n_fact)
            # Cell factor sd:
            self.cell_fact_sd_hyp = pm.Gamma('cell_fact_sd_hyp', 1, 0.01, shape=self.n_fact)
    
            self.cell_factors = pm.Gamma('cell_factors', mu=self.cell_fact_mu_hyp,
                                         sigma=self.cell_fact_sd_hyp,
                                         shape=(self.n_cells, self.n_fact))
    
            # =====================Cell-specific additive component======================= #
            # molecule contribution that cannot be explained by cell state signatures
            # these counts are distributed between all genes not just expressed genes
            self.cell_add_hyp = pm.Gamma('cell_add_hyp', 1, 1, shape=2)
            self.cell_add = pm.Gamma('cell_add', self.cell_add_hyp[0],
                                     self.cell_add_hyp[1], shape=(self.n_cells, 1))
    
            # =====================Gene-specific additive component (soup)======================= #
            # per gene molecule contribution that cannot be explained by cell state signatures
            # these counts are distributed equally between all cells (e.g. soup or free-floating RNA)
            self.gene_add_hyp = pm.Gamma('gene_add_hyp', 1, 1, shape=2)
            self.gene_add = pm.Gamma('gene_add', self.gene_add_hyp[0],
                                     self.gene_add_hyp[1], shape=(self.n_genes, 1))
            
            # =====================Expected expression ======================= #
            # expected expression
            self.mu_biol = pm.math.dot(self.cell_factors, self.gene_factors.T) * self.gene_level.T + self.gene_add.T + self.cell_add
            #tt.printing.Print('mu_biol')(self.mu_biol.shape)
    
            # =====================DATA likelihood ======================= #
            # Likelihood (sampling distribution) of observations & add overdispersion via NegativeBinomial / Poisson
            self.data_target = pm.Poisson('data_target', mu=self.mu_biol,
                                          observed=self.x_data,
                                          total_size=self.X_data.shape)