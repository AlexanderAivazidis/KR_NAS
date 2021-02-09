# -*- coding: utf-8 -*-
"""Default Generative Model for GeoMx counts"""

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
from functools import wraps
from adjustText import adjust_text
import matplotlib.cm as cm

from pycell2location.models.pymc3_model import Pymc3Model 

# defining the model itself
class WTACounts_GeneralModel(Pymc3Model):
    r"""GeoMx Generative Model:
    :param X_data: Numpy array of gene probe counts (cols) in ROIs (rows)
    :param Y_data: Numpy array of negative probe counts (cols) in ROIs (rows)
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param n_iter: number of training iterations
    :param total_grad_norm_constraint: gradient constraints in optimisation
    """

    def __init__(
        self,
        X_data: np.ndarray,
        Y_data: np.ndarray,
        data_type: str = 'float32',
        n_iter = 200000,
        learning_rate = 0.001,
        total_grad_norm_constraint = 200,
        verbose = True,
        var_names=None, var_names_read=None,
        obs_names=None, fact_names=None, sample_id=None,
        n_factors = 7,
        cutoff_poisson = 1000,
        h_alpha = 1
    ):
        
        ############# Initialise parameters ################
        super().__init__(X_data, 0,
                         data_type, n_iter, 
                         learning_rate, total_grad_norm_constraint,
                         verbose, var_names, var_names_read,
                         obs_names, fact_names, sample_id)
        self.Y_data = Y_data
        self.y_data = theano.shared(Y_data.astype(self.data_type))
        self.n_rois = Y_data.shape[0]
        self.l_r = np.array([np.sum(X_data[i,:]) for i in range(self.n_rois)]).reshape(self.n_rois,1)/self.n_genes
        self.n_factors = n_factors
        self.n_npro = Y_data.shape[1]
        self.cutoff_poisson = cutoff_poisson
        self.poisson_residual = self.X_data < self.cutoff_poisson
        self.gamma_residual = self.X_data > self.cutoff_poisson
        self.X_data1 = self.X_data[self.poisson_residual]
        self.X_data2 = self.X_data[self.gamma_residual]
        self.genes = var_names
        self.sample_names = obs_names
        self.h_alpha = h_alpha
        
        ############# Define the model ################
        self.model = pm.Model()
        with self.model:
            
            ### Negative Probe Counts ###
            
            # Prior for distribution of negative probe count levels:
            self.b_n_hyper = pm.Gamma('b_n_hyper', alpha = np.array((3,1)), beta = np.array((1,1)), shape = 2)
            self.b_n = pm.Gamma('b_n', mu = self.b_n_hyper[0], sigma = self.b_n_hyper[1], shape = (1,self.n_npro))
            self.y_rn = self.b_n*self.l_r
            
            ### Gene Counts ###
            
            # Background for gene probes, drawn from the same distribution as negative probes:
            self.b_g = pm.Gamma('b_g', mu = self.b_n_hyper[0], sigma = self.b_n_hyper[1], shape = (1,self.n_genes))

            # Gene expression modeled as combination of non-negative factors:
            self.h_hyp = pm.Gamma('h_hyp', 1, 1, shape = 1)
            self.h = pm.Gamma('h', alpha = 1, beta = self.h_hyp, shape=(self.n_genes, self.n_factors))
            self.w_hyp = pm.Gamma('w_hyp', np.array((1,1)), np.array((1,1)), shape=(self.n_factors,2))
            self.w = pm.Gamma('w', mu=self.w_hyp[:,0], sigma=self.w_hyp[:,1], shape=(self.n_rois, self.n_factors))
            self.a_gr =  pm.Deterministic('a_gr', pm.math.dot(self.w, self.h.T))
            
            # Expected gene counts are sum of gene expression and background counts, scaled by library size:
            self.x_rg = (self.a_gr + self.b_g)*self.l_r
            
            self.data_target = pm.DensityDist('data_target', self.get_logDensity, observed=tt.concatenate([self.y_data, self.x_data], axis = 1))
    
    
    def get_logDensity(self,x):
        # Likelihood function combines likelihood for negative probes and gene probes and replaces poisson sampling
        # for gamma distribution for very high counts (for numerical stability)
            mu1 = self.y_rn
            mu2 = self.x_rg[self.poisson_residual]
            mu3 = self.x_rg[self.gamma_residual]
            logDensity = tt.sum(pm.Poisson.dist(mu1).logp(x[:,:self.n_npro])) + tt.sum(pm.Poisson.dist(mu2).logp(x[:,self.n_npro:][self.poisson_residual])) + tt.sum(pm.Gamma.dist(mu = mu3, sd = tt.sqrt(mu3)).logp(x[:,self.n_npro:][self.gamma_residual]))
            return logDensity
    
    def compute_expected(self):
        r""" Save expected value of negative probe poisson mean and negative probe level"""
        # compute poisson mean of negative probes:
        self.y_rn_mean = self.samples['post_sample_means']['b_n'] * self.l_r
        # compute poisson mean of unnormalized gene probe counts:
        self.x_rg_mean = (self.samples['post_sample_means']['a_gr'] + self.samples['post_sample_means']['b_g'])*self.l_r
        
    def compute_logFC(self, groupA = 3, groupB = 6, n_samples = 1000):
        r"""Compute log-fold change of genes between two ROIs or two groups of ROIs, as well as associated
        standard deviation and 0.05 and 0.95 percentile"""
        groupA = np.squeeze(groupA)
        groupB = np.squeeze(groupB)
        post_sample_a_gr = self.mean_field['init_1'].sample_node(self.a_gr, size = n_samples).eval()
        self.logFC = pd.DataFrame(index = self.genes, columns = ('groupA', 'groupB', 'mean', 'sd', 'q05', 'q95'))
        
        if sum(np.shape(groupA)) < 2:
            groupA_value = np.log2(post_sample_a_gr[:,groupA,:])
            self.logFC['groupA'] = self.sample_names[groupA]
        else:
            groupA_value = np.log2(np.mean(post_sample_a_gr[:,groupA,:], axis = 1))
            self.logFC['groupA'] = ', '.join(self.sample_names[groupA])
        if sum(np.shape(groupB)) < 2:
            groupB_value = np.log2(post_sample_a_gr[:,groupB,:])
            self.logFC['groupB'] = self.sample_names[groupB]
        else:
            groupB_value = np.log2(np.mean(post_sample_a_gr[:,groupB,:], axis = 1))
            self.logFC['groupB'] = ', '.join(self.sample_names[groupB])
        
        self.logFC_sample =  groupA_value - groupB_value
        self.logFC['mean'] = self.logFC_sample.mean(axis=0)
        self.logFC['sd'] = self.logFC_sample.std(axis=0) 
        self.logFC['q05'] = np.quantile(self.logFC_sample, 0.05, axis=0)
        self.logFC['q95'] = np.quantile(self.logFC_sample, 0.95, axis=0)  
        
    def compute_FDR(self, logFC_threshold = 1):
        r"""Compute probability that logFC is above a certain threshold 
        and also include FDR for each probability level.
        :logFC_threshold: logFC threshold above which we define a discovery"""
        self.logFC['threshold'] = logFC_threshold
        self.logFC['probability'] = np.sum(np.abs(self.logFC_sample) > logFC_threshold, axis = 0)/np.shape(self.logFC_sample)[0]
        probability = self.logFC['probability']
        self.logFC['FDR'] = np.array([sum(1-probability[probability >= p])/sum(probability >= p) for p in probability])
        self.logFC = self.logFC.sort_values('FDR')
    
    def plot_volcano(self, genesOfInterest = None, n_max_genes = 1, alpha = 0.25, FDR_cutoff = 0.05,
                     height = 10, width = 10):
        r""" Make a volcano plot of the differential expression analysis.
        :genesOfInterest: numpy array of genes to annotate in the plot
        :n_max_genes: number of genes to automatically annotate at the extreme ends of the plot,
        i.e. the most differentially expressed genes
        :alpha: transparency of dots 
        :FDR_cutoff: what false discovery rate to use
        :height: height of figure
        :width: width of figure
        """
        
        # Set figure parameters:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        plt.figure(figsize=(width,height))
        colours = np.repeat('grey', self.n_genes)
        colours[[self.logFC['mean'][i] > 0 and self.logFC['FDR'][i] < FDR_cutoff for i in range(len(self.logFC['mean']))]] = 'blue'
        colours[[self.logFC['mean'][i] < 0 and self.logFC['FDR'][i] < FDR_cutoff for i in range(len(self.logFC['mean']))]] = 'red'
        plt.scatter(self.logFC['mean'], -np.log10(1-self.logFC['probability']+0.0001), c=colours, alpha = 0.1)
        #plt.hlines(np.amin(self.logFC['probability'][self.logFC['FDR'] < FDR_cutoff]), np.amin(self.logFC['mean']),
        #           np.amax(self.logFC['mean']), linestyles = 'dashed')
        #plt.text(np.amin(self.logFC['mean']),np.amin(self.logFC['probability'][self.logFC['FDR'] < FDR_cutoff]) + 0.01,
        #         'FDR < ' + str(FDR_cutoff))
        plt.xlabel('Log2FC')
        plt.ylim(0,3.5)
        plt.ylabel('-log10(P(Log2FC > ' + str(self.logFC['threshold'][0]) + '))')
        
        if n_max_genes > 0:
            
            maxGenes = np.array((self.logFC.index[np.argmax(self.logFC['mean'])],
                                 self.logFC.index[np.argmin(self.logFC['mean'])]))
        if genesOfInterest is None:
            
            genesOfInterest = maxGenes
        else:
            genesOfInterest = np.concatenate((genesOfInterest, maxGenes))
        
        if genesOfInterest is not None:
        
            geneIndex_to_annotate = np.squeeze([np.where(self.logFC.index == genesOfInterest[i])
                                                for i in range(len(genesOfInterest))])
            
            ts = []    
            for i,j in enumerate(geneIndex_to_annotate):
                ts.append(plt.text(self.logFC['mean'][j], -np.log10(1-self.logFC['probability'][j]), genesOfInterest[i]))
            adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 2.5,
                       force_points = 2.5, force_objects = 2.5)
        
        plt.show()
        
        
    def plot_Log2FC_vs_sd(self, genesOfInterest = None, n_max_genes = 3, ylim = None):
        r""" Make a volcano plot of the differential expression analysis.
        :genesOfInterest: numpy array of genes to annotate in the plot
        :n_max_genes: number of genes to annotate at the extreme ends of the plot,
        i.e. the most differentially expressed genes
        """
        
        # Set figure parameters:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        f = plt.figure(figsize=(20,10))
        colours = np.repeat('grey', self.n_genes)
        colours[[self.logFC['mean'][i] > 0 and self.logFC['FDR'][i] < 0.05 for i in range(len(self.logFC['mean']))]] = 'blue'
        colours[[self.logFC['mean'][i] < 0 and self.logFC['FDR'][i] < 0.05 for i in range(len(self.logFC['mean']))]] = 'red'
        plt.scatter(self.logFC['mean'], self.logFC['sd'], c = colours)
        plt.xlabel('Log2FC_mean')
        plt.ylabel('Log2FC_sd')    
        ts = []    
        for i,j in enumerate(geneIndex_to_annotate):
            ts.append(plt.text(self.logFC['mean'][j], self.logFC['sd'][j], genesOfInterest[i]))
        adjust_text(ts, arrowprops=dict(arrowstyle='->', color='black'), force_text = 1.5, force_objects = 1, force_points = 1)    
        plt.show()    
        
    def plot_factor_weights(self, order = None):
        r""" Plot the factor weights in a heatmap, with the rows (factors) cluster and the samples in the order provided
        :genesOfInterest: numpy array of order (indexes) to put the samples in the heatmap, 
        if None then no reordering will take place
        """ 
        if order is None:
            order = np.arrange(1,len())
        w = self.samples['post_sample_means']['w'][order,:]
        sns.clustermap(w.T, col_cluster=False)        
        
    def plot_negativeProbes_vs_geneBackground(self, samples, n_bins = 100, n_samples = 1000):
        r""" Plot histogram of the negative probe counts and the poisterior poisson mean of background counts
        across all genes
        :samples: numpy array with sequence of samples for which to do this plot
        :n_bins: number of bins for the geneBackground histogram
        :n_samples: number of samples to use for approximating posterior distribution of gene background
        """
        post_sample_b_g = self.mean_field['init_1'].sample_node(self.b_g, size = n_samples).eval()
        
        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle('Negative Probe Counts and Posterior Poisson Mean of all Background Counts')
        
        for i in range(len(samples)):
            ax[0].hist(self.Y_data[:,samples[i]], bins = 10, alpha = 0.75)
        
        ax[0].set_ylabel('Number of Probes')
        
        for i in range(len(samples)):        
            ax[1].hist(np.squeeze(post_sample_b_g[:,:,:]*self.l_r[samples[i]]).flatten(),
                       bins = n_bins, density = True, label = 'sample ' + str(i), alpha = 0.75)

        ax[1].set_xlabel('Counts')
        ax[1].set_ylabel('Probability Density')
        ax[1].set_xscale('log')
        ax[1].legend()
        plt.show()                            
                                              
    def plot_single_geneCounts_and_poissonMean(self, gene, samples, n_samples = 1000):
        r""" Plot a scatter plot of gene counts and a histogram of predicted poisson mean
        :gene: which example gene to plot
        :samples: numpy array with sequence of samples for which to do this plot, maximum is 6
        :n_samples: how many samples to take to approximate posterior
        """
        
        post_sample_a_gr = self.mean_field['init_1'].sample_node(self.a_gr, size = n_samples).eval()
        post_sample_b_g = self.mean_field['init_1'].sample_node(self.b_g, size = n_samples).eval()
        
        if len(samples) > 5:
            print('Maximum Number of Samples is 6')
        colourPalette = c('blue', 'green', 'red', 'yellow', 'black', 'grey')
        
        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle('Raw Counts and Posterior Poisson Mean of Gene Counts (' + gene + ')')
        ax[0].scatter(X_data[samples, self.genes == gene], ('Sample1','Sample2'),  40,
                      c = [colourPalette[i] for i in range(len(samples))])
        
        for i in range(len(samples)):
            ax[1].hist(np.squeeze(post_sample_a_gr[:,i,self.genes == gene]*self.l_r[i]),
                       bins = n_bins, density = True, label = 'sample 1', color = 'blue')
        
        ax[1].set_xlabel('Counts')
        ax[1].set_ylabel('Probability Density')
        ax[1].legend()

        fig, ax = plt.subplots(2, sharex=True)
        fig.suptitle('Negative Probe Counts and Posterior Poisson Mean of Background Counts (' + gene + ')')
        
        for i in range(len(samples)):
            ax[0].hist(negProbes_subset[:,samples[i]], color = 'blue', bins = 100, alpha = 0.5)        
        
        ax[0].set_ylabel('Number of Probes')
        
        for i in range(len(samples)):
        
            ax[1].hist(np.squeeze(post_sample_b_g[:,:, self.genes == gene]*self.l_r[samples[i]]), bins = 10,
                       density = True, label = 'sample ' + str(i), color = 'blue', alpha = 0.75)

        ax[1].set_xlabel('Counts')
        ax[1].set_ylabel('Probability Density')
        ax[1].set_xscale('log')
        ax[1].legend()
    
    def plot_multiple_geneCounts_and_PoissonMean(self, gene, samples):
        r""" Plot histogram of the negative probe counts anad the predicted gene backgrounds counts
        across all genes
        :samples: numpy array with sequence of samples for which to do this plot
        """
        print('test')
    