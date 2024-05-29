import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from harmonicgp import regionandlockdown, harmonic_matrix, harmonic_model, mask_posterior_exceedance, plot, csv_function, plotPrediction
import s3fs
import os
import boto3
from botocore.client import Config
import time
import pickle
import arviz as az



def do_gp(rawdataset, lockdowndataset, regiondataset, task_config, mindate, ax):

    """
    Calls on the functions in the binomGP file using the inputs given from the harmonicgparray.py file.
    Formats raw datasets, runs gaussian process and produces plots.

    Returns nothing.
    
    """

    formatteddf = regionandlockdown(rawdataset,
                                    lockdowndataset,
                                    regiondataset,
                                    task_config)
    

    times = np.arange(formatteddf.shape[0])*7.0

    design_matrix = np.concatenate(
        [
            np.ones((formatteddf.shape[0],1)),  # Ones vector for intercept
            times[:, np.newaxis] - np.mean(times),  # Centred time
            harmonic_matrix(times, num_harmonics=3), # Harmonic matrix
        ],
        axis=-1
    )


    harmonicresults = harmonic_model(y=formatteddf[task_config['mpc']].to_numpy(),
                              t = times,
                              n=formatteddf['total_count'].to_numpy(),
                              design_matrix=design_matrix,
			                  l = formatteddf['Lockdownoffset'],
                              mcmc_iter=5000)

    print(az.loo(harmonicresults['trace']['pi_star']))

    exceedance_mask = mask_posterior_exceedance(harmonicresults['pred']['s_star'])


    plottinginformation = plotPrediction(ax,formatteddf['day'], formatteddf[task_config['mpc']], formatteddf['total_count'], harmonicresults['pred']['y_star'], mindate, task_config, prev_mult=1000, lag=None,plot_gp=False)

    #plot(ax=ax, 
    #     df=formatteddf, 
    #     exceedance_mask=exceedance_mask, 
    #     task_config=task_config, 
    #     mindate=mindate, 
    #     harmonicresults=harmonicresults)

    csv_function(formatteddf, harmonicresults, plottinginformation, task_config, exceedance_mask)

