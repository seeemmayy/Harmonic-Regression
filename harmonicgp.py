import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import time
from datetime import date
import s3fs
import os
import boto3
from botocore.client import Config
import json

s3 = s3fs.S3FileSystem(profile="agile-upload",
                       client_kwargs=dict(
                         endpoint_url="https://fhm-chicas-storage.lancs.ac.uk"
                                         )
                      )

def regionandlockdown(df, lockdowndataset, regiondataset, task_config):
    """
    Takes the base dataset and adds in a binary lockdown information
    column and a region column based on the local authority codes for the GP

    Returns a dataframe to be input into the GP function
    """

    #aggregates lockdown dataset by year and week number
    #takes the average lockdown for the week
    #lockdowndataset['consult_date']= lockdowndataset['consult_date'].dt.strftime('%d/%m/%Y')
    #lockdowndataset['year'] = lockdowndataset['consult_date'].dt.year
    lockdown = lockdowndataset[["year", "week_number", "Lockdown"]].groupby(['year','week_number']).mean()
    lockdown['Lockdown'] = lockdown['Lockdown'].apply(lambda x: 1 if x >= 0.5 else 0)
    lockdown = lockdown.reset_index()

    
    df['week_number'] = df["date"].dt.isocalendar()["week"]
    df['year'] = df["date"].dt.isocalendar()["year"]
    df = df.loc[(df['year'] >= 2019)]
    
    #merging region column
    regiondataset = regiondataset.rename(columns={"LAD20CD":"location"})
    df = df.merge(regiondataset, on='location', how='left')
    df = df.rename(columns= {"ITL121NM": "region", "LAD20NM":"local authority", "ITL221NM":"nuts2"})
    df = df[df['species'] == task_config['species']]

    if task_config['location'] is not None:
        df = df[df['region'] == task_config['location']]
        df['region'] = task_config['location']
    

    df = df[["year", "week_number", "gastroenteric", "pruritus", "respiratory", "total_count"]].groupby(['year','week_number']).sum()
    df = df.reset_index()
    df['day'] = df.index * 7
    df['date'] = date.fromisoformat('2018-12-31')
    df['date'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['day'], unit='d')
    
    #merging lockdown column
    df = pd.merge(df,lockdown[["year","week_number","Lockdown"]], how="left")
    df['Lockdown'] = df['Lockdown'].fillna(0)
    Lockdownmean = np.mean(df['Lockdown'])
    df['Lockdownoffset'] = df['Lockdown'] - Lockdownmean
    df = df.dropna(how='any', axis=0)
 

    with s3.open(f"savsnet-agile-artefacts/archive/"+str(time.strftime("%Y-%m-%d"))+"/consult_data/harmonic_gp/input_data/"+task_config['species']+"/"+task_config['mpc']+"/"+task_config['plot_filename']+".csv", "w") as f:
        df.to_csv(f)

    return(df)
    
__refdate = pd.to_datetime('2000-01-01')



def harmonic_matrix(t, num_harmonics, period=365):
    """
    Builds matrix of dummy harmonic variables
    
    t - a vector of times of length T
    num_harmonics - the value of number of harmonics
    period - period of the data, in the same units as 't'
    
    
    returns a [T, 2*num_harmonics] matrix
    """
    
    harmonics = 2*np.pi/period * np.arange(1, num_harmonics+1)
    t_harmonic = t[:, np.newaxis] * harmonics[np.newaxis, :]
    
    # Concatenate cos and sin functions by column
    harmonic_matrix = np.concatenate([np.cos(t_harmonic), 
                                      np.sin(t_harmonic)], axis=-1)
    
    return harmonic_matrix


def harmonic_model(y, t, n, design_matrix, l, mcmc_iter=5000, start={}):
    
    """
    Runs Harmonic Regression and Gaussian Process

    y - a vector of observed values
    t - a vector of times of length T
    n - a total number of trials per each timepoint
    design_matrix - a TxP matrix where T=t.shape[0]
    l - vector of binary offset values whether the UK was in a covid restriction or not
    mcmc_iter - number of itertions for mcmc sampler
    start - a dictionary of starting values for mcmc sampler
    
    returns dictionary of model, trace and predictions
    """
    
    
    num_covariates = design_matrix.shape[-1]
    
    with pm.Model() as model:
        
        #sample prior distributions
        coefficients = pm.Normal('coefficients', 0, 10, shape=num_covariates)
        length_scale = pm.HalfNormal('length_scale', sigma=10)
        gp_variance = pm.HalfNormal('gp_variance', sigma=1)
        observation_noise_variance = pm.HalfNormal('observation_noise_variance', sigma=10)
        lockdown = pm.Normal('lockdown', 0, 100)
        
        #create kernels for gaussian process
        kernel_correlated = gp_variance * pm.gp.cov.Matern32(1, length_scale)
        kernel_noise = pm.gp.cov.WhiteNoise(observation_noise_variance)
        
        #running gaussian process
        gp_u = pm.gp.Latent(cov_func=kernel_correlated)
        gp_epsilon = pm.gp.Latent(cov_func=kernel_noise)
        gp = gp_u + gp_epsilon
        
        t_column = t[:, None]
        s = gp.prior("gp", X=t_column)
        u = gp_u.prior("gp_u", X=t_column)
        e = gp_epsilon.prior("gp_epsilon", X=t_column)
        
        #dt product of harmonic matrix and coefficents + gp + lockdown
        mean_t = (design_matrix @ coefficients) + s + (lockdown * l)
        
        #conditioning on data
        yobs = pm.Binomial("yobs", n=n, p=pm.invlogit(mean_t), observed=y)

        #sample mcmc
        trace = pm.sample(mcmc_iter,
                          chains=1,
                          start=start,
                          tune=1000,
                          idata_kwargs = {'log_likelihood': True})
        
        #prediction ------
        #s_star = gp_u.conditional('s_star', t[:, None]) #outbreak detection variable
        s_star = gp_u.conditional("s_star", 
                                  Xnew=t[:, None], 
                                  given=dict(X=t[:, None], f=u, gp=gp)
                                 )
        #epsilon_star = gp_epsilon.conditional('epsilon_star', t[:, None])
        epsilon_star = gp_epsilon.conditional("epsilon_star", 
                                  Xnew=t[:, None], 
                                  given=dict(X=t[:, None], f=e, gp=gp)
                                 )
        mean_t_star = (design_matrix @ coefficients) + epsilon_star + (lockdown * l) # Business as usual, no outbreak.
        pi_star = pm.Deterministic('pi_star', pm.invlogit(mean_t_star))
        y_star = pm.Binomial('y_star', n, pi_star, observed=y)
	
        
        #u_star = gp_u.conditional("u_star", 
        #                          Xnew=t[:, None], 
        #                          given=dict(X=t[:, None], f=s, gp=gp)
        #                         )
        
        #epsilon_star = gp_epsilon.conditional("epsilon_star", 
        #                          Xnew=t[:, None], 
        #                          given=dict(X=t[:, None], f=s, gp=gp)
        #                         )
        
        pred = pm.sample_posterior_predictive(trace, var_names=['y_star', 's_star', 'pi_star', 'epsilon_star'])
        
        
        return {'model': model, 'trace': trace, 'pred': pred}
    
    

def mask_posterior_exceedance(posterior_gp, threshold=0.0, prob=0.95):
    exceed_prob = (posterior_gp > threshold).mean(axis=0)
    print("exceed prob size:", exceed_prob.shape)
    not_exceed_flag = exceed_prob < prob
    exceed_prob[not_exceed_flag] = np.nan
    return exceed_prob



def plot(ax, df, exceedance_mask, task_config, mindate, harmonicresults):
    
    X = df['day']
    ts = slice(0,X.shape[0])
    x = np.array([mindate + pd.Timedelta(d,unit='D') for d in X[ts]])


    ax[0].plot(x, df[task_config['mpc']], label="MPCs")
    ax[0].scatter(x[~np.isnan(exceedance_mask)], 
                  df[task_config['mpc']][~np.isnan(exceedance_mask)],
                  color="C1",
                  label="Outbreak")
    ax[0].set_ylabel("Number of MPCs")
    ax[0].set_title("Case counts"+str(task_config['year']) + str(task_config['week_number']))
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].set_xlim(xmin=pd.Timestamp('2019-01-01'), xmax=max(x))
    ax[1].plot(x,
               np.mean((harmonicresults['pred']['s_star']) > 0, axis=0),
               label="Pr(gp>0)")
    ax[1].plot(x, exceedance_mask, label="Outbreak")
    ax[1].set_title("Outbreak detection")
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_xlim(xmin=pd.Timestamp('2019-01-01'), xmax=max(x))
    plt.gcf().autofmt_xdate()

    #plt.savefig("Harmonic Gaussian Process"+str(task_config['year']) + str(task_config['week_number'])+".png", dpi=300, bbox_inches='tight')


def plotPrediction(ax,X,y,N,pred,mindate,task_config,lag=None,prev_mult=1,plot_gp=False):
    """Predictive time series plot with (by default) the prediction
    summarised as [0.01,0.05,0.5,0.95,0.99] quantiles, and observations colour-coded
    by tail-probability.
    Parameters
    ==========
    ax -- a set of axes on which to plot
    X  -- 1D array-like of times of length n
    y  -- 1D array-like of observed number of cases at each time of length n
    N  -- 1D array-like of total number at each time of length n
    pred -- 2D m x n array with numerical draws from posterior
    mindate -- a pandas.Timesav xxtamp representing the time origin wrt X
    lag     -- how many days prior to max(X) to plot
    prev_mult -- prevalence multiplier (to get in, eg. prev per 1000 population)
    plot_gp -- plots a GP smudge-o-gram rather than 95% and 99% quantiles.
    Returns
    =======
    Nothing.   Just modifies ax
    """


    # Time slice
    ts = slice(0,X.shape[0])
    if lag is not None:
        ts = slice(X.shape[0]-lag, X.shape[0])
    
    # Data
    x = np.array([mindate + pd.Timedelta(d,unit='D') for d in X[ts]])
    pbar = np.array(y/N)[ts] * prev_mult

    # Prediction quantiles
    phat = pred/np.array(N)[np.newaxis,:].mean(axis=0) * prev_mult
    #phat = pred[:,ts] * prev_mult
    pctiles = np.percentile(phat, [1,5,50,95,99], axis=0)

    # Tail probabilities for observed p
    prp = np.sum(pbar > phat, axis=0)/phat.shape[0]
    prp[prp > .5] = 1. - prp[prp > .5]
    
    # Risk masks
    red = prp <= 0.01
    orange = (0.01 < prp) & (prp <= 0.05)
    green = 0.05 < prp

    # Construct plot
    if plot_gp is True:
        from pymc3.gp.util import plot_gp_dist
        plot_gp_dist(ax,phat,x,plot_samples=False, palette="Blues")
    else:
        ax.fill_between(x, pctiles[4,:], pctiles[0,:], color='lightgrey',alpha=.5,label="99% credible interval")
        ax.fill_between(x, pctiles[3,:], pctiles[1,:], color='lightgrey',alpha=1,label='95% credible interval')
        grey = mpatches.Patch(color='silver', label='At Least one lockdown restriction in place')
        blue = mpatches.Patch(color='cornflowerblue', label='National Lockdown in place')
        ax.axvspan('2020-03-21','2020-07-03', color='royalblue', alpha = .5, zorder = -1)
        ax.axvspan('2020-09-17','2020-12-03', color='royalblue', alpha = .5, zorder = -1)
        ax.axvspan('2021-01-04','2021-03-31', color='royalblue', alpha = .5, zorder = -1)
        ax.axvspan('2020-03-21','2021-03-31', color='lightsteelblue', alpha = .4)
        ax.plot(x, pctiles[2,:], c='grey', ls='-', label="Predicted prevalence")
        ax.scatter(x[green],pbar[green],c='green',s=8,alpha=0.5,label='0.05<p')
        ax.scatter(x[orange],pbar[orange],c='orange',s=8,alpha=0.5,label='0.01<p<=0.05')
        ax.scatter(x[red],pbar[red],c='red',s=8,alpha=0.5,label='p<=0.01')
        legend_elements = [Line2D([0], [0], color='grey', lw=2, label="Predicted Prevalence"),
                           Patch(facecolor='lightgrey', alpha=.5, label = "99% credible interval"),
                           Patch(facecolor='lightgrey', alpha=.2, label = '95% credible interval'),
                           Line2D([0], [0], color='green', marker='o', linestyle='None', label=str('0.05 < p')),
                           Line2D([0], [0], color='orange', marker='o', linestyle='None',label=str('0.01<p<=0.05')),
                           Line2D([0], [0], color='red', marker='o', linestyle='None', label=('p<=0.01')),
                           Patch(facecolor='royalblue', alpha=.5, label = "National Lockdown in Place"),
                           Patch(facecolor='lightsteelblue', alpha=.5, label = "At Least One Lockdown in Place")]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.04,1), loc="upper left")
        ax.set_xlabel('Date')
        ax.yaxis.set_label_text('Prevalence')
        ax.set_title(task_config['plot_title'])
        #ax.set_xlim(pd.Timestamp('2021-01-01'), date.today())
        ax.set_xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2023-02-23'))
        ax.figure.savefig(task_config['plot_filename'], format='png', dpi=300, bbox_inches='tight')


        s3 = s3fs.S3FileSystem(profile="agile-upload",
                       client_kwargs=dict(
                         endpoint_url="https://fhm-chicas-storage.lancs.ac.uk"
                                         )
                      )

        with s3.open(f"savsnet-agile-artefacts/archive/"+str(date.today())+"/consult_data/harmonic_gp/plots/"+task_config['species']+"/"+task_config['mpc']+"/"+task_config['plot_filename']+".png", "wb") as f:
            ax.figure.savefig(f, dpi=300, bbox_inches='tight')


        with s3.open(f"savsnet-agile-artefacts/public/consult_data/latest/harmonic_gp/"+task_config['plot_filename']+".png", "wb") as f:
            ax.figure.savefig(f, dpi=300, bbox_inches='tight')

        os.remove(task_config['plot_filename'])

        return {'percentiles': pctiles, 'pbar': pbar}




def csv_function(formatteddf, gaussianprocess, plotpercentiles, task_config, exceedance_mask):

    """
    This function takes in the gaussian process results and the other
    datasets in order to create a dataset of the output to create a plot.

    param formatteddf: dataframe created in the earlier code that puts in the
                  information needed to put into the gaussian process
    param gaussian process: the raw gaussian process results
    param ggplot: the current plotting function to obtain the different plotting
                  variables (percentiles and prediction etc)


    returns csv document of the information needed to create a plot given the
    results
    
    """

    df = pd.DataFrame()

    df['year'] = formatteddf['year']
    df['week_number'] = formatteddf['week_number']
    df['date_week_beginning'] = formatteddf['date']
    df['species'] = task_config['species']
    df['mpc'] = task_config['mpc']
    df['exceedance_probabilities'] = exceedance_mask
    df['mpc_value'] = plotpercentiles['pbar']
    df['total_count'] = formatteddf['total_count']
    df['percentile1'] = plotpercentiles['percentiles'][0]
    df['percentile2'] = plotpercentiles['percentiles'][1]
    df['percentile3'] = plotpercentiles['percentiles'][3]
    df['percentile4'] = plotpercentiles['percentiles'][4]
    df['prediction'] = plotpercentiles['percentiles'][2]
    df['mapstatus'] = pd.Series()
    df['gpcolour'] = pd.Series()

    for i in range(len(df)):
        if ((plotpercentiles['pbar'][i] > plotpercentiles['percentiles'][4][i])):
            df['mapstatus'][i] = 'high'
        elif ((plotpercentiles['pbar'][i] < plotpercentiles['percentiles'][4][i]) and (plotpercentiles['pbar'][i] > plotpercentiles['percentiles'][3][i])):
            df['mapstatus'][i] = 'medium'
        else:
            df['mapstatus'][i] = 'low'

    
    for i in range(len(df)):
        if ((plotpercentiles['pbar'][i] > plotpercentiles['percentiles'][4][i]) or (plotpercentiles['pbar'][i] < plotpercentiles['percentiles'][0][i])):
            df['gpcolour'][i] = 'red'
        elif ((plotpercentiles['pbar'][i] < plotpercentiles['percentiles'][3][i]) and (plotpercentiles['pbar'][i] > plotpercentiles['percentiles'][1][i])):
            df['gpcolour'][i] = 'palegreen'
        else:
            df['gpcolour'][i] = 'orange'


    if task_config['location'] == None:
        df['region'] = 'Nationwide'
    else:
        df['region'] = task_config['location']



    with s3.open(f"savsnet-agile-artefacts/archive/"+str(time.strftime("%Y-%m-%d"))+"/consult_data/harmonic_gp/plot_data/"+task_config['species']+"/"+task_config['mpc']+"/"+task_config['plot_filename']+".csv", "w") as f:
        df.to_csv(f)


