#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:38:21 2022

Module containing functions that grab stock data from Alpha Vantage
and use historical returns (or provided expected returns) to create
a Resampled Efficient Frontier for the asset portfolio. The idea
of resampled frontiers comes from Michaud & Michaud 2008 and produces
smoother, more conservative portfolio transitions from one risk level
to another compared to classical mean-variance efficient frontiers from
Markowitz 1956.

Note that the frontier resampler and plot functions can be used without
querying Alpha Vantage. This module was written in part to get experience
pulling stock data from Alpha Vantage but any array of asset returns,
standard deviations, and correlations will work for the efficient 
frontier calculations. 

Functions
---------
get_stock_info
frontier_resampler
ref_plots
    
Package dependencies
--------------------
numpy
pandas
alpha_vantage.timeseries
cvxopt
matplotlib

@author: Wilson Cauley
"""

import time
import numpy as np
import pandas as pd
from datetime import date as dt
from alpha_vantage.timeseries import TimeSeries
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
import matplotlib
import matplotlib.pyplot as plt


def get_stock_info(av_api_key: str, tickers: list[str], start_date='2017-01-01',
                   end_date=None):

    """Retrieve annualized stock returns, standard deviations, and correlations
    from Alpha Vantage.
    
    Parameters
    ----------
    av_api_key : str
        Alpha Vantage API key
    tickers : string
        A list of srings containing the stock ticker abbreviations of the
        desired stocks. Ex. tickers = ['TSLA','GOOGL','AAPL']
    start_date : str, optional
        'YYYY-MM-DD' string specifying the date to start the retrieval. Default
        is the very arbitrary date '2017-01-01'.
    end_date : str, optional
        'YYYY-MM-DD' string specifying the date to end the retrieval. Default
        is the current date.
        
    Returns
    -------
    ann_returns : float
        List of annualized returns for each of the stocks in tickers        
    stddev_returns :  float
        List of the standard deviation of the returns for the stocks in tickers
    stock_corr : pandas dataframe
        Dataframe containing the correlations between the returns for the
        stocks in TICKERS
    
    """
    
    # Instance the AV API with the provided key
    app = TimeSeries(av_api_key)

    # Define some variables
    nstocks = len(tickers)
    stock_data = pd.DataFrame()
    time_series = app.get_monthly_adjusted(tickers[0])
    dates = list(time_series[0].keys())

    # Check for endate
    if end_date == None:
        end_date = str(dt.today())

    # Set up stock_data dataframe
    stock_data['Date'] = dates
    stock_data = stock_data[(stock_data['Date'] > start_date) &
                            (stock_data['Date'] < end_date)]
    stock_data.reset_index(inplace=True)
    tint = 60./5.+.5  # only 5 calls per minute to Alpha Vantage

    # Loop through stocks, pull out closing prices, and add to the big dataframe
    for tickr in tickers:

        # Get the monthly adjusted numbers for each stock from Alpha Vantage
        time_series = app.get_monthly_adjusted(tickr)
        dates = list(time_series[0].keys())
        df = pd.DataFrame(dates, columns=['dates'])

        # Loop through dictionary and grab adjusted closing price
        pclose = []
        for date in dates:
            pclose.append(float(time_series[0][date]['5. adjusted close']))

        # Add closing price to dummy dataframe and limit to > start_date
        df[tickr] = pclose
        df = df[(df['dates'] > start_date) & (df['dates'] < end_date)]
        df.reset_index(inplace=True)

        # Now add current closing price to stock_data
        stock_data[tickr] = df[tickr]

        # Now calculate monthly returns in decimal form, not percent
        pvec = stock_data[tickr].to_numpy()
        rvec = np.zeros(len(pvec))
        rvec[:-1] = ((pvec[:-1]-pvec[1:])/pvec[1:])
        stock_data[tickr+'_returns'] = rvec

        # Have to wait due to limited queries per minute...
        if tickr != tickers[nstocks-1]:
            print(f'Finished with {tickr}, waiting to query next stock')
            time.sleep(tint)
        else:
            print('Done querying stocks')

    # Calculate annualized returns and standard deviation of each stock. Standard deviations
    # are multiplied by sqrt(12) because these are monthly returns but the expected returns
    #vector is annualized
    ann_returns = []
    stddev_returns = []
    tdelta = (pd.to_datetime(stock_data['Date'][0])-pd.to_datetime(
        stock_data['Date'][len(stock_data['Date'])-1])).days
    year_exp = 1.0/(tdelta/365.0)
    for tickr in tickers:
        monthly_returns = stock_data[tickr+'_returns'].to_numpy()
        ann_returns.append(np.prod(monthly_returns+1)**year_exp-1)
        stddev_returns.append(np.std(monthly_returns)*np.sqrt(12))

    stock_corr = stock_data[tickers].corr()

    # Return the useful stuff
    return ann_returns, stddev_returns, stock_corr


def frontier_resampler(expected_returns, expected_stddev, stock_corrs,
                       ret_sig_level=0.2, sig_sig_level=0.05,
                       constraints=np.zeros(1)):

    """Calculate a resampled efficient frontier for the provided assets.
    
    Parameters
    ----------
    expected_returns : float
        List of the expected returns of each asset in the portfolio, should
        be given in decimal form, not percent
    expected_stddev : float
        List of the known standard deviation of historical returns for the
        assets in the portfolio
    stock_corrs : pandas dataframe
        Dataframe containing the correlations between each of the assets in
        the portfolio
    ret_sig_level : float, optional
        Fractional uncertainty level for expected_returns. Default value is
        0.2 (20% uncertainty). Can also be a list of the same length as
        expected_returns if a different value is desired for each asset
    sig_sig_level : float, optional
        Fractional uncertainty level for expected_stddev. Default value is
        0.05 (5% uncertainty). Can also be a list of the same length as
        expected_stddev if a different value is desired for each asset
    constraints : numpy array, optional
        NumPy array of dimension nassets x 2 where each row must contain a 
        lower and upper bound for the portfolio weights of that asset. For
        example, if asset i is required to have a minimum weight of 10% and
        a maximum weight of 30%, the i-th row of constrains should be [0.1,.3].
        Default is to assume no constraints and let the optimization procedure
        determine the portfolio composition.

    Returns
    -------
    risks_orig : float
        Array of the standard deviations for the original efficient frontier
    returns_orig : float
        Array of the expected returns for the original efficient frontier
    risk_resamp : float
        Array of the standard deviations for the resampled efficient frontier
    returns_resamp : float 
        Array of the expected returns of the resampled efficient frontier
    resamp_comp : float
        Array of the portfolio weights for each asset at each value of
        risk_resamp
    
    """
    
    # Construct the covariance matrix
    nstocks = len(expected_returns)
    a = np.zeros((nstocks, nstocks), float)
    np.fill_diagonal(a, expected_stddev)
    diag = pd.DataFrame(data=a, index=stock_corrs.index,
                        columns=stock_corrs.columns)
    diag_cor = diag.dot(stock_corrs)
    cov_matrix_orig = matrix(diag_cor.dot(diag).values.tolist())
    ann_returns = matrix(expected_returns)

    """
    Now we have our vector of expected returns, ann_returns, and a covariance matrix
    describing the covariance between the stock price timeseries. Before optimizing the 
    portfolio we need to construct our constraint matrices. We are keeping it simple
    and only forcing the portfolio components to sum to 1, plus no short-selling.
    These are linear constraints on the problem. G constrains all portfolio weights
    to be > 0 (or all negative weights to be < 0 for this problem; they are equivalent),
    and A constrains the sum of the weights to be equal to 1. Once our matrices are
    defined we plug them into the QP solver (see CVXOPT documentation for more details:
    https://cvxopt.org/examples/book/portfolio.html) 
    """

    """
    If upper and lower bounds for asset weights are provided in CONSTRAINTS
    then modify the G and h matrices appropriately. The G matrix should be
    m x n where m = # of constraints and n = number of assets. h will have 
    size m. Note that in this case one constraint for each asset is n TOTAL
    constraints. So for example if each asset has an upper and lower bound
    then there will be 2*n total constraints.
    """
    if len(constraints) == nstocks:
        G_lower = -np.eye(nstocks)
        G_upper = np.eye(nstocks)
        G = matrix(np.concatenate([G_lower, G_upper]))
        h = matrix(np.concatenate((-constraints[:, 0], constraints[:, 1])))
    else:
        G = matrix(0.0, (nstocks, nstocks))
        G[::nstocks+1] = -1.0
        h = matrix(0.0, (nstocks, 1))

    A = matrix(1.0, (1, nstocks))
    b = matrix(1.0)

    # Mus are > 0 values that determine relative risk: mu = 0 is minimum risk
    # and mu -> infinity is maximum risk. Below we optimize for the classical
    # efficient frontier
    N = 300
    mus = [10**(5.0*t/N-1.0) for t in range(N)]
    options['show_progress'] = False
    xs = [qp(mu*cov_matrix_orig, -ann_returns, G, h, A, b)['x'] for mu in mus]
    returns_orig = [dot(ann_returns, x) for x in xs]
    risks_orig = [np.sqrt(dot(x, cov_matrix_orig*x)) for x in xs]

    """
    Now we do the resampling part: by assuming some uncertainty in our expected 
    returns, we re-run the optimization after randomly sampling the expected 
    return distribution of each asset assuming normality about the mean expected 
    return. We then average the asset composition at each risk level to get the 
    resampled efficient frontier.
    """

    sig_returns = ret_sig_level*np.array(ann_returns)
    sig_sigs = sig_sig_level*np.array(expected_stddev)

    n_sims = 500
    resamp_comp = np.array((nstocks, N))
    for i in range(n_sims):

        # Find new random returns and variance
        ret_change = matrix(np.random.randn(nstocks)*np.transpose(sig_returns))
        new_returns = ann_returns + ret_change.T
        new_stddev = expected_stddev + np.random.randn(nstocks)*sig_sigs

        # Reconstruct the covariance matrix with new std dev estimates
        #a = np.zeros((nstocks,nstocks),float)
        np.fill_diagonal(a, new_stddev)
        diag = pd.DataFrame(data=a, index=stock_corrs.index,
                            columns=stock_corrs.columns)
        diag_cor = diag.dot(stock_corrs)
        cov_matrix = matrix(diag_cor.dot(diag).values.tolist())

        # Now do the optimization again with new returns and new cov matrix
        xs = [qp(mu*cov_matrix, -new_returns, G, h, A, b)['x'] for mu in mus]
        returns = [dot(new_returns, x) for x in xs]
        risks = [np.sqrt(dot(x, cov_matrix*x)) for x in xs]

        # Save all return and risk vectors in an array for plotting
        if i == 0:
            all_returns = returns
            all_risks = risks
        else:
            all_returns = np.vstack((all_returns, returns))
            all_risks = np.vstack((all_risks, risks))

        # Need to average the portfolio weights at each MU across all simulations
        # to create the resampled efficient frontier. How to do this...
        if i == 0:
            # Initiate the resampled efficient frontier composition array
            resamp_comp = np.reshape(np.transpose(
                np.array([np.array(xs[i]) for i in range(N)])), (nstocks, N))
        else:
            # Define weights for the average between already computed compositions and current iteration
            wt0 = float(i/(i+1))
            wt1 = 1.0 - wt0
            resamp_comp = wt0*resamp_comp + wt1 * \
                np.reshape(np.transpose(
                    np.array([np.array(xs[i]) for i in range(N)])), (nstocks, N))

        # Once we've simulated our efficient frontiers and averaged the portfolio compositions,
        # find the final resampled risk and return vectors.
        risk_resamp = [np.sqrt(resamp_comp[:, i].dot(np.array(cov_matrix_orig).dot(
            np.transpose(resamp_comp[:, i])))) for i in range(N)]
        returns_resamp = [np.transpose(np.array(ann_returns)).dot(
            np.transpose(resamp_comp[:, i])) for i in range(N)]

    return risks_orig, returns_orig, risk_resamp, returns_resamp, resamp_comp


def ref_plots(tickers, ann_returns, stddev_returns, risks_orig, returns_orig, 
              risk_resamp, returns_resamp, resamp_comp, 
              ef_file_name='efficient_frontiers.pdf',
              comp_file_name='portfolio_compositions.pdf'):

    """Generate some useful resampled efficient frontier plots, in particular
    1. a comparison of the classical efficient frontier and the resampled
    efficient frontier; 2. a portfolio composition plot showing the distribution
    of assets as a function of risk (portfolio standard deviation).
    
    Parameters
    ----------
    tickers : string
        A list of srings containing the stock ticker abbreviations of the
        desired stocks. Ex. tickers = ['TSLA','GOOGL','AAPL']
    ann_returns : float
       List of annualized returns for each of the stocks in tickers       
    stddev_returns :  float
       List of the standard deviation of the returns for the stocks in tickers
    risks_orig : float
        Array of the standard deviations for the original efficient frontier
    returns_orig : float
        Array of the expected returns for the original efficient frontier
    risk_resamp : float
        Array of the standard deviations for the resampled efficient frontier
    returns_resamp : float
        Array of the expected returns for the original efficient frontier
    resamp_comp : float
        Array of the portfolio composition at each value of risk_resamp
    ef_file_name : string, optional
        String of the desired filename for the efficient frontier plot
    comp_file_name : string, optional
        String of the desired filename for the portfolio composition plot
        
    Returns
    -------
    ef_file_name and comp_file_name are saved as PDFs.
    
    """
    
    nstocks = len(ann_returns)

    # Determine plot ranges based on limits of vectors
    xmod = .05
    xmax = np.maximum(max(stddev_returns), max(risks_orig))*(1.0+xmod)
    xmin = np.minimum(min(stddev_returns), min(risks_orig))*(1.0-xmod)
    ymod = .05
    ymax = np.maximum(max(ann_returns), max(returns_orig))*(1.0+ymod)
    ymin = 0.0  # don't show negative returns at the moment

    # Determine text offset
    toff_x = .0075*(xmax-xmin)
    toff_y = .0075*(ymax-ymin)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(risk_resamp, returns_resamp, color='r', label='Resampled EF')
    ax.plot(risks_orig, returns_orig, 'b', label='Mean-variance EF')
    ax.plot(stddev_returns, ann_returns, 'go', alpha=.4)

    for i in range(nstocks):
        ax.text(stddev_returns[i]+toff_x, ann_returns[i]+toff_y, tickers[i])

    ax.set_xlabel('Risk (standard deviation)', fontsize=20)
    ax.set_ylabel('Return', fontsize=20)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, ymax])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title('Resampled efficient frontier', fontsize=20)

    ax.legend(loc='upper left', fontsize=20)

    fig.savefig(ef_file_name)

    # Now make composition rainbow plot
    cmap = matplotlib.cm.get_cmap('jet', nstocks)
    #color_nums = np.arange(0.0,1.0,1.0/nstocks) + .5/nstocks

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(nstocks):

        if i == 0:
            ax.fill_between(risk_resamp, resamp_comp[0, :], color=cmap(i))
        else:
            iy1 = i+1
            ax.fill_between(risk_resamp, np.sum(resamp_comp[0:iy1, :], axis=0),
                            y2=np.sum(resamp_comp[0:i, :], axis=0), color=cmap(i))

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Risk (standard deviation)', fontsize=20)
    ax.set_ylabel('Composition fraction', fontsize=20)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([min(risk_resamp), max(risk_resamp)])
    ax.set_title('Portfolio Composition vs. Risk', fontsize=20)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax, aspect=10,
                        ticks=np.linspace(.5/nstocks, 1-.5/nstocks, nstocks))
    cbar.ax.set_yticklabels(tickers)

    fig.savefig(comp_file_name)
