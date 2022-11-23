# efficient_frontiers
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
