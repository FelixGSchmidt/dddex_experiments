"""

...


"""

# Import utils
import numpy as np
import pandas as pd
import time
import copy
import datetime as dt
import joblib
import contextlib
from tqdm import tqdm
from scipy import stats

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from matplotlib.patches import PathPatch

# sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# dddex
from dddex.crossValidation import QuantileCrossValidation
from dddex.levelSetKDEx_univariate import LevelSetKDEx, LevelSetKDEx_NN
from dddex.wSAA import RandomForestWSAA, SampleAverageApproximation

# Import Gurobi
import gurobipy as gp
from gurobipy import GRB




    

#### ...
class Evaluation:

    """
    ...
    
    """

    ## Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """
        
        return None



    #### Function to load evaluation data
    def load_data(self, **kwargs):


        return None

    
    
    
    #### Function to pre-process evaluation data
    def preprocess_data(self, **kwargs):

        """

        ...


        """

        return None
    
    
    
    
    
    def plot_medians_(self, grid, x_var, y_var, hue_var, **kwargs):
        
        """
        
        ...
        
        
        """
    
        # Defaults
        linewidth = 1.0
        width=0.66
        fliersize=0
        whis=0

        # Box-plots
        grid.map_dataframe(
            sns.boxplot, x=x_var, y=y_var, hue=hue_var, 
            width=kwargs.get('width', width), fliersize=kwargs.get('fliersize', fliersize), 
            whis=kwargs.get('whis', whis), linewidth=kwargs.get('linewidth', linewidth), 
            palette=kwargs.get('palette', 'magma'))

        # Box-plot sizes
        fac = 0.75

        # iterating through Axes instances
        for ax in grid.axes.flatten():

            # iterating through axes artists:
            for c in ax.get_children():

                # searching for PathPatches
                if isinstance(c, PathPatch):

                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5*(xmin+xmax)
                    xhalf = 0.5*(xmax - xmin)

                    # setting new width of box
                    xmin_new = xmid-fac*xhalf
                    xmax_new = xmid+fac*xhalf
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        #if np.all(l.get_xdata() == [xmin, xmax]):
                        if (l.get_xdata()[0] == xmin) & (l.get_xdata()[1] == xmax):
                            l.set_xdata([xmin_new, xmax_new])
                            l.set_linewidth(3.0)
                            l.set_color('red')


        return grid
    
    
    
    
    
    
    
    def plot_totals_(self, grid, x_var, y_var, hue_var, **kwargs):
        
        """
        
        ...
        
        
        """
    
        # Point plots
        grid.map_dataframe(
            sns.pointplot, x=x_var, y=y_var, hue=hue_var, 
            dodge=0.5, join=False, markers='x',
            palette=kwargs.get('palette', 'magma'))





        return grid
    
    
    
    

    

    def plot_prescriptive_performance(self, plotData, grid_var, x_var, y_var, hue_var, 
                                      kind='medians', facet_h=6, facet_w=3, **kwargs):
        
        """
        
        ...
        
        
        """

        # Defaults
        linewidth = 1.0
        xlabel='overtime penalty'
        ylabel='cost relative to SAA'
        ylim=(-0.05, 1.85)

        # Setup facet grid
        grid = sns.FacetGrid(plotData, col=grid_var, height=facet_h, aspect=facet_w/facet_h)

        # Box-plots
        if kind=='medians':

            grid = self.plot_medians_(grid, x_var, y_var, hue_var, **kwargs)

        elif kind=='totals':

            grid = self.plot_totals_(grid, x_var, y_var, hue_var, **kwargs)

        else:

            return None


        # Reference line
        grid.refline(y=1.0, color='red', linewidth=kwargs.get('linewidth', linewidth), linestyle='--')

        # Facet borders
        for ax in grid.axes.flatten(): 
            for _, spine in ax.spines.items():
                spine.set_visible(True) 
                spine.set_color('black')
                spine.set_linewidth(1.0)

        # Axis labels
        for ax in grid.axes.flatten():

            ax.set_xlabel(xlabel=kwargs.get('xlabel', xlabel), fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel=kwargs.get('ylabel', ylabel), fontsize=10, fontweight='bold')
            ax.title.set_size(10)
            ax.title.set_weight('bold')
            ax.xaxis.set_label_position('bottom')


        # Legend
        grid.add_legend(ncol=1, loc='right')

        # Limits
        grid.set(ylim=kwargs.get('ylim', ylim))

        return grid.figure
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def differences(self, data, cost_var:str, copres_var:str, groups:list, 
                    test='unpaired', alternative='two-sided', n_hypotheses=1, **kwargs):

        """

        ...

        """


        # Initialize
        result = []

        # For each group in the data
        for grp, data_ in data.groupby(groups):

            models = list(data.model.unique())

            # For each model
            for model in models:

                # Models to compare to (all others)
                benchmarks = [i for (i, v) in zip(models, [model != m for m in models]) if v]

                # For all models to compare to
                for benchmark in benchmarks:

                    # Cost
                    cost_model = np.array(data_.loc[data_.model == model][cost_var])
                    cost_benchmark = np.array(data_.loc[data_.model == benchmark][cost_var])

                    # Prescriptive performance
                    copres_model_ = np.array(data_.loc[data_.model == model][copres_var])
                    copres_benchmark_ = np.array(data_.loc[data_.model == benchmark][copres_var])

                    # Differences
                    with np.errstate(divide='ignore'):

                        diffs_ = (
                            (cost_model == cost_benchmark) * 0 
                            + (cost_model != cost_benchmark) * (copres_model_ - copres_benchmark_)
                        )

                    # Remove inf / nan
                    diffs = diffs_[np.isfinite(diffs_)]
                    copres_model = copres_model_[np.isfinite(copres_model_) & np.isfinite(copres_benchmark_)]
                    copres_benchmark = copres_benchmark_[np.isfinite(copres_model_) & np.isfinite(copres_benchmark_)]

                    ## Paired test of differences (Wilcoxon Signed Rank Sum Test)
                    if test == 'paired':

                        # Mean of differences
                        mean_of_differences = np.mean(diffs)

                        # Median of differences
                        median_of_differences = np.median(diffs)

                        # Share of cases where model is better than benchmark
                        share_model_is_better = sum(diffs < 0) / len(diffs)

                        # Statictical significance
                        statistic, pvalue = stats.wilcoxon(diffs, alternative=alternative, nan_policy='raise')

                        # Store
                        res = {

                            'model': model,
                            'benchmark': benchmark,
                            'mean_of_differences': mean_of_differences,
                            'median_of_differences': median_of_differences,
                            'share_model_is_better': share_model_is_better,
                            'statistic': statistic,
                            'pvalue': pvalue,
                            'sig0001': '***' if pvalue < 0.001 / n_hypotheses else '',
                            'sig0010': '**' if pvalue < 0.01 / n_hypotheses else '',
                            'sig0050': '*' if pvalue < 0.05 / n_hypotheses else ''
                        }

                        # Append
                        result += [res]


                    ## Unpaired test of differences (Mann-Whitney U Test)
                    elif test == 'unpaired':

                        # Difference of means
                        difference_of_means = np.mean(copres_model) - np.mean(copres_benchmark)

                        # Difference of medians
                        difference_of_medians = np.median(copres_model) - np.median(copres_benchmark)

                        # Share of cases where model is better than benchmark
                        share_model_is_better = sum(diffs < 0) / len(diffs)

                        # Statictical significance
                        statistic, pvalue = stats.mannwhitneyu(copres_model, copres_benchmark, 
                                                               alternative=alternative, nan_policy='raise')

                        # Store
                        res = {

                            **dict(zip(groups, list(grp))),

                            **{
                                'model': model,
                                'benchmark': benchmark,
                                'difference_of_means': difference_of_means,
                                'difference_of_medians': difference_of_medians,
                                'share_model_is_better': share_model_is_better,
                                'statistic': statistic,
                                'pvalue': pvalue,
                                'sig0001': '***' if pvalue < 0.001 / n_hypotheses else '',
                                'sig0010': '**' if pvalue < 0.01 / n_hypotheses else '',
                                'sig0050': '*' if pvalue < 0.05 / n_hypotheses else ''
                            }
                        }

                        # Append
                        result += [res]

        # Result         
        return pd.DataFrame(result)