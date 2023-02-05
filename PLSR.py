#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:54:39 2023

@author: Manuel
# Modified from sources:
https://www.kaggle.com/code/phamvanvung/partial-least-squares-regression-in-python
https://stackoverflow.com/a/73405250/11611246
"""
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Production"

#-----------------------------------------------------------------------------|
# Import modules
## General
import os, argparse
import pandas as pd
import numpy as np
import pickle as pk
## Figures
import matplotlib.pyplot as plt
## Model training and quality assessment
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
## Continuum removal
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

#-----------------------------------------------------------------------------|
# Functions
def try_read(in_df, hdr = "infer", idx_col = None):
    '''Try different separatrs to read the input table data.'''
    df = pd.read_table(in_df, header = hdr, index_col = idx_col)
    possible_separators = ["\t", ";", ","]
    for separator in possible_separators:
        df_new = pd.read_table(in_df, sep = separator,
                               header = hdr, index_col = idx_col)
        df = df_new if len(df_new.columns) > len(df.columns) else df
    return df

def check_column_name(c, c_list):
    '''Check if a column (either key or int) is available in input data.'''
    if c not in c_list:
        c = c.strip()
        if c not in c_list:
            try:
                idx = int(c)
                mssg = "Assuming column index input for variable '{0}'." \
                    .format(c)
                raise Warning(mssg)
                c = c_list[idx]
            except:
                mssg = "Column '{0}' not found in y_in data set.".format(c)
                c = None
                raise LookupError(mssg)
    return c

def optimise_pls_cv(X_vals, y_vals, n_comp, crossval, mcreps,
                    CV_method = "ss"):
    '''Fit PLS regression model using cross-validation.'''
    # Define PLS object
    pls = PLSRegression(n_components = n_comp)

    # Cross-validation
    if CV_method.lower() == "ss":
        cv_split = ShuffleSplit(n_splits = mcreps, test_size = 1/crossval,
                                random_state = 0)
    elif CV_method.lower() == "lpo":
        from sklearn.model_selection import LeavePOut
        cv_split = LeavePOut(p = crossval)
    elif CV_method.lower() == "loo":
        from sklearn.model_selection import LeaveOneOut
        cv_split = LeaveOneOut()
    elif CV_method.lower() == "kfold":
        from sklearn.model_selection import KFold
        cv_split = KFold(n_splits = crossval)
    else:
        mssg = "Invalid argument: {0} for parameter CV_method. Use " + \
                "one in 'kfold': k-fold, 'loo': leave-one-out, 'lpo': " + \
                    "leave-p-out, 'ss': shuffle split."
        ValueError(mssg.format(CV_method))
    cvs = cross_validate(pls, X_vals, y_vals, cv = cv_split,
                         scoring = ["r2", "neg_mean_squared_error"])
    mean_r2_error = np.mean(cvs["test_r2"])
    test_mse = -np.mean(cvs["test_neg_mean_squared_error"])
    return pls, mean_r2_error, test_mse

def plot_metrics(vals, ylabel, objective, xticks, plt_dir = None):
    '''Plot accurracy metrics.'''
    plt.figure(figsize = (8, 4.5))
    with plt.style.context("ggplot"):
        plt.plot(xticks, np.array(vals), "-v", color = "blue", mfc = "blue")
        if objective == "min":
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], "P", ms = 10, mfc = "red")
        plt.xlabel("Number of PLS components")
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title("PLS")
        if plt_dir is not None:
            outfile = os.path.join(plt_dir, ylabel + ".pdf") if os.path. \
                isdir(plt_dir) else plt_dir
            plt.savefig(outfile)
            plt.close()
        else:
            plt.show()

#-----------------------------------------------------------------------------|
# Classes
class SpectralTable():
    '''Class to store and process spectral information.'''
    def __init__(self, data):
        table = pd.DataFrame(data)
        self.CR = False
        self.SF = False
        self.values = table
        self.sampleIDs = table.columns
        self.wavelengths = table.index.astype("float")
    
    @property
    def T(self):
        return self.values.T
    
    @property
    def array(self):
        return self.values.values
    
    @property
    def arrayT(self):
        return self.values.values.T
    
    def cut_spectrum(self, overwrite = True, *boundaries):
        '''Replace segments of the spectra by linear functions.
        boundaries = a list of lists of shape [min, max]'''
        out = self.values if overwrite else self.values.copy()
        for bound in boundaries:
            [lambda_min, lambda_max] = bound
            rows = out.index
            wls = self.wavelengths
            row_idx_min = np.squeeze(np.where(wls <= lambda_min)).max()
            row_idx_max = np.squeeze(np.where(wls >= lambda_max)).min()
            row_min = rows[row_idx_min]
            row_max = rows[row_idx_max]
            for name, values in out.iteritems():
                col = values.values
                m = (col[row_max] - col[row_min]) / (wls[row_max] -
                                                     wls[row_min])
                b = col[row_min] - (m * wls[row_min])
                col[row_min : row_max] = (m * wls[row_min]) + b
                out[name] = col
        return
    
    def remove_continuum(self, overwrite = True, show = False):
        '''Apply continuum removal to the spectra.'''
        if self.CR:
            print("Continuum already removed. Skip operation.")
        else:
            x = np.array(self.wavelengths)
            out = self.values if overwrite else self.values.copy()
            for name, values in out.iteritems():
                col = np.array(values.values)
                pts = np.concatenate((x.reshape(-1, 1), col.reshape(-1, 1)),
                                     axis = 1)
                augmented = np.concatenate([pts, [(x[0], np.min(col) - 1),
                                                  (x[-1], np.min(col) - 1)]],
                                           axis = 0)
                conv_hull = ConvexHull(augmented)
                continuum_points = pts[np.sort([v for v in conv_hull.vertices \
                    if v < len(pts)])]
                continuum_function = interp1d(*continuum_points.T)
                denominator = continuum_function(self.wavelengths)
                denominator[denominator == 0] = 1e-10
                cr = col / denominator
                out[name] = cr
                if show:
                    fig, axes = plt.subplots(2, 1, sharex = True)
                    axes[0].plot(x, col, label = "Spectrum")
                    axes[0].plot(*continuum_points.T, label = "Continuum")
                    axes[0].legend()
                    axes[1].plot(x, cr, label = "Spectrum / Continuum")
                    axes[1].legend()
                    plt.show()
            self.CR = True
        return

#-----------------------------------------------------------------------------|
# Get input variables
def parseArguments():
    '''Obtain input parameters from calling process.'''
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("in_x",
                        help = "Input spectral data.",
                        type = str)
    parser.add_argument("in_y",
                        help = "Input true values for predicted variable(s).",
                        type = str)
    parser.add_argument("out",
                        help = "Output directory.",
                        type = str)
    parser.add_argument("id_col",
                        help = "Sample ID column name (in_y table).",
                        type = str)
    # Optional arguments
    parser.add_argument("-vars", "--vars",
                        help = "Variable names (column names to use from" + \
                            "in_y). Separate multiple names by ';'.",
                        type = str, default = None)
    parser.add_argument("-mlv", "--mlv",
                        help = "Maximum number of latent variables.",
                        type = int, default = 15)
    parser.add_argument("-mcr", "--mcr",
                        help = "Monte-Carlo repetitions.",
                        type = int, default = 100)
    parser.add_argument("-cv", "--cv",
                        help = "Cross-validation split.",
                        type = str, default = 5)
    parser.add_argument("-cr", "--cr",
                        help = "Apply continuum removal.",
                        action = "store_true")
    parser.add_argument("-save_fig", "--save_fig",
                        help = "Export figures to output directory.",
                        action = "store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArguments()

in_x = args.in_x
in_y = args.in_y
cols = args.vars
id_col = args.id_col
mlv = args.mlv
mcr = args.mcr
cv = args.cv
out = args.out
cr = args.cr
save_fig = args.save_fig

col_list = cols.split(";")
os.makedirs(out, exist_ok = True)

#-----------------------------------------------------------------------------|
# Read X data
x_df = try_read(in_x, hdr = 0, idx_col = 0)
X = SpectralTable(x_df)
x_ids = X.sampleIDs

# Read y dataframe
y_df = try_read(in_y)

# Check whether integers have been handed instead of column names
y_col_names = list(y_df.columns)

if id_col not in y_col_names:
    try:
        idx = int(id_col)
        raise Warning("Assuming column index input for ID column.")
        id_col = y_col_names[idx]
    except:
        raise LookupError("ID column not found in y data set.")

# Check(/correct) column names for y data
for i, c in enumerate(col_list):
    catch_c = check_column_name(c, y_col_names)
    if catch_c != c:
        col_list[i] = catch_c

# Arrange y data
y_ids = list(y_df[id_col].values)

order = [i for i, item in enumerate(y_ids) if str(item) in x_ids]

#-----------------------------------------------------------------------------|
# Plot data
plt.figure(figsize = (8, 4.5))
wl = X.wavelengths
with plt.style.context("ggplot"):
    plt.plot(wl, X.values)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Reflectance [0, 10000]")
    if save_fig:
        outfile = os.path.join(out, "Spectrum.pdf")
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

if cr:
    plt.figure(figsize = (8, 4.5))
    X.remove_continuum()
    with plt.style.context("ggplot"):
        plt.plot(wl, X.values)
        plt.xlabel("Wavelengths (nm)")
        plt.ylabel("Hull Quotient [0, 1]")
    if save_fig:
        outfile = os.path.join(out, "Spectrum_CR.pdf")
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

#-----------------------------------------------------------------------------|
# Loop through y variables
report_vals = dict()

for i, column in enumerate(col_list):
    out_folder = os.path.join(out, column)
    os.makedirs(out_folder, exist_ok = True)
    
    ##------------------------------------------------------------------------|
    ## Read y data
    y = y_df[column].values.squeeze()[order]
    ## Mean y value for calculation of the relative Root Mean Squared Error
    y_mean = np.mean(y)
    
    ##------------------------------------------------------------------------|
    ## Test with varying number of components
    r2s = []
    mses = []
    try_latent_vars = np.arange(1, mlv)
    
    for n_comp in try_latent_vars:
        model, r2, mse = optimise_pls_cv(X_vals = X.arrayT,
                                         y_vals = y,
                                         n_comp = n_comp,
                                         crossval = cv,
                                         mcreps = mcr)
        r2s.append(r2)
        mses.append(mse)
    
    rmses = [np.sqrt(mse) for mse in mses]
    rrmses = [rmse / y_mean for rmse in rmses]
    
    fig_dir = None if save_fig == False else os.path.join(out_folder)
    plot_metrics(mses, "MSE", "min", try_latent_vars, fig_dir)
    plot_metrics(rmses, "RMSE", "min", try_latent_vars, fig_dir)
    plot_metrics(rrmses, "relative RMSE", "min", try_latent_vars, fig_dir)
    plot_metrics(r2s, "R2", "max", try_latent_vars, fig_dir)
    
    index_max_r2s = np.argmax(r2s)
    lv = try_latent_vars[index_max_r2s]
    
    ##------------------------------------------------------------------------|
    ## Optimise
    model, r2, mse = optimise_pls_cv(X_vals = X.arrayT,
                                     y_vals = y,
                                     n_comp = lv,
                                     crossval = cv,
                                     mcreps = mcr)
    metrics = {"R2" : r2,
               "MSE" : mse,
               "RMSE" : np.sqrt(mse),
               "RRMSE" : np.sqrt(mse) / y_mean}
    metrics_str = "R2: %0.4f, MSE: %0.4f" % (r2, mse)
    report_vals[column] = metrics
    
    model.fit(X.arrayT, y)
    
    model_dir = os.path.join(out_folder, column + "_fit.pkl")
    pk.dump(model, open(model_dir, "wb"))
    
    y_cv = model.predict(X.arrayT)
    
    plt.figure(figsize = (6, 6))
    with plt.style.context("ggplot"):
        plt.scatter(y, y_cv, color = "red")
        plt.plot(y, y, "-g", label = "Expected regression line")
        z = np.polyfit(y, y_cv, 1)
        plt.plot(np.polyval(z, y), y, color = "blue",
                 label = "Predicted regression line")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        if save_fig:
            outfile = os.path.join(out_folder, "Scatter_" + column + ".pdf")
            plt.savefig(outfile)
            plt.close()
        else:
            plt.plot()
    print(i)

#-----------------------------------------------------------------------------|
# Write report
report_file = os.path.join(out, "Report.txt")
with open(report_file, "w") as f:
    for model_i in col_list:
        f.write(model_i + "\n")
        tmp_dict = report_vals[model_i]
        for k in tmp_dict.keys():
            f.write(k + ": " + str(tmp_dict[k]) + "\n")
        f.write("\n")
