#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:54:39 2023

@author: Manuel

This script implements the PLS Regression as commandline callable Python code.
Based on scikit-learn, the implemented algorithm is NIPALS.
"""
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Production"

#-----------------------------------------------------------------------------|
# Import modules
## General
import os, argparse, random
import pandas as pd
import numpy as np
import pickle as pk
## Figures
import matplotlib.pyplot as plt
## Model training and quality assessment
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
## Continuum removal
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

#-----------------------------------------------------------------------------|
# Functions
def try_read(in_df, hdr = "infer", idx_col = None, sheet = 0):
    '''Try different separatrs to read the input table data.

    Parameters
    ----------
    in_df : str
        Full path to the input data (see pandas.read_table documentation for
        readable data formats).
    hdr : {int, list of int, None}, optional
        Either integer(s) indicating a row(s) of the data, "infer", or None.
        The default is "infer".
    idx_col : {int, str, sequence of int/str, False, None}, optional
        Integer/string indicating the column that contains row names of the
        input table, or False/None. The default is None.
    sheet : {str, int, or None}, optional
        Sheet name or position in case the input data is an Excel spreadsheet.
        The default is 0.

    Returns
    -------
    df : pandas.DataFrame
        An instance of type pandas.DataFrame.
    '''
    if os.path.splitext(in_df)[1] == ".xlsx":
        df = pd.read_excel(in_df, header = hdr, index_col = idx_col,
                           sheet_name = sheet)
    else:
        df = pd.read_table(in_df, header = hdr, index_col = idx_col)
        possible_separators = ["\t", ";", ","]
        for separator in possible_separators:
            df_new = pd.read_table(in_df, sep = separator,
                                   header = hdr, index_col = idx_col)
            df = df_new if len(df_new.columns) > len(df.columns) else df
    return df

def check_column_name(c, c_list):
    '''Check if a column (either key or int) is available in input data.

    Parameters
    ----------
    c : {int, str}
        Name or index of a column to search for.
    c_list : {list}
        List of column names from a pandas.DataFrame.

    Raises
    ------
    Warning
        Warn in case the first parameter is interreted as numeric.
    LookupError
        Raise error in case the first parameter cannot be found in the list of
        columns (neither literal, nor as an index).

    Returns
    -------
    c : {str, None}
        Column name within the list of column names that best matches the
        search criteria. None, if no matching column name can be found.
    '''
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

def test_input(x0, x1):
    x2, x3 = np.squeeze(x0), np.squeeze(x1)
    if x2.shape != x3.shape:
        mssg = "Input shapes {0} and {1} do not match after removal of " + \
            "length zero dimensions."
        raise Warning(mssg(x2.shape, x3.shape))
        if x2.shape == x3.T.shape:
            x4, x5 = x2, x3.T
            raise Warning("Attempt to transform x1.")
    else:
        x4, x5 = x2, x3
    return x4, x5

def r_squared(y_true, y_pred):
    y_true, y_pred = test_input(y_true, y_pred)
    cc = np.corrcoef(y_true, y_pred)
    r2 = cc[1][0] ** 2
    return r2

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = test_input(y_true, y_pred)
    se = np.square(np.subtract(y_true, y_pred))
    mse = se.mean()
    return mse

def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = test_input(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def relative_root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = test_input(y_true, y_pred)
    num = np.sum(np.square(y_true - y_pred)) / len(y_true)
    den = np.sum(np.square(y_pred))
    squared_error = num / den
    rrmse = np.sqrt(squared_error)
    return rrmse

metrics = {"r2" : r_squared,
           "MSE" : mean_squared_error,
           "RMSE" : root_mean_squared_error,
           "relRMSE" : relative_root_mean_squared_error
           }

def cross_validate_plsr(X_vals, y_vals, n_comp, crossval, mcreps,
                    metrics = metrics):
    '''Fit PLS regression model using Monte-Carlo Cross-Validation.

    Parameters
    ----------
    X_vals : numpy.array
        Array containing the X (predictor) values.
    y_vals : numpy.array
        Array containing the y (dependent variable) values.
    n_comp : int
        Number of components/latent variables to use in the model.
    crossval : {int, float}
        Cross-validation split.
    mcreps : int
        Number of Monte-Carlo repetitions.

    Returns
    -------
    pls : sklearn.cross_decomposition.PLSRegression
        Model fitted to the data.
    mean_r2_error : float
        Mean r-squared error across cross-validation sets.
    test_mse : float
        Average mean squared error across cross-validation sets.
    '''
    # Check input data shape
    if len(X_vals) != len(y_vals):
        mssg = "Dimension mismatch. Argument X_vals is of shape {0} while" + \
            " argument y_vals is of shape {1}. Expected equal number of rows."
        raise ValueError(mssg.format(X_vals.shape, y_vals.shape))
    
    # Set variables
    print("Number of latent variables is {0}.".format(n_comp))
    n_samples = len(y_vals)
    samples = np.array(list(range(n_samples)))
    test_size = int(n_samples / crossval) if crossval > 1. else int(
        n_samples * crossval)
    train_size = n_samples - test_size
    print("Train size: {0}, test size {1}.".format(train_size, test_size))
    
    cvs = {}
    for key in metrics:
        cvs[key] = []
    
    # Monte-Carlo repetitions
    for mcrep in range(mcreps):
        # Train model
        train_set, test_set = train_test_split(samples, test_size = test_size)
        pls = PLSRegression(n_components = n_comp)
        X_train, y_train = X_vals[train_set,:], y_vals[train_set]
        pls.fit(X_train, y_train)
        
        # Predict test data
        X_test, y_test = X_vals[test_set,:], y_vals[test_set]
        y_pred = pls.predict(X_test)
        
        # Calculate metrics
        for key in metrics:
            metric = metrics[key]
            cvs[key].append(metric(y_test, y_pred))
        del pls
    mssg = "Shape X: {0} and shape y: {1}"
    print(mssg.format(X_train.shape, y_train.shape))
    
    # Average metrics
    scores_out = {}
    for key in metrics:
        scores_out[key] = np.mean(np.array(cvs[key]))
    
    return scores_out

def plot_metrics(vals, ylabel, objective, xticks, plt_dir = None):
    '''Plot accurracy metrics.

    Parameters
    ----------
    vals : {numpy.array, list}
        List or array containing the values to plot against xticks.
    ylabel : str
        Y-axis label for the plot.
    objective : {"min", "max"}
        Whether a minimum or a maximum value of the metric indicates the best
        model fit.
    xticks : {list, numpy.array}
        List or array indicating the x-axis tick positions (numbers of
        components).
    plt_dir : {str, None}, optional
        Directory to save figures to. The default is None.

    Returns
    -------
    None.
    '''
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
    '''Class to store and process spectral information.
    
    Parameters
    ----------
    data: {numpy.ndarray, pandas.DataFrame, iterable, dict}
        Any object transferrable into a pandas.DataFrame.
    
    Attributes
    ----------
    CR: bool
        Information on whether continuum removal was applied to the data.
    values: pandas.DataFrame
        Values as a pandas.DataFrame.
    T: pandas.DataFrame
        Values as a pandas.DataFrame after transformation.
    array: numpy.array
        Values as a numpy.array.
    arrayT: numpy.array
        Values as a numpy.array after transformation.
    sampleIDs: pandas.DataFrame.columns
        Column names of the object.
    wavelengths: numpy.array
        Row names of the object (wavelengths of the spectral data) as "float"
        type.
    
    Methods
    -------
    cut_spectrum():
        Replace segments of the spectra by a linear function.
    remove_continuum():
        Apply continuum removal to the spectra.
    '''
    def __init__(self, data):
        table = pd.DataFrame(data)
        self.CR = False
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
        boundaries = a list of lists of shape [min, max]

        Parameters
        ----------
        overwrite : bool, optional
            Overwrite the original pandas.DataFrame. The default is True.
        *boundaries : (nested) list
            (List of) boundary values (shape: [min, max]) to indicate
            wavelength segments that are to be replaced.

        Returns
        -------
        output: {pandas.DataFrame, None}
            Dataframe after replacing wavelength segments or None, if
            overwrite is True.
        '''
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
        if overwrite:
            output = None
        else:
            output = out
        return output
    
    def remove_continuum(self, overwrite = True, show = False):
        '''Apply continuum removal to the spectra.

        Parameters
        ----------
        overwrite : boot, optional
            Overwrite the original pandas.DataFrame. The default is True.
        show : bool, optional
            Plot and show figure of the updated spectra. The default is False.

        Returns
        -------
        output: {pandas.DataFrame, None}
            Dataframe after replacing wavelength segments or None, if
            overwrite is True.
        '''
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
        if overwrite:
            output = None
        else:
            output = out
        return output

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
    parser.add_argument("-sheet_x", "--sheet_x",
                        help = "Input sheet index if in_x is an Excel sheet.",
                        type = str, default = "0")
    parser.add_argument("-sheet_y", "--sheet_y",
                        help = "Input sheet index if in_y is an Excel sheet.",
                        type = str, default = "0")
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
    id_col = args.id_col
    cols = args.vars
    mlv = int(args.mlv)
    mcr = int(args.mcr)
    cv = float(args.cv)
    out = args.out
    sheet_x = int(args.sheet_x) if args.sheet_x.isnumeric() else args.sheet_x
    sheet_y = int(args.sheet_y) if args.sheet_y.isnumeric() else args.sheet_y
    cr = args.cr
    save_fig = args.save_fig
    prog_bar = True
    
    col_list = None if cols is None else cols.split(";")
    os.makedirs(out, exist_ok = True)
    
    #-------------------------------------------------------------------------|
    # Read X data
    x_df = try_read(in_x, hdr = 0, idx_col = 0, sheet = sheet_x)
    X = SpectralTable(x_df)
    x_ids = X.sampleIDs
    
    # Read y dataframe
    y_df = try_read(in_y, sheet = sheet_y)
    
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
    if col_list is None:
        col_list = y_col_names.copy()
        col_list.remove(id_col)
    else:
        for i, c in enumerate(col_list):
            catch_c = check_column_name(c, y_col_names)
            if catch_c != c:
                col_list[i] = catch_c
    
    # Arrange y data
    y_ids = list(y_df[id_col].values)
    
    order = [i for i, item in enumerate(y_ids) if str(item) in x_ids]
    
    ## Export a new y df to tmp if y needs to be re-ordered
    if order != list(range(len(y_ids))):
        y_df.reindex(order)
    
    #-------------------------------------------------------------------------|
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
    
    #-------------------------------------------------------------------------|
    # Loop through y variables
    report_vals = dict()
    report_vals_production_lvl_model = dict()
        
    for i, column in enumerate(col_list):
        out_folder = os.path.join(out, column)
        os.makedirs(out_folder, exist_ok = True)
        
        ##--------------------------------------------------------------------|
        ## Read y data
        y = y_df[column].values.squeeze()[order]
        ## Mean y value for calculation of the relative Root Mean Squared Error
        y_mean = np.mean(y)
        y_min = np.min(y)
        y_max = np.max(y)
        
        ##--------------------------------------------------------------------|
        ## Test with varying number of components
        cv_scores = {}
        for key in metrics:
            cv_scores[key] = []
        
        try_latent_vars = np.arange(1, mlv + 1)
        
        for n_comp in try_latent_vars:
            scores = cross_validate_plsr(X_vals = X.arrayT,
                                         y_vals = y,
                                         n_comp = n_comp,
                                         crossval = cv,
                                         mcreps = mcr)
            for key in scores:
                cv_scores[key].append(scores[key])
        
        fig_dir = None if save_fig == False else os.path.join(out_folder)
        for key in cv_scores:
            obj = "max" if key == "r2" else "min"
            plot_metrics(cv_scores[key], key, obj, try_latent_vars, fig_dir)
        
        ## Find metric optimum
        index_max_r2s = np.argmax(cv_scores["r2"])
        index_min_mse = np.argmin(cv_scores["MSE"])
        index_min_rmse = np.argmin(cv_scores["RMSE"])
        index_min_rrmse = np.argmin(cv_scores["relRMSE"])
        
        ## Get optimum number of latent variables
        index_best = index_min_rrmse
        nlv = try_latent_vars[index_best]
        
        ##--------------------------------------------------------------------|
        ## Get best cross-validation scores
        best_scores = dict()
        for key in cv_scores:
            best_scores[key] = cv_scores[key][index_best]
        
        ## Prepare cross-validation scores for best model for report
        report_vals[column] = best_scores
        
        ##--------------------------------------------------------------------|
        ## Fit model with optimum number of latent variables and  all data
        ## points
        ## = "production-level" model
        model = PLSRegression(n_components = nlv)
        model.fit(X.arrayT, y)
        
        XL, yl, XS, YS, beta = model.x_scores_, model.y_scores_, \
            model.x_loadings_, model.y_loadings_, model.coef_
        PCTVAR = model.score(X.arrayT, y)
        
        ## Export model
        model_dir = os.path.join(out_folder, column + "_fit.pkl")
        pk.dump(model, open(model_dir, "wb"))
        
        ## Export beta scores
        coefs_dir = os.path.join(out_folder, column + "_coefs.pkl")
        coef_dict = {"x_mean" : model._x_mean,
             "x_std" : model._x_std,
             "y_mean" : model._y_mean,
             "coefs" : model.coef_}
        pk.dump(coef_dict, open(coefs_dir, "wb"))
        
        ## Predict y using final model
        y_pred = model.predict(X.arrayT)
        
        ## Prepare goodness-of-fit metrics of production-lvl model for export
        production_lvl_scores = dict()
        production_lvl_scores["Latent variables"] = nlv
        for key in metrics:
            metric = metrics[key]
            production_lvl_scores[key] = metric(y, y_pred)
        
        report_vals_production_lvl_model[column] = production_lvl_scores
        
        ##--------------------------------------------------------------------|
        ## Plot final model true vs prediction scatter plot
        plt.figure(figsize = (6, 6))
        with plt.style.context("ggplot"):
            plt.scatter(y, y_pred, color = "red")
            plt.plot(y, y, "-g")
            plt.plot(y, y, "-g", label = "Expected regression line")
            z = np.polyfit(y, y_pred, 1)
            plt.plot(np.polyval(z, y), y, color = "blue",
                     label = "Predicted regression line")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.legend()
            if save_fig:
                outfile = os.path.join(out_folder, "Scatter_" + column +
                                       ".pdf")
                plt.savefig(outfile)
                plt.close()
            else:
                plt.plot()
        
        del model, scores, cv_scores
        print(i)
    
    #-------------------------------------------------------------------------|
    # Write report
    header_string = "# NIPALS algorithm PLS regression"
    header_string = f"{header_string:-<80}" + "\n"
    header_string += "Regression report for {0} variables\nVariable names" + \
        ": {1}\nContinuum removal: {2}\n\n"
    section_sep = "-" * 80 + "\n"
    
    report_file = os.path.join(out, "Report.txt")
    with open(report_file, "w") as f:
        '''Write header'''
        f.write(header_string.format(int(i + 1), ", ".join(col_list), str(cr)))
        '''Report cross-validation results'''
        f.write(section_sep + "Cross-validation metrics\n")
        for model_i in col_list:
            f.write(model_i + "\n")
            tmp_dict = report_vals[model_i]
            for k in tmp_dict.keys():
                f.write(k + ": " + str(tmp_dict[k]) + "\n")
            f.write("\n")
        '''Report final model metrics'''
        f.write(section_sep + "Production-level model goodness of fit\n")
        for model_i in col_list:
            f.write(model_i + "\n")
            tmp_dict = report_vals_production_lvl_model[model_i]
            for k in tmp_dict.keys():
                f.write(k + ": " + str(tmp_dict[k]) + "\n")
            f.write("\n")
#-----------------------------------------------------------------------------|
# Predict using only the exported coefs dictionary:
'''
def predict_Python_PLS(X, coefs):
    """Predict targets of given samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples.

    copy : bool, default=True
        Whether to copy `X` and `Y`, or perform in-place normalization.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Returns predicted values.

    Notes
    -----
    This call requires the estimation of a matrix of shape
    `(n_features, n_targets)`, which may be an issue in high dimensional
    space.
    """
    # Normalize
    X -= coefs["x_mean"]
    X /= coefs["x_std"]
    Ypred = np.dot(X, coefs["coefs"])
    return Ypred + coefs["y_mean"]
'''
