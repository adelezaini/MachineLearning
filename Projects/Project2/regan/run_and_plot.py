# The MIT License (MIT)
#
# Copyright © 2021 Fridtjof Gjengset, Adele Zaini, Gaute Holen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import matplotlib.pyplot as plt
from typing_extensions import runtime
from regan import *
import numpy as np
from time import time
import os

def run_CV(k,x,y,z,solver,m, lmd = 10**(-12)):
    complexity = []
    MSE_train_set = []
    MSE_test_set = []
    runtime_set = []
    for i in range(2,m):
        t_0 = time()
        X = create_X(x, y, i)
        MSE_train, MSE_test = cross_validation(k,X,z,solver=solver,lmd=lmd)
        complexity.append(i)
        MSE_train_set.append(MSE_train)
        MSE_test_set.append(MSE_test)
        runtime_set.append((time()-t_0)/k)

    return complexity,MSE_train_set,MSE_test_set, runtime_set


def run_plot_compare(datapoints, n_resampling, N = 50, plot=False, lmd=10**(-12), k = 5, poly_degree = 10, plot_runtime=True, saveplots=False, foldername = 'Figures', title="Terrain"):
    """[summary]

    Args:
        datapoints ([type]): [description]
        title ([type]): [description]
        N (int, optional): [description]. Defaults to 50.
        plot (bool, optional): [description]. Defaults to False.
        n_lambdas (int, optional): [description]. Defaults to 20.
        k (int, optional): [description]. Defaults to 5.
        poly_degree (int, optional): [description]. Defaults to 10.
    """

    #----------------------------------------------------------------------------
    #                         Storing figures
    #----------------------------------------------------------------------------
    """if saveplots and plot:
        path = 'Proj1_Plots'
        foldername = title.replace(' ','')
        try:
            os.mkdir(path+'/'+foldername)
        except FileExistsError:
            print("Dir already exists")
        path = path+'/'+foldername+'/'
    """
    path = "../reports/Proj1_Plots/"+foldername+"/"

    #----------------------------------------------------------------------------
    #                           Preparing the dataset
    #----------------------------------------------------------------------------
    #Plot datapoints   
    m = poly_degree # polynomial order
    datapoints = datapoints[:N,:N]
    z = datapoints

    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(z)[0])
    y = np.linspace(0,1, np.shape(z)[1])
    x,y = np.meshgrid(x,y)
    z = z.ravel()

    if plot:
        fig, ax = plt.subplots(figsize = ( 10, 7))
        im = ax.imshow(datapoints, cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im)
        if saveplots: fig.savefig(path+"Terrain.png")

    solvers = ['OLS','RIDGE','LASSO']


    #----------------------------------------------------------------------------
    #                           Bootstrap
    #----------------------------------------------------------------------------

    BS_bias = []
    BS_error = []
    BS_variance = []
    BS_runtimes = []
    for solver in solvers:
        error, bias, variance = bias_variance_complexity(x,y,z,m,n_resampling=n_resampling,plot=False,solver=solver,lmd=lmd)
        BS_error.append(error)
        BS_bias.append(bias)
        BS_variance.append(variance)

    if plot:
        fig3, ax3 = plt.subplots(figsize = ( 10, 7))
        #ax3.set_yscale('log')
        #ax4 = ax3.twinx()
        BS_complexity = [x for x in range(0,m+1)]
        lns = []
        for i in range(len(solvers)):
            color = next(ax3._get_lines.prop_cycler)['color']
            lns.append(ax3.plot(BS_complexity,BS_error[i], label =solvers[i]+ " error", color=color))
            lns.append(ax3.plot(BS_complexity,BS_bias[i], label =solvers[i]+" bias$^2$", color=color, linestyle = 'dashed'))
            lns.append(ax3.plot(BS_complexity,BS_variance[i], label =solvers[i]+ " variance", linestyle = 'dotted', color=color ))

        lns = [l[0] for l in lns]
        labs = [l.get_label() for l in lns]
        ax3.legend(lns,labs)

        ax3.set_xlabel("complexity")
        ax3.set_ylabel("")
        #ax4.set_ylabel("variance")
        ax3.set_title(f"Bias-variance tradeoff analysis as a function of complexity from bootstrap resampling\n resampling = {n_resampling} times, and $\lambda$ = {lmd}")
        ax3.grid()
        if saveplots: fig3.savefig(path+"BS.png")



    #----------------------------------------------------------------------------
    #                           CrossValidation
    #----------------------------------------------------------------------------
    CV_MSE_train = []
    CV_MSE_test = []
    CV_complexity = []
    CV_runtimes = []

    for solver in solvers:
        complexity,MSE_train_set,MSE_test_set,runtimes = run_CV(k,x,y,z,solver,m,lmd=lmd)
        CV_MSE_train.append(MSE_train_set)
        CV_MSE_test.append(MSE_test_set)
        CV_complexity.append(complexity)
        CV_runtimes.append(runtimes)
    
    
    if plot:
        fig1, ax1 = plt.subplots(figsize = ( 10, 7))
        fig2, ax2 = plt.subplots(figsize = ( 10, 7))
        ax1.set_yscale('log')
        if plot_runtime:
            ax2.set_yscale('log')
        for i in range(len(solvers)):
            ax1.plot(CV_complexity[i],CV_MSE_train[i], label =solvers[i]+" train",linestyle = 'dashed')  
            ax1.plot(CV_complexity[i],CV_MSE_test[i], label =solvers[i]+ " test")  
            if plot_runtime:
                ax2.plot(CV_complexity[i],CV_runtimes[i], label =solvers[i]+ " runtime")
    
        ax1.set_xlabel("complexity")
        ax1.set_ylabel("MSE")
        ax1.set_title(f"Plot of the MSE as a function of complexity with CV resampling\n with k = {k}, and $\lambda$ = {lmd}")
        ax1.legend()
        ax1.grid()
        if saveplots: fig1.savefig(path+"CV.png")

        if plot_runtime:
            ax2.set_xlabel("complexity")
            ax2.set_ylabel("Runtime [s]")
            ax2.set_title("Plot of the runtime as a function of complexity of the models")
            ax2.legend()
            ax2.grid()
            if saveplots: fig2.savefig(path+"Runtime.png")


        plt.show() 
    

def compare_lmd_CV(datapoints, N, k, lambdas, poly_degree, solver = 'RIDGE', saveplots = False, foldername = 'Figures'):
    m = poly_degree # polynomial order
    datapoints = datapoints[:N,:N]
    z = datapoints

    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(z)[0])
    y = np.linspace(0,1, np.shape(z)[1])
    x,y = np.meshgrid(x,y)
    z = z.ravel()

    CV_MSE_train = []
    CV_MSE_test = []
    CV_complexity = []
    for lmd in lambdas:
        complexity,MSE_train_set,MSE_test_set,runtimes = run_CV(k,x,y,z,solver,m,lmd=lmd)
        CV_MSE_train.append(MSE_train_set)
        CV_MSE_test.append(MSE_test_set)
        CV_complexity.append(complexity)

    fig1, ax1 = plt.subplots(figsize = ( 10, 7))
    ax1.set_yscale('log')
    for i in range(len(lambdas)):
        #ax1.plot(CV_complexity[i],CV_MSE_train[i], label = f"$\lambda$ = {lambdas[i]} train",linestyle = 'dashed')  
        ax1.plot(CV_complexity[i],CV_MSE_test[i], label =f"$\lambda$ = {lambdas[i]}")

    ax1.set_xlabel("complexity")
    ax1.set_ylabel("MSE")
    ax1.set_title(f"Plot of the MSE as a function of complexity of {solver} regression with CV resampling\n with k = {k}, and different $\lambda$s")
    ax1.legend()
    ax1.grid()
    if saveplots:
        fig1.savefig("../reports/Proj1_Plots/"+foldername+"/"+solver+"k"+str(k)+"_CV.png")
    plt.show()


def compare_lmd_BS(datapoints, N, lambdas, poly_degree, solver = 'RIDGE', n_resampling = 100, saveplots = False, foldername = 'Figures'):
 
    m = poly_degree # polynomial order
    datapoints = datapoints[:N,:N]
    z = datapoints

    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(z)[0])
    y = np.linspace(0,1, np.shape(z)[1])
    x,y = np.meshgrid(x,y)
    z = z.ravel()

    BS_bias = []
    BS_error = []
    BS_variance = []
    BS_complexity = [x for x in range(0,m+1)]
    for lmd in lambdas:
        error, bias, variance = bias_variance_complexity(x,y,z,m,n_resampling=n_resampling,plot=False,solver=solver,lmd=lmd)
        BS_error.append(error)
        BS_bias.append(bias)
        BS_variance.append(variance)

    fig1, ax1 = plt.subplots(figsize = ( 10, 7))
    for i in range(len(lambdas)):
        color = next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(BS_complexity,BS_bias[i], label =f"$\lambda$ = {lambdas[i]} bias$^2$", color=color)
        ax1.plot(BS_complexity,BS_error[i], label =f"$\lambda$ = {lambdas[i]} error", linestyle = 'dashed', color=color)
        ax1.plot(BS_complexity,BS_variance[i], label =f"$\lambda$ = {lambdas[i]} variance", linestyle = 'dotted', color=color) 

    ax1.set_xlabel("complexity")
    ax1.set_ylabel("")
    ax1.set_title(f"Bias-variance tradeoff analysis as a function of complexity of {solver} regression from bootstrap resampling\n resampling = {n_resampling} times, and different $\lambda$s")
    ax1.legend(loc=2, prop={'size': 6})
    ax1.grid()
    if saveplots: fig1.savefig("../reports/Proj1_Plots/"+foldername+"/"+solver+"_BS.png")
    plt.show()
