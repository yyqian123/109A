import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def basic_plots(amazon,figsize=(10,14)):
    fig,axs = plt.subplots(nrows=3,figsize=figsize)
    ax = axs[0]
    amazon.plot('DATE','PX LAST',ax=ax)

    ax =axs[1]
    bins = np.linspace(amazon['PX LAST'].min(),amazon['PX LAST'].max(),30)
    amazon.hist('PX LAST',bins=bins,rwidth=0.8,ax=ax)
    ax.set_yscale('log')

    ax =axs[2]
    dam = amazon['PX LAST'].diff()
    bins=np.arange(-200,201,10)
    dam.hist(rwidth=0.8,bins=bins,ax=ax)
    ax.set_yscale('log')
    plt.subplots_adjust(hspace=0.4)
    return fig,axs

def acf(amazon):
    fig,axs = plt.subplots(nrows=2,figsize=(10,10))
    ax = axs[0]
    plot_acf(amazon['PX LAST'],ax=ax)
    ax = axs[1]
    plot_pacf(amazon['PX LAST'],ax=ax)
    return fig,axs

amazon = pd.read_csv('data/Google.csv',parse_dates=['DATE'])
fig,ax = basic_plots(amazon)
fig2,ax2 = acf(amazon)
