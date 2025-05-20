# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# # Imports

# +
#imports
import os.path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn import datasets, mixture
import scipy.spatial
import scipy.stats
import time
import ast
from itertools import groupby, chain
import itertools
import pandas as pd
import seaborn as sns
from numpy import log10
import matplotlib.ticker as ticker

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.io.votable import parse_single_table
from astropy.io import ascii
from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord, Galactic
from astropy import units

from shapely.geometry import Point, Polygon
from datetime import datetime
# -

# # Cuts on stellar and planet properties

#cuts on sample - parameters
age_cut = [1,4.5]
mass_cut = [0.7,2]
frac_rad_uncert_cut = 0.2
orb_per_cut = [1,100]
rad_cut = [1,4]

# # Filtering and preparing for plotting

#plotting parameters
plt.rc('xtick',direction='in',top=True)
plt.rc('ytick',direction='in',right=True)
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.labelsize"] = 11  # Font size for axis labels
plt.rcParams["xtick.labelsize"] = 11  # Font size for x-ticks
plt.rcParams["ytick.labelsize"] = 11  # Font size for y-ticks
plt.rcParams["lines.markersize"] = 2 # Marker size
plt.rcParams["lines.linewidth"] = 0.5 # Line width
colors = ['cornflowerblue']


class Exoplanet_Sample:

    def __init__(self, data):

        self.data_full = data
        self.folder_name = 'exoplanet sample'

        self.rad_val_colors = ['r', 'm', 'g']
    
    def Nr_stars(self):
        return len(set(self.data['hostname']))

    def Nr_planets(self):
        return len(self.data)
    
    def pl_radius_abs_uncert(self):
        return (self.data['pl_radeerr1'] - self.data['pl_radeerr2'])/2 

    def pl_radius_frac_uncert(self):
        return self.pl_radius_abs_uncert()/self.data['pl_rade']
    

    def threshold_frac_radius_cut(self,i):
        if self.frac_threshold != np.inf:
            self.data = self.data[self.pl_radius_frac_uncert() <= self.frac_threshold]
            print("Fractional radius uncertainty cut:",len(self.data)) 
            
            self.filename += f'_threshold{i}_{self.frac_threshold}'
            self.label += f'\n Max. Fractional Radius Uncertainty = {self.frac_threshold}'
            
    
    def orbital_period_cut(self,i):
        if self.orbper_cut != None:
            self.data = self.data[self.data['pl_orbper']>= self.orbper_cut[0]]
            self.data = self.data[self.data['pl_orbper']<= self.orbper_cut[1]]
            
            print("Orbital period cut:", len(self.data)) 
            self.filename += f'_orbper{i}_{self.orbper_cut}'
            #self.label += f'\n Orbital Period = {self.orbper_cut}'

            

    def planet_radius_cut(self,i):
        if self.radius_cut!= None:
            self.data = self.data[self.data['pl_rade']>= self.radius_cut[0]]
            self.data = self.data[self.data['pl_rade']<= self.radius_cut[1]]

            print("Planet radius cut:", len(self.data))
            self.filename += f'_plrade{i}_{self.radius_cut}'
            #self.label += f'\n Planet Radius = {self.radius_cut}'
        

    def transit_planet_cut(self,i):
        if self.transit_cut == True:
            self.data = self.data[self.data['tran_flag']==1]
            print("Transit planet cut:", len(self.data))
            
            self.filename += f'__transit_only{i}_{self.transit_cut}'
            self.label += f'\n Transit only = {self.transit_cut}'

    def stellar_age_cut(self,i):
        if self.st_age_cut[0] != 0:
            self.data = self.data[self.data['st_age']>= self.st_age_cut[0]]
            if self.st_age_cut[1] != np.inf:
                self.data = self.data[self.data['st_age']<= self.st_age_cut[1]]
            print("Stellar age cut:", len(self.data))
            
            self.filename += f'_st_age_cut{i}_{self.st_age_cut}'
            self.label += f'\n Stellar Age = {self.st_age_cut}'
        
        else:
            if self.st_age_cut[1] != np.inf:
                self.data = self.data[self.data['st_age']<= self.st_age_cut[1]]
                print("Stellar age cut:", len(self.data))

                self.filename += f'_st_age_cut{i}_{self.st_age_cut}'
                self.label += f'\n Stellar Age = {self.st_age_cut}'

    
    def stellar_mass_cut(self,i):
        if self.st_mass_cut[0] != 0:
            self.data = self.data[self.data['st_mass']>= self.st_mass_cut[0]]
            if self.st_mass_cut[1] != np.inf:
                self.data = self.data[self.data['st_mass']<= self.st_mass_cut[1]]
            print("Stellar mass cut:", len(self.data))
            
            self.filename += f'_st_mass_cut{i}_{self.st_mass_cut}'
            self.label += f'\n Stellar mass = {self.st_mass_cut}'
            
        else:
            if self.st_mass_cut[1] != np.inf:
                self.data = self.data[self.data['st_mass']<= self.st_mass_cut[1]]
                print("Stellar mass cut:", len(self.data))

                self.filename += f'_st_mass_cut{i}_{self.st_mass_cut}'
                self.label += f'\n Stellar mass = {self.st_mass_cut}'

    
    
    def filtering(self, i,
                  frac_threshold,
                  orbpercut,
                  radiuscut,
                  transitcut,
                  stellar_age_cut,
                  stellar_mass_cut):

        #definitions
        self.frac_threshold = frac_threshold
        self.orbper_cut = orbpercut
        self.radius_cut = radiuscut
        self.transit_cut = transitcut
        self.st_age_cut = stellar_age_cut
        self.st_mass_cut = stellar_mass_cut


        #filtering
        self.data = self.data_full
        self.label = ''
        print("No cuts:",len(self.data))
        
        self.threshold_frac_radius_cut(i)
        self.orbital_period_cut(i)
        self.planet_radius_cut(i)
        self.transit_planet_cut(i)
        self.stellar_age_cut(i)
        self.stellar_mass_cut(i)
      
        return self.data, self.label

    def KDE_plot_hist_loglog(self,x,y,
                      axlabel = ['',''],
                      scale = ['log','log'],
                      xlim = None,
                      ylim = None,
                      label = '',
                      filename = None,
                      plot = False,
                      nr_levels=10,
                      fine_tuning=0.5,
                      save = False):
        

        fig, (ax_main, ax_hist) = plt.subplots(1,2,figsize=(15, 8),gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

        #Definitions
        self.x = x
        self.y = y
        
        #Logarithmic scale
        log_x = np.log10(self.x)
        log_y = np.log10(self.y)
        
        
        #KDE plot
        sns.kdeplot(x=log_x, y=log_y, color='b', fill=True, ax=ax_main, cmap="Blues", thresh=0, bw_adjust=fine_tuning, levels=nr_levels)
        
        #Scatter plot
        ax_main.scatter(log_x,log_y, label=label)
        
        ax_main.set_xlabel(axlabel[0])
        ax_main.set_ylabel(axlabel[1])

        #Axes limits
        if xlim == None:
            xlimit1, xlimit2 = ax_main.get_xlim()
            if xlimit1 <= 0:
                xlim = [10**0.01, 10**xlimit2]
            else:
                xlim = [10**xlimit1, 10**xlimit2]

        #print(xlim)
            
        if ylim == None:
            ylimit1, ylimit2 = ax_main.get_ylim()
            if ylimit1 <= 0:
                ylim = [10**0.01, 10**ylimit2]
            else:
                ylim = [10**ylimit1, 10**ylimit2]

        #print(ylim)
            
        x_min = np.log10(xlim[0])
        x_max =  np.log10(xlim[1])
        y_min = np.log10(ylim[0])
        y_max = np.log10(ylim[1])
        
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)

        

        
        #x-axis ticks
        x_logticks_values = np.arange(x_min, x_max+1)
        x_logticks = [value for value in x_logticks_values if value <= x_max ]
        x_ticklabels = [rf"$10^{{{int(tick)}}}$" for tick in x_logticks]  
        ax_main.set_xticks(x_logticks)
        ax_main.set_xticklabels(x_ticklabels)

        x_base_minor = np.arange(1,10)
        x_exponent_minor = np.arange(x_min, x_max + 1)
        x_minor_ticks = []
        for i in range(0, len(x_exponent_minor)):
            for j in range(0, len(x_base_minor)):
                value = x_base_minor[j]*10**(x_exponent_minor[i])
                if value <= xlim[1]:
                    x_minor_ticks.append(value)

        x_log_minor_ticks = np.log10(np.array(x_minor_ticks))
        ax_main.set_xticks(x_log_minor_ticks, minor=True)

        
        #y-axis ticks
        y_logticks_values = np.arange(y_min, y_max+1)
        y_logticks = [value for value in y_logticks_values if value <= y_max ]
        y_ticklabels = [rf"$10^{{{int(tick)}}}$" for tick in y_logticks if tick.is_integer()]  
        ax_main.set_yticks(y_logticks)
        ax_main.set_yticklabels(y_ticklabels)

        y_base_minor = np.arange(1,10)
        y_exponent_minor = np.arange(y_min, y_max + 1)
        y_minor_ticks = []
        y_minor_ticklabels = []
        for i in range(0, len(y_exponent_minor)):
            for j in range(0, len(y_base_minor)):
                value = y_base_minor[j]*10**(y_exponent_minor[i])
                if value <= ylim[1]:
                    y_minor_ticks.append(value)
                    y_minor_ticklabels.append(rf"${y_base_minor[j]}\times 10^{{{int(y_exponent_minor[i])}}}$")

        y_log_minor_ticks = np.log10(np.array(y_minor_ticks))
        
        ax_main.set_yticks(y_log_minor_ticks, minor=True)
        if len(y_ticklabels) <= 1:
            ax_main.set_yticklabels(y_minor_ticklabels, minor=True)
        
        
        
        #Radius valley
        if self.radius_valley1:
            log_P = np.linspace(x_min, x_max, 100) 
            log_R = self.radius_valley(log_P,1)
            ax_main.plot(log_P, log_R, color=self.rad_val_colors[0], label='Van Eylen (2018)')

        if self.radius_valley2:
            log_P = np.linspace(x_min, x_max, 100) 
            log_R = self.radius_valley(log_P,2)
            ax_main.plot(log_P, log_R, color=self.rad_val_colors[1], label = 'Martinez (2019)')
            
        #Set legend
        ax_main.legend(fontsize=8)
        
        #Histogram
        ax_hist.hist(log_y, bins=20, orientation='horizontal', alpha=1)
        ax_hist.set_ylim(y_min, y_max)


        #Adjust spacing
        plt.subplots_adjust(wspace=0.06) 
        
        #Plotting
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        
        #Save plot
        self.filename += '_KDE_loglog'
        
        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')

        
        return


    def KDE_plot_hist_semilog(self,x,y,
                      axlabel = ['',''],
                      xlim = None,
                      ylim = None,
                      label = '',
                      filename = None,
                      plot = False,
                      nr_levels=10,
                      fine_tuning=0.5,
                      save = False):
        

        fig, (ax_main, ax_hist) = plt.subplots(1,2,figsize=(15, 8),gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

        #Definitions
        self.x = x
        self.y = y
        
        #Logarithmic scale
        log_x = np.log10(self.x)
        
        
        #KDE plot
        sns.kdeplot(x=log_x, y=y, color='b', fill=True, ax=ax_main, cmap="Blues", thresh=0, bw_adjust=fine_tuning, levels=nr_levels)
        
        #Scatter plot
        ax_main.scatter(log_x,y, label=label)
        
        ax_main.set_xlabel(axlabel[0])
        ax_main.set_ylabel(axlabel[1])

        #Axes limits
        if xlim == None:
            xlimit1, xlimit2 = ax_main.get_xlim()
            if xlimit1 <= 0:
                xlim = [10**0.01, 10**xlimit2]
            else:
                xlim = [10**xlimit1, 10**xlimit2]

        #print(xlim)
            
        if ylim == None:
            ylimit1, ylimit2 = ax_main.get_ylim()
            if ylimit1 <= 0:
                ylim = [0.01, ylimit2]
            else:
                ylim = [ylimit1, ylimit2]

        #print(ylim)
            
        x_min = np.log10(xlim[0])
        x_max =  np.log10(xlim[1])
        y_min = ylim[0]
        y_max = ylim[1]
        
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)


        
        #x-axis ticks
        x_logticks_values = np.arange(x_min, x_max+1)
        x_logticks = [value for value in x_logticks_values if value <= x_max ]
        x_ticklabels = [rf"$10^{{{int(tick)}}}$" for tick in x_logticks]  
        ax_main.set_xticks(x_logticks)
        ax_main.set_xticklabels(x_ticklabels)

        x_base_minor = np.arange(1,10)
        x_exponent_minor = np.arange(x_min, x_max + 1)
        x_minor_ticks = []
        for i in range(0, len(x_exponent_minor)):
            for j in range(0, len(x_base_minor)):
                value = x_base_minor[j]*10**(x_exponent_minor[i])
                if value <= xlim[1]:
                    x_minor_ticks.append(value)

        x_log_minor_ticks = np.log10(np.array(x_minor_ticks))
        ax_main.set_xticks(x_log_minor_ticks, minor=True)

        
        #Radius valley
        if self.radius_valley1:
            log_P = np.linspace(x_min, x_max, 100) 
            R = 10**(self.radius_valley(log_P,1))
            ax_main.plot(log_P, R, color=self.rad_val_colors[0])

        if self.radius_valley2:
            log_P = np.linspace(x_min, x_max, 100) 
            R = 10**(self.radius_valley(log_P,2))
            ax_main.plot(log_P, R, color=self.rad_val_colors[1])
            

        #Histogram
        ax_hist.hist(y, bins=20, orientation='horizontal', alpha=1)
        ax_hist.set_ylim(y_min, y_max)


        #Adjust spacing
        plt.subplots_adjust(wspace=0.06) 
        
        #Plotting
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        
        #Save plot
        self.filename += '_KDE_semilog'
        
        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')

        
        return

    def KDE_plot_hist_linear(self,x,y,
                      axlabel = ['',''],
                      xlim = None,
                      ylim = None,
                      label = '',
                      filename = None,
                      plot = False,
                      nr_levels=10,
                      fine_tuning=0.5,
                      save = False):
        

        fig, (ax_main, ax_hist) = plt.subplots(1,2,figsize=(15, 8),gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

          
        #KDE plot
        sns.kdeplot(x=x, y=y, color='b', fill=True, ax=ax_main, cmap="Blues", thresh=0, bw_adjust=fine_tuning, levels=nr_levels)
        
        #Scatter plot
        ax_main.scatter(x,y, label=label)
        
        ax_main.set_xlabel(axlabel[0])
        ax_main.set_ylabel(axlabel[1])

        #Axes limits
        if xlim == None:
            xlimit1, xlimit2 = ax_main.get_xlim()
            if xlimit1 <= 0:
                xlim = [0.01, xlimit2]
            else:
                xlim = [xlimit1, xlimit2]

            
        if ylim == None:
            ylimit1, ylimit2 = ax_main.get_ylim()
            if ylimit1 <= 0:
                ylim = [0.01, ylimit2]
            else:
                ylim = [ylimit1, ylimit2]

        ax_main.set_xlim(xlim[0], xlim[1])
        ax_main.set_ylim(ylim[0], ylim[1])
        
        #Radius valley
        if self.radius_valley1:
            P = np.linspace(xlim[0], xlim[1], 100) 
            R = 10**(self.radius_valley(np.log10(P),1))
            ax_main.plot(P,R, color=self.rad_val_colors[0])

        if self.radius_valley2:
            P = np.linspace(xlim[0], xlim[1], 100) 
            R = 10**(self.radius_valley(np.log10(P),2))
            ax_main.plot(P, R, color=self.rad_val_colors[1])

        
        #Histogram
        ax_hist.hist(y, bins=20, orientation='horizontal', alpha=1)
        ax_hist.set_ylim(ylim[0], ylim[1])

        #Adjust spacing
        plt.subplots_adjust(wspace=0.06) 
        
        #Plotting
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        
        #Save plot
        self.filename += '_KDE_linear'
        
        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')

        
        return

        
        
    def scatter_plot_hist_loglog(self,x,y,
                          axlabel = ['',''],
                          scale = ['log','log'],
                          xlim =  None,
                          ylim =  None,
                          label = '',
                          plot = False,
                          save = False):


        fig, (ax_main, ax_hist) = plt.subplots(1,2,figsize=(15, 8),gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

        #Definitions
        self.x = x
        self.y = y
        
        #Logarithmic scale
        log_x = np.log10(self.x)
        log_y = np.log10(self.y)

        #Scatter plot
        ax_main.scatter(log_x,log_y, label=label)
        ax_main.set_xlabel(axlabel[0])
        ax_main.set_ylabel(axlabel[1])

        #Axes limits
        if xlim == None:
            xlimit1, xlimit2 = ax_main.get_xlim()
            xlim = [10**xlimit1, 10**xlimit2]

        if ylim == None:
            ylimit1,ylimit2 = ax_main.get_ylim()
            ylim = [10**ylimit1, 10**ylimit2]

        
        x_min = np.log10(xlim[0])
        x_max =  np.log10(xlim[1])
        y_min = np.log10(ylim[0])
        y_max = np.log10(ylim[1])
        
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)


        
        #x-axis ticks
        x_logticks_values = np.arange(x_min, x_max+1)
        x_logticks = [value for value in x_logticks_values if value <= x_max ]
        x_ticklabels = [rf"$10^{{{int(tick)}}}$" for tick in x_logticks]  
        ax_main.set_xticks(x_logticks)
        ax_main.set_xticklabels(x_ticklabels)
        
        x_base_minor = np.arange(1,10)
        x_exponent_minor = np.arange(x_min, x_max + 1)
        x_minor_ticks = []
        for i in range(0, len(x_exponent_minor)):
            for j in range(0, len(x_base_minor)):
                value = x_base_minor[j]*10**(x_exponent_minor[i])
                if value <= xlim[1]:
                    x_minor_ticks.append(value)

        x_log_minor_ticks = np.log10(np.array(x_minor_ticks))
        ax_main.set_xticks(x_log_minor_ticks, minor=True)

        if len(x_ticklabels) <= 1:
            ax_main.set_xticklabels(x_minor_ticklabels, minor=True)

        
        #y-axis ticks
        y_logticks_values = np.arange(y_min, y_max+1)
        y_logticks = [value for value in y_logticks_values if value <= y_max ]
        y_ticklabels = [rf"$10^{{{int(tick)}}}$" for tick in y_logticks]  
        ax_main.set_yticks(y_logticks)
        ax_main.set_yticklabels(y_ticklabels)

        y_base_minor = np.arange(1,10)
        y_exponent_minor = np.arange(y_min, y_max + 1)
        y_minor_ticks = []
        y_minor_ticklabels = []
        for i in range(0, len(y_exponent_minor)):
            for j in range(0, len(y_base_minor)):
                value = y_base_minor[j]*10**(y_exponent_minor[i])
                if value <= ylim[1]:
                    y_minor_ticks.append(value)
                    y_minor_ticklabels.append(rf"${y_base_minor[j]}\times 10^{{{int(y_exponent_minor[i])}}}$")

        y_log_minor_ticks = np.log10(np.array(y_minor_ticks))
        
        ax_main.set_yticks(y_log_minor_ticks, minor=True)
        if len(y_ticklabels) <= 1:
            ax_main.set_yticklabels(y_minor_ticklabels, minor=True)
        
        
        
        
        
        #Radius valley
        
        if self.radius_valley1:
            log_P = np.linspace(x_min, x_max, 100) 
            log_R = self.radius_valley(log_P,1)
            ax_main.plot(log_P, log_R, color=self.rad_val_colors[0])

        if self.radius_valley2:
            log_P = np.linspace(x_min, x_max, 100) 
            log_R = self.radius_valley(log_P,2)
            ax_main.plot(log_P, log_R, color=self.rad_val_colors[1])


        
        #Histogram
        ax_hist.hist(log_y, bins=20, orientation='horizontal', alpha=1)
        ax_hist.set_ylim(y_min, y_max)


        #Adjust spacing
        plt.subplots_adjust(wspace=0.06) 
        
        #Plotting
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        
        #Save plot
        self.filename += '_hist_loglog'
        
        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')
        
        return


    def scatter_plot_hist_semilog(self,x,y,
                          axlabel = ['',''],
                          xlim =  None,
                          ylim =  None,
                          label = '',
                          plot = False,
                          save = False):


        fig, (ax_main, ax_hist) = plt.subplots(1,2,figsize=(15, 8),gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

        #Definitions
        self.x = x
        
        #Logarithmic scale
        log_x = np.log10(self.x)

        #Scatter plot
        ax_main.scatter(log_x,y, label=label)
        ax_main.set_xlabel(axlabel[0])
        ax_main.set_ylabel(axlabel[1])

        #Axes limits
        if xlim == None:
            xlimit1, xlimit2 = ax_main.get_xlim()
            xlim = [10**xlimit1, 10**xlimit2]

        if ylim == None:
            ylimit1,ylimit2 = ax_main.get_ylim()
            ylim = [ylimit1, ylimit2]

        
        x_min = np.log10(xlim[0])
        x_max =  np.log10(xlim[1])
        y_min = ylim[0]
        y_max = ylim[1]
        
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)


        
        #x-axis ticks
        x_logticks_values = np.arange(x_min, x_max+1)
        x_logticks = [value for value in x_logticks_values if value <= x_max ]
        x_ticklabels = [rf"$10^{{{int(tick)}}}$" for tick in x_logticks]  
        ax_main.set_xticks(x_logticks)
        ax_main.set_xticklabels(x_ticklabels)
        
        x_base_minor = np.arange(1,10)
        x_exponent_minor = np.arange(x_min, x_max + 1)
        x_minor_ticks = []
        for i in range(0, len(x_exponent_minor)):
            for j in range(0, len(x_base_minor)):
                value = x_base_minor[j]*10**(x_exponent_minor[i])
                if value <= xlim[1]:
                    x_minor_ticks.append(value)

        x_log_minor_ticks = np.log10(np.array(x_minor_ticks))
        ax_main.set_xticks(x_log_minor_ticks, minor=True)

        if len(x_ticklabels) <= 1:
            ax_main.set_xticklabels(x_minor_ticklabels, minor=True)

        
        #Radius valley
        
        if self.radius_valley1:
            log_P = np.linspace(x_min, x_max, 100) 
            R = 10**(self.radius_valley(log_P,1))
            ax_main.plot(log_P, R, color=self.rad_val_colors[0])

        if self.radius_valley2:
            log_P = np.linspace(x_min, x_max, 100) 
            R = 10**(self.radius_valley(log_P,2))
            ax_main.plot(log_P, R, color=self.rad_val_colors[1])

        
        #Histogram
        ax_hist.hist(y, bins=20, orientation='horizontal', alpha=1)
        ax_hist.set_ylim(y_min, y_max)


        #Adjust spacing
        plt.subplots_adjust(wspace=0.06) 
        
        #Plotting
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        
        #Save plot
        self.filename += '_hist_semilog'
        
        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')
        
        return
    
    def scatter_plot_hist_linear(self,x,y,
                          axlabel = ['',''],
                          xlim =  None,
                          ylim =  None,
                          label = '',
                          plot = False,
                          save = False):


        fig, (ax_main, ax_hist) = plt.subplots(1,2,figsize=(15, 8),gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

        #Scatter plot
        ax_main.scatter(x,y, label=label)
        ax_main.set_xlabel(axlabel[0])
        ax_main.set_ylabel(axlabel[1])

        #Axes limits
        if xlim == None:
            xlimit1, xlimit2 = ax_main.get_xlim()
            xlim = [xlimit1, xlimit2]

        if ylim == None:
            ylimit1,ylimit2 = ax_main.get_ylim()
            ylim = [ylimit1, ylimit2]

        
        ax_main.set_xlim(xlim[0], xlim[1])
        ax_main.set_ylim(ylim[0], ylim[1])  
        
        #Radius valley
        
        if self.radius_valley1:
            P = np.linspace(xlim[0], xlim[1], 100) 
            R = 10**(self.radius_valley(np.log10(P),1))
            ax_main.plot(P,R, color=self.rad_val_colors[0])

        if self.radius_valley2:
            P = np.linspace(xlim[0], xlim[1], 100) 
            R = 10**(self.radius_valley(np.log10(P),2))
            ax_main.plot(P,R, color=self.rad_val_colors[1])

        
        #Histogram
        ax_hist.hist(y, bins=20, orientation='horizontal', alpha=1)
        ax_hist.set_ylim(ylim[0], ylim[1]) 

        #Adjust spacing
        plt.subplots_adjust(wspace=0.06) 
        
        #Plotting
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        
        #Save plot
        self.filename += '_hist_linear'
        
        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')
        
        return    
    
    def scatter_plot(self, x, y, 
                      axlabel=['',''],  
                      scale=['log','log'],
                      xlim=None, 
                      ylim=None, 
                      label='', 
                      plot=False,
                      save=False):
        
        
        fig, ax = plt.subplots(figsize=(12,7))

        ax.scatter(x,y, label=label)
        ax.set_xlabel(axlabel[0])
        ax.set_ylabel(axlabel[1])
        
        if xlim != None:
            ax.set_xlim(xlim)
            xmin, xmax = xlim
        else:
            xmin, xmax = ax.get_xlim()
        
        if ylim != None:
            ax.set_ylim(ylim)
            ymin, ymax = ylim
        else:
            ymin, ymax = ax.get_ylim()
            
        
        ax.legend(fontsize=8)
        
        #Radius valley
        if self.radius_valley1:
            P = np.linspace(xmin, xmax, 100) 
            R = 10**(self.radius_valley(np.log10(P),1))
            ax.plot(P, R, color=self.rad_val_colors[0])

        if self.radius_valley2:
            P = np.linspace(xmin, xmax, 100) 
            R = 10**(self.radius_valley(np.log10(P),2))
            ax.plot(P, R, color=self.rad_val_colors[1])

        ax.set_xscale(scale[0])
        ax.set_yscale(scale[1])

        
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        self.filename += '_'+scale[0]+'_'+scale[1]

        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')
        
        return
        
    def scatter_plot2(self, x1, y1, x2, y2,
                      axlabel_1=['',''], 
                      axlabel_2=['',''],  
                      scale_1=['log','log'],
                      scale_2=['log','log'],
                      xlim_1=None, 
                      xlim_2=None,
                      ylim_1=None, 
                      ylim_2=None,
                      label1='', 
                      label2='', 
                      sharex=False, 
                      sharey=False, 
                      plot=False,
                      save=False):
        
        
        fig, ax = plt.subplots(1,2, figsize=(20,6),sharey=sharey)

        ax[0].scatter(x1,y1, label=label1)
        ax[0].set_xlabel(axlabel_1[0])
        ax[0].set_ylabel(axlabel_1[1])
        
        if xlim_1 != None:
            ax[0].set_xlim(xlim_1)
            xmin_1, xmax_1 = xlim_1
        else:
            xmin_1, xmax_1 = ax[0].get_xlim()
        
        if ylim_1 != None:
            ax[0].set_ylim(ylim_1)
            ymin_1, ymax_1 = ylim_1
        else:
            ymin_1, ymax_1 = ax[0].get_ylim()
            
        ax[0].set_xscale(scale_1[0])
        ax[0].set_yscale(scale_1[1])
        ax[0].legend(fontsize=8)
        
        ax[1].scatter(x2,y2, label=label2)
        ax[1].set_xlabel(axlabel_2[0])
        ax[1].set_ylabel(axlabel_2[1])

        #Axes limits
        if xlim_2 != None:
            ax[1].set_xlim(xlim_2)
            xmin_2, xmax_2 = xlim_2
        else:
            xmin_2, xmax_2 = ax[1].get_xlim()
            
        if ylim_2 != None:
            ax[1].set_ylim(ylim_2)
            ymin_2, ymax_2 = ylim_2
        else:
            ymin_2, ymax_2 = ax[1].get_ylim()


        #Scale
        ax[1].legend(fontsize=8)

        
        #Radius valley
        if self.radius_valley1:
            P1 = np.linspace(xmin_1, xmax_1, 100) 
            P2 = np.linspace(xmin_2, xmax_2, 100)
            R1 = 10**(self.radius_valley(np.log10(P1),1))
            R2 = 10**(self.radius_valley(np.log10(P2),1))
            ax[0].plot(P1, R1, color=self.rad_val_colors[0])
            ax[1].plot(P2, R2, color=self.rad_val_colors[0])

        if self.radius_valley2:
            P1 = np.linspace(xmin_1, xmax_1, 100) 
            P2 = np.linspace(xmin_2, xmax_2, 100)
            R1 = 10**(self.radius_valley(np.log10(P1),2))
            R2 = 10**(self.radius_valley(np.log10(P2),2))
            ax[0].plot(P1, R1, color=self.rad_val_colors[1])
            ax[1].plot(P2, R2, color=self.rad_val_colors[1])

        ax[1].set_xscale(scale_2[0])
        ax[1].set_yscale(scale_2[1])

        #Adjust spacing
        if sharey:
            plt.subplots_adjust(wspace=0.06) 


        self.filename += '_'+scale_1[0]+scale_1[1]+'_'+scale_2[0]+scale_2[1]
        
        #Saving plot
        if plot == True:
            plt.show(fig)
        else:
            plt.close(fig)

        if save == True:
            if self.filename != None:
                pathname = f'{self.folder_name}/{self.filename}.png'
                if os.path.exists(pathname) == False:
                    fig.savefig(pathname)
                    print(f'Saved figure at: {pathname}')
                else:
                    print('\n'+f'Figure already exists at: {pathname}')
        
        return

    
    def orbper_radius(self,
                      frac_threshold1=np.inf,
                      frac_threshold2=np.inf,
                      orbpercut_1=None,
                      radiuscut_1=None,
                      orbpercut_2=None,
                      radiuscut_2=None,
                      two_plots=False,
                      transit_only1=False,
                      transit_only2=False,
                      sharex=False, 
                      sharey=False,
                      hist = False,
                      stellar_age_cut_1 = [0,np.inf],
                      stellar_age_cut_2 = [0, np.inf],
                      stellar_mass_cut_1 = [0,np.inf],
                      stellar_mass_cut_2 = [0,np.inf],
                      KDE_plot = False,
                      plot = True,
                      save = False,
                      nr_levels=10,
                      fine_tuning=0.8,
                      radius_valley1 = False,
                      radius_valley2 = False,
                      scale = 'loglog'):

        #radius valley
        self.radius_valley1 = radius_valley1
        self.radius_valley2 = radius_valley2

        
        #preparing file
        self.filename = 'orbper_radius'

        
        #determine cuts
        self.frac_thresh_1 = frac_threshold1
        self.frac_thresh_2 = frac_threshold2

        self.orbpercut_1 = orbpercut_1
        self.orbpercut_2 = orbpercut_2

        self.radiuscut_1 = radiuscut_1
        self.radiuscut_2 = radiuscut_2

        self.transit_1 = transit_only1
        self.transit_2 = transit_only2

        self.st_age_cut_1 = stellar_age_cut_1
        self.st_age_cut_2 = stellar_age_cut_2 

        self.st_mass_cut_1 = stellar_mass_cut_1
        self.st_mass_cut_2 = stellar_mass_cut_2 

        self.scale = scale

        #determine whether 1 or 2 plots should be created
        self.two_plots = two_plots

        
        #Filter data

        if self.two_plots:
            print("Data 1:")
            
            self.data1, self.label1 = self.filtering(1,self.frac_thresh_1,
                                        self.orbpercut_1,
                                        self.radiuscut_1,
                                        self.transit_1,
                                        self.st_age_cut_1,
                                        self.st_mass_cut_1)
            print("\n")
            print("Data 2:")
            
            self.data2, self.label2 = self.filtering(2, self.frac_thresh_2,
                                        self.orbpercut_2,
                                        self.radiuscut_2,
                                        self.transit_2,
                                        self.st_age_cut_2,
                                        self.st_mass_cut_2)
            print("\n")                           
        
            #Define axes
            self.orbper_1 = self.data1['pl_orbper']
            self.plrade_1 = self.data1['pl_rade']
            
            self.orbper_2 = self.data2['pl_orbper']
            self.plrade_2 = self.data2['pl_rade']

        else:
            print("Data:")
            self.data, self.label = self.filtering(1,self.frac_thresh_1,
                                    self.orbpercut_1,
                                    self.radiuscut_1,
                                    self.transit_1,
                                    self.st_age_cut_1,
                                    self.st_mass_cut_1)
            print("\n")
            #Define axes
            self.orbper = self.data['pl_orbper']
            self.plrade = self.data['pl_rade']
            

        
        #Plotting 
        
        if KDE_plot:
            if self.scale == 'loglog':
                self.KDE_plot_hist_loglog(self.orbper, self.plrade,
                                           xlim = self.orbpercut_1,
                                           ylim = self.radiuscut_1,
                                           axlabel= ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                           label = self.label,
                                           nr_levels=10,
                                           fine_tuning=0.8,
                                           plot=plot,
                                           save=save)
            if self.scale == 'semilog':
                self.KDE_plot_hist_semilog(self.orbper, self.plrade,
                                           xlim = self.orbpercut_1,
                                           ylim = self.radiuscut_1,
                                           axlabel= ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                           label = self.label,
                                           nr_levels=10,
                                           fine_tuning=0.8,
                                           plot=plot,
                                           save=save)
            if self.scale == 'linear':
                self.KDE_plot_hist_linear(self.orbper, self.plrade,
                                           xlim = self.orbpercut_1,
                                           ylim = self.radiuscut_1,
                                           axlabel= ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                           label = self.label,
                                           nr_levels=10,
                                           fine_tuning=0.8,
                                           plot=plot,
                                           save=save)

        else:
            if self.two_plots:
                if self.scale == 'loglog':
                    self.scatter_plot2(self.orbper_1, self.plrade_1, self.orbper_2, self.plrade_2,
                                      xlim_1 = self.orbpercut_1,
                                      ylim_1 = self.radiuscut_1,
                                      xlim_2 = self.orbpercut_2,
                                      ylim_2 = self.radiuscut_2,
                                      axlabel_1= ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                      axlabel_2= ['Orbital Period [days]',''],
                                      label1 = self.label1,
                                      label2 = self.label2,
                                      sharey=sharey,
                                      plot=plot,
                                      scale_1=['log','log'],
                                      scale_2=['log','log'],
                                      save=save)
                if self.scale == 'semilog':
                        self.scatter_plot2(self.orbper_1, self.plrade_1, self.orbper_2, self.plrade_2,
                                          xlim_1 = self.orbpercut_1,
                                          ylim_1 = self.radiuscut_1,
                                          xlim_2 = self.orbpercut_2,
                                          ylim_2 = self.radiuscut_2,
                                          axlabel_1= ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                          axlabel_2= ['Orbital Period [days]',''],
                                          label1 = self.label1,
                                          label2 = self.label2,
                                          sharey=sharey,
                                          plot=plot,
                                          scale_1=['log','linear'],
                                          scale_2=['log','linear'],
                                          save=save)
                if self.scale == 'linear':
                    self.scatter_plot2(self.orbper_1, self.plrade_1, self.orbper_2, self.plrade_2,
                                      xlim_1 = self.orbpercut_1,
                                      ylim_1 = self.radiuscut_1,
                                      xlim_2 = self.orbpercut_2,
                                      ylim_2 = self.radiuscut_2,
                                      axlabel_1= ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                      axlabel_2= ['Orbital Period [days]',''],
                                      label1 = self.label1,
                                      label2 = self.label2,
                                      sharey=sharey,
                                      plot=plot,
                                      scale_1=['linear','linear'],
                                      scale_2=['linear','linear'],
                                      save=save)

            else:
                if hist == False:
                    if self.scale == 'loglog':
                        self.scatter_plot(self.orbper, self.plrade,
                                         xlim = self.orbpercut_1,
                                         ylim = self.radiuscut_1,
                                         axlabel = ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                         label = self.label,
                                         scale=['log','log'],
                                         plot=plot,
                                         save=save)
                    if self.scale == 'semilog':
                        self.scatter_plot(self.orbper, self.plrade,
                                         xlim = self.orbpercut_1,
                                         ylim = self.radiuscut_1,
                                         axlabel = ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                         label = self.label,
                                         scale=['log','linear'],
                                         plot=plot,
                                         save=save)
                    if self.scale == 'linear':
                        self.scatter_plot(self.orbper, self.plrade,
                                         xlim = self.orbpercut_1,
                                         ylim = self.radiuscut_1,
                                         axlabel = ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                         label = self.label,
                                         scale=['linear','linear'],
                                         plot=plot,
                                         save=save)
                else:
                    if self.scale == 'loglog':
                        self.scatter_plot_hist_loglog(self.orbper, self.plrade,
                                                       xlim = self.orbpercut_1,
                                                       ylim = self.radiuscut_1,
                                                       axlabel = ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                                       label = self.label,
                                                       plot=plot,
                                                       save=save)
                    if self.scale == 'semilog':
                        self.scatter_plot_hist_semilog(self.orbper, self.plrade,
                                                       xlim = self.orbpercut_1,
                                                       ylim = self.radiuscut_1,
                                                       axlabel = ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                                       label = self.label,
                                                       plot=plot,
                                                       save=save)
                    if self.scale == 'linear':
                        self.scatter_plot_hist_linear(self.orbper, self.plrade,
                                                       xlim = self.orbpercut_1,
                                                       ylim = self.radiuscut_1,
                                                       axlabel = ['Orbital Period [days]',r'Planet radius [$R_\oplus$]'],
                                                       label = self.label,
                                                       plot=plot,
                                                       save=save)

    def radius_valley(self, P,i):
        if i==1:
            #van Eylen (2018)
            m = -0.09
            R_b = 1.9
            a = np.log10(R_b)-m
            self.filename += '_radval1'
        
        if i==2:
            #Martinez (2019)
            m = -0.11
            R_b = 1.9
            a = np.log10(R_b)-m
            self.filename += '_radval2'
        
        R = m * P + a
        return R


# # Collect data

#processing data into a table
filename ='PSCompPars_2025.05.16_03.09.22.csv'
planets_table = ascii.read(filename)


# # Find Gaia DR3 ids for all host stars

# +
include_Sol = True

#fix "Qatar-n" -> "Qatar n"
planets_table['hostname'] = [t.replace("Qatar-","Qatar ") for t in planets_table['hostname']]
#Praesepe: "Prnnnn" isn't a catalogue in Simbad, and I can't find it's published anywhere
#exoplanet.eu ids Pr0201 as BD+20 2184 but no id for Pr0211...
planets_table['hostname'] = [t.replace("Pr0201","BD+20 2184") for t in planets_table['hostname']]
#HIP 65A is just HIP 65 in Simbad

targets = [p['hostname'] for p in planets_table]
planets = [p['pl_name'] for p in planets_table]

print(f"nr of exoplanet systems: {len(targets)}")
print(f"nr of exoplanets: {len(planets)}")

#remove duplicate hosts
targets = list(set(targets))
planets = list(set(planets))

print(f"nr of exoplanet systems: {len(targets)}")
print(f"nr of exoplanets: {len(planets)}")


#logfile
logdir = 'log'
if not os.path.exists(logdir):
    os.mkdir(logdir)
logfile = 'log/xmatch_ids_1747391354.376178.txt'

#cross-matching
if len(targets) > 0:
    dr2id = []
    dr3id = []

    if os.path.exists(logfile):
        print("Reading from file ", logfile)
        xmatch = ascii.read(logfile,delimiter=',',data_start=1,format='csv')
        restored_from_file = True
        dr2id = []
        dr3id = []
        for t in targets:
            index = np.where(xmatch['target'] == t)
            dr2id.append(xmatch['dr2id'][index][0])
            dr3id.append(xmatch['dr3id'][index][0])
        
        missing_dr3ids = dr3id[dr3id == 'None']
        missing_dr2ids = dr2id[dr2id == 'None']
        print("Number of Gaia DR3 IDs with 'None':", len(missing_dr3ids)) 
        print("Number of Gaia DR2 IDs with 'None':", len(missing_dr2ids)) 
        dr2id[dr2id == 'None'] = None
        dr3id[dr3id == 'None'] = None

    
    else:
        for i in range(len(targets)):
            try:
                print('Accessing Simbad...')
                match2 = Simbad.query_objectids(targets[i], criteria="ident.id LIKE 'Gaia DR2%'")
                match3 = Simbad.query_objectids(targets[i], criteria="ident.id LIKE 'Gaia DR3%'")
                if len(match3) == 1:
                    dr3id.append(match3[0][0])
                else:
                    dr3id.append(None)      
                if len(match2) == 1:
                    dr2id.append(match2[0][0])
                else:
                    dr2id.append(None)
            except:
                print("Access Simbad failed")
        
        file = logdir+'/xmatch_ids_'+str(time.time())+'.txt'
        with open(file,'w') as f:
            print('target,    dr2id,    dr3id',file=f)
            for i in range(len(targets)):
                print(targets[i],',',dr2id[i],',',dr3id[i],file=f)


if include_Sol:
    if targets[-1] != 'Sol':
        targets.append('Sol')
        dr3id.append('Sol')
        dr2id.append('Sol')


#for i in range(5):
#    print(targets[i],dr2id[i], dr3id[i])

# -

# # Find which host stars within 1kpc have Gaia DR3 radial velocities

# +
filepath1 = 'results/data_densities_DR3.txt'

if os.path.exists(filepath1):
    savefile1 = ascii.read(filepath1,delimiter=',',data_start=1,format='csv')
    new_targets_list = []
    new_dr3ids_list = []
    densities_list = []
    density_comps_list = []
    P_1components_list = []
    P_1components_v_list = []
    N_samples_list = []

    for i in range(len(targets)):
        index = np.where((savefile1['name'] == targets[i]) & (savefile1['designation'] == dr3id[i]))[0]
        if len(index) == 1:
            new_targets_list.append(savefile1['name'][index][0])
            new_dr3ids_list.append(savefile1['designation'][index][0])
            densities_list.append(float(savefile1['rho'][index][0]))
            density_comps_list.append(float(str(savefile1['P_target'][index][0]).strip('[]')))
            P_1components_list.append(np.fromstring(savefile1['P_1comp'][index][0].strip('[]'), sep=' ').tolist())
            P_1components_v_list.append(np.fromstring(savefile1['P_1compv'][index][0].strip('[]'), sep=' ').tolist())
            N_samples_list.append(float(savefile1['N_sam