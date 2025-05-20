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

# +
import os.path
import os
import sys
import numpy as np
import scipy.spatial
import scipy.stats
import time
import ast
from itertools import groupby, chain
import itertools
import pandas as pd

from sklearn import mixture

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.io.votable import parse_single_table
from astropy.io import ascii
from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord, Galactic
from astropy import units

from shapely.geometry import Point, Polygon
from datetime import datetime

# +
plt.rc('xtick',direction='in',top=True)
plt.rc('ytick',direction='in',right=True)
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "stix"

from astropy.visualization import quantity_support
quantity_support()
# -

# First we need to import the Gaia data (DR2 or eDR3, or a smaller sample centred on Sol for speed), set up the coordinates and add the Sun

# +
#add directory for data
data_directory = 'data'

# what is our source? DR2 or eDR3? For Solar nbhd (80pc), use plain DR2/3 as they're (much) quicker to load
#source_cat = 'eDR3'
#source_cat = 'DR2'
#source_cat = 'DR2_all'
#source_cat = 'eDR3_all'
#source_cat = 'DR3_500pc'
source_cat = 'DR3_1kpc'
#source_cat = 'DR3_all'


# get 80 pc RV sample
if source_cat == 'eDR3':
    name_file = "eDR3_RV_80pc.vot"
    filename = os.path.join(data_directory, source_cat, name_file)
    if os.path.exists(filename):
        data_all = parse_single_table(filename).to_table()
    else:
        job = Gaia.launch_job_async("select * from gaiaedr3.gaia_source where parallax > 12.5 and "
                                        "dr2_radial_velocity IS NOT NULL",dump_to_file=True,output_format = "votable",
                                    output_file=filename)
        r = job.get_results()
        data_all = parse_single_table(filename).to_table()
    data_all.rename_column('dr2_radial_velocity','radial_velocity')

if source_cat == 'DR2':
    name_file = "DR2_RV_80pc.vot"
    filename = os.path.join(data_directory, name_file)
    if os.path.exists(filename):
        data_all = parse_single_table(filename).to_table()
    else:
        job = Gaia.launch_job_async("select * from gaiadr2.gaia_source where parallax > 12.5 and "
                                        "radial_velocity IS NOT NULL",dump_to_file=True,output_format = "votable",
                                    output_file=filename)
        r = job.get_results()
        data_all = parse_single_table(filename).to_table()

#get full RV sample
if source_cat == 'DR2_all':
    start = time.time()
    tables = []
    for i in range(6):
        name_file = "DR2_RV_all_"+"{:1d}".format(i)+".fits"
        filename = os.path.join(data_directory, source_cat, name_file)
        if os.path.exists(filename):
            #tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:1d}".format(i+1)+" of 6")
        else:
            job = Gaia.launch_job_async("select designation, source_id, ref_epoch, ra, ra_error, dec, dec_error, "
                                            "parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, ra_dec_corr, "
                                            "ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, "
                                            "dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, "
                                            "astrometric_gof_al, astrometric_excess_noise, astrometric_excess_noise_sig, "
                                            "phot_g_mean_flux, phot_g_mean_flux_error, phot_g_mean_mag, phot_bp_mean_flux, "
                                            "phot_bp_mean_flux_error, phot_bp_mean_mag, phot_rp_mean_flux, "
                                            "phot_rp_mean_flux_error, phot_rp_mean_mag, radial_velocity, "
                                            "radial_velocity_error from gaiadr2.gaia_source where "
                                            "radial_velocity IS NOT NULL and ra >= "+str(60*i)+" and "
                                            "ra < "+str(60*(i+1)),dump_to_file=True,output_format = "fits",
                                        output_file=filename)
            print("Downloaded table "+"{:1d}".format(i+1)+" of 6")
            r = job.get_results()
            #tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:1d}".format(i+1)+" of 6")
        if i == 0:
            data_all = tmp
        else:
#            stackstart = time.time()
#            data_all = vstack([data_all,tmp])
            tables.append(tmp)
#            stackend = time.time()
#            print(f'append took {stackend-stackstart} seconds')
    stackstart = time.time()
    data_all = vstack(tables)
    stackend = time.time()
    print(f'vstack took {stackend-stackstart} seconds')
    end = time.time()
    print(f'Total {end-start} seconds')

    

if source_cat == 'eDR3_all':
    for i in range(6):
        name_file = "eDR3_RV_all_"+"{:1d}".format(i)+".vot"
        filename = os.path.join(data_directory, source_cat, name_file)
        if os.path.exists(filename):
            tmp = parse_single_table(filename).to_table()
            print("Read table "+"{:1d}".format(i+1)+" of 6")
        else:
            job = Gaia.launch_job_async("select designation, source_id, ref_epoch, ra, ra_error, dec, dec_error, "
                                            "parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, ra_dec_corr, "
                                            "ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, "
                                            "dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, "
                                            "astrometric_gof_al, astrometric_excess_noise, astrometric_excess_noise_sig, "
                                            "phot_g_mean_flux, phot_g_mean_flux_error, phot_g_mean_mag, phot_bp_mean_flux, "
                                            "phot_bp_mean_flux_error, phot_bp_mean_mag, phot_rp_mean_flux, "
                                            "phot_rp_mean_flux_error, phot_rp_mean_mag, dr2_radial_velocity, "
                                            "dr2_radial_velocity_error from gaiaedr3.gaia_source where "
                                            "dr2_radial_velocity IS NOT NULL and ra >= "+str(60*i)+" and "
                                            "ra < "+str(60*(i+1)),dump_to_file=True,output_format = "votable",
                                        output_file=filename)
            print("Downloaded table "+"{:1d}".format(i+1)+" of 6")
            r = job.get_results()
            tmp = parse_single_table(filename).to_table()
            print("Read table "+"{:1d}".format(i+1)+" of 6")
        if i == 0:
            data_all = tmp
        else:
            data_all = vstack([data_all,tmp])
    
    data_all.rename_column('dr2_radial_velocity','radial_velocity')
    data_all.rename_column('dr2_radial_velocity_error','radial_velocity_error')            


if source_cat == 'DR3_500pc':
    start = time.time()
    tables = []
    for i in range(72):
        name_file = "DR3_RV_all_500pc_"+"{:02d}".format(i+1)+".fits"
        filename = os.path.join(data_directory, source_cat, name_file)
        if os.path.exists(filename):
#            tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:02d}".format(i+1)+" of 72; "+str(len(tmp))+ " entries")
        else:
            job = Gaia.launch_job_async(f"select designation, source_id, ref_epoch, ra, dec, "
                                            f"parallax, parallax_error, pmra, pmdec, "
                                            f"phot_g_mean_mag, "
                                            f"phot_bp_mean_mag, "
                                            f"phot_rp_mean_mag, radial_velocity "
                                            f"from gaiadr3.gaia_source where "
                                            f"radial_velocity IS NOT NULL and ra >= {(5*i)} and "
                                            f"ra < {5*(i+1)} and "
                                            f"parallax >= 2", 
                                        dump_to_file=True,output_format = "fits",
                                        output_file=filename)
            print("Downloaded table "+"{:02d}".format(i+1)+" of 72")
            r = job.get_results()
#            tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:02d}".format(i+1)+" of 72; "+str(len(tmp))+ " entries")
        if i == 0:
            data_all = tmp
        else:
#            stackstart = time.time()
#            data_all = vstack([data_all,tmp])
            tables.append(tmp)
#            stackend = time.time()
#            print(f'append took {stackend-stackstart} seconds')
    stackstart = time.time()
    data_all = vstack(tables)
    stackend = time.time()
    print(f'vstack took {stackend-stackstart} seconds')
    end = time.time()
    print(f'Total {end-start} seconds')



if source_cat == 'DR3_1kpc':
    start = time.time()
    tables = []
    for i in range(72):
        name_file = "DR3_RV_1kpc_"+"{:02d}".format(i+1)+".fits"
        filename = os.path.join(data_directory, source_cat, name_file)
        if os.path.exists(filename):
#            tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:02d}".format(i+1)+" of 72; "+str(len(tmp))+ " entries")
        else:
            job = Gaia.launch_job_async(f"select designation, source_id, ref_epoch, ra, dec, "
                                            f"parallax, parallax_error, pmra, pmdec, "
                                            f"phot_g_mean_mag, "
                                            f"phot_bp_mean_mag, "
                                            f"phot_rp_mean_mag, radial_velocity "
                                            f"from gaiadr3.gaia_source where "
                                            f"radial_velocity IS NOT NULL and ra >= {(5*i)} and "
                                            f"ra < {5*(i+1)} and "
                                            f"parallax >= 1", 
                                        dump_to_file=True,output_format = "fits",
                                        output_file=filename)
            print("Downloaded table "+"{:02d}".format(i+1)+" of 72")
            r = job.get_results()
#            tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:02d}".format(i+1)+" of 72; "+str(len(tmp))+ " entries")
        if i == 0:
            data_all = tmp
        else:
#            stackstart = time.time()
#            data_all = vstack([data_all,tmp])
            tables.append(tmp)
#            stackend = time.time()
#            print(f'append took {stackend-stackstart} seconds')
    stackstart = time.time()
    data_all = vstack(tables)
    stackend = time.time()
    print(f'vstack took {stackend-stackstart} seconds')
    end = time.time()
    print(f'Total {end-start} seconds')


if source_cat == 'DR3_all':
    start = time.time()
    tables = []
    for i in range(72):
        name_file = "DR3_RV_all_"+"{:02d}".format(i+1)+".fits"
        filename = os.path.join(data_directory, source_cat, name_file)
        if os.path.exists(filename):
#            tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:02d}".format(i+1)+" of 72; "+str(len(tmp))+ " entries")
        else:
            job = Gaia.launch_job_async(f"select designation, source_id, ref_epoch, ra, dec, "
                                            f"parallax, parallax_error, pmra, pmdec, "
                                            f"phot_g_mean_mag, "
                                            f"phot_bp_mean_mag, "
                                            f"phot_rp_mean_mag, radial_velocity "
                                            f"from gaiadr3.gaia_source where "
                                            f"radial_velocity IS NOT NULL and ra >= {(5*i)} and "
                                            f"ra < {5*(i+1)}",
                                        dump_to_file=True,output_format = "fits",
                                        output_file=filename)
            print("Downloaded table "+"{:02d}".format(i+1)+" of 72")
            r = job.get_results()
#            tmp = parse_single_table(filename).to_table()
            tmp = Table.read(filename,format='fits')
            print("Read table "+"{:02d}".format(i+1)+" of 72; "+str(len(tmp))+ " entries")
        if i == 0:
            data_all = tmp
        else:
#            stackstart = time.time()
#            data_all = vstack([data_all,tmp])
            tables.append(tmp)
#            stackend = time.time()
#            print(f'append took {stackend-stackstart} seconds')
    stackstart = time.time()
    data_all = vstack(tables)
    stackend = time.time()
    print(f'vstack took {stackend-stackstart} seconds')
    end = time.time()
    print(f'Total {end-start} seconds')



            
if source_cat != 'eDR3' and source_cat != 'DR2' and source_cat != 'DR2_all' and source_cat != 'eDR3_all' and source_cat != 'DR3_all' and source_cat != 'DR3_500pc' and source_cat != 'DR3_1kpc':
    print('Specify correct DR')
    assert(False)
    
data_all.add_column(data_all['phot_bp_mean_mag'] - data_all['phot_rp_mean_mag'],name='BP_RP')      #colour BP_RP
data_all.add_column(data_all['phot_g_mean_mag']+5*np.log10(data_all['parallax']/100),name='M_G')   #absolute M_G

# tidy up some units
data_all['pmra'].unit = units.mas/units.yr
data_all['pmdec'].unit = units.mas/units.yr
data_all['radial_velocity'].unit = units.km/units.s

N_stars_all = len(data_all)

print('{:d} stars read'.format(N_stars_all))
# -

data_all = data_all[data_all['parallax'] > 0]
N_stars_all = len(data_all)
print('{:d} stars with positive parallax'.format(N_stars_all))

# +
coord = SkyCoord(data_all['ra'],data_all['dec'],distance=1000*units.pc/np.array(data_all['parallax']),
                 pm_ra_cosdec=data_all['pmra'],pm_dec=data_all['pmdec'],
                 radial_velocity=data_all['radial_velocity'],
                 frame='icrs').transform_to(Galactic)
coord.representation_type = 'cartesian'

# stuff for Mahalanobis distance
data_all.add_column(coord.u,name='u')
data_all.add_column(coord.v,name='v')
data_all.add_column(coord.w,name='w')
data_all.add_column(coord.U,name='U')
data_all.add_column(coord.V,name='V')
data_all.add_column(coord.W,name='W')
data_all.add_column(1000/data_all['parallax']*units.pc,name='d_Sol')

# add Sol at end of table (index -1)

data_all.add_row({'u':0*units.pc,'v':0*units.pc,'w':0*units.pc,'d_Sol':0*units.pc,
                      'U':0*units.km/units.s,'V':0*units.km/units.s,'W':0*units.km/units.s,
                      'designation':'Sol'})




# +
# colour and magnitude for Sol, from Casagrande+18 (DR2):
data_all[-1]['M_G'] = 4.67
data_all[-1]['BP_RP'] = 0.82

#Solar motion wrt LSR, from Schönrich+10:
U_Sol = 11.1 * units.km/units.s
V_Sol = 12.14 * units.km/units.s
W_Sol = 7.25 * units.km/units.s

data_all['U'] += U_Sol.value #???
data_all['V'] += V_Sol.value
data_all['W'] += W_Sol.value

# +
#coord
# position is (u,v,w)
# velocity is (U,V,W)
# this won't be confusing at all...
# -

# Now we set up our target list, check if the target is in Gaia DRn, and make sure we handle Sol properly

# +
# some global variables
d_query = 80 * units.pc #radius of sphere to query
N_thresh = 20 # k = N_thresh for k-NN calculation
rho_thr = 50 # if rescaled rho above this, cut from Gaussian mixture model and class as high-rho
N_models = 10 # number models for GMM
v_factor = 1.25 # check stars with v \in (v_target/v_factor,v_target*v_factor)
N_stars_min = 100 # min number of neighbours within d_query

#rough thin disc, thick disc, halo boundaries from Bensby+14
v_thin = 50
v_thick_min = 70
v_thick_max = 180
v_halo = 200

# +
if not 'd_target' in data_all.colnames:
    data_all.add_column(np.zeros(len(data_all))*units.pc,name='d_target')

data_M_G_9 = data_all[np.where(data_all['M_G'] <= 9.0)]
data_M_G_8 = data_all[np.where(data_all['M_G'] <= 8.0)]


# +
class Target:
    
    def __init__(self,name_short,gaia_id):
        
        self.gaia_id = gaia_id
        self.name_short = name_short
        self.N_sample = 0
        self.P_1comp = (np.nan,np.nan)
        self.P_1comp_v = (np.nan,np.nan)
   
        return

    def get_neighbours(self,data_all=data_all):
        
        if self.gaia_id is None:
            print(self.name_short+": no Gaia id")
            self.data = None
            self.N_stars = 0
            return
        
        self.target = data_all[data_all['designation'] == self.gaia_id]    #the data of the target star: position, velocity, etc. 
        
        if len(self.target) == 0:
            self.data = None
            self.N_stars = 0
            return

        self.folder = 'results/' + self.name_short.replace(' ','') + '/' + source_cat + '/'

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.folder = self.folder+self.name_short.replace(' ','')+'_'+source_cat+'_'



        self.u = self.target['u']
        self.v = self.target['v']
        self.w = self.target['w']

        #calculate distance between target stars and all other stars
        d = np.sqrt((self.u-data_all['u'])**2 + 
                    (self.v-data_all['v'])**2 + 
                    (self.w-data_all['w'])**2) * units.pc

        if 'd_target' in data_all.colnames:
            data_all['d_target'] = d
        else:
            data_all.add_column(d,name='d_target')

        
        self.data = data_all[d <= d_query]               #only select stars within 80 pc of target star
        
        self.i_target = np.where(self.gaia_id == self.data['designation'])[0][0]

        self.N_stars = len(self.data)
        print(self.name_short+":     Sample: "+str(self.N_stars)+" of "+str(N_stars_all)+" stars")

        return

    def distance_histograms(self):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=[10,4])

        ax1.hist(self.data['d_Sol'])
        ax1.plot([self.target['d_Sol']]*2,ax1.get_ylim(),'--k',label=self.name_short)
        ax1.set_xlabel('distance to Sol [pc]')
        ax1.set_ylabel('# stars')
        ax1.legend()

        ax2.hist(self.data['d_target'])
        ax2.set_xlabel('distance to '+self.name_short+' [pc]')
        ax2.set_ylabel('# stars')

        plt.savefig(self.folder+'distance_histograms.pdf')
        plt.close()
        
        return

    def distance_histograms_fine(self,M_G_lim=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=[10,10])
        if self.target['parallax'] > 0:
            print("self.target['d_Sol']: ", self.target['d_Sol'].value)
            print("d_query.value: ", d_query.value)
            print("Computed value: ", [np.array([0]), self.target['d_Sol'].value - d_query.value])
            #min_bin = np.max([0,self.target['d_Sol']-d_query.value]) 
            min_bin = np.max([np.array([0]), self.target['d_Sol'].value - d_query.value])
            max_bin = self.target['d_Sol']+d_query.value
        else:
            min_bin = 0.
            max_bin = d_query.value
        n_bins = int(np.floor(max_bin-min_bin)+1)
        d = np.array(np.linspace(min_bin,max_bin,n_bins)).reshape(n_bins)
        V = np.zeros(n_bins-1)
        V_Sol = np.zeros(n_bins-1)

#don't calculate first bin to avoid some numerical artefacts
        for i in range(n_bins-2):
            V[i+1] = (4*np.pi/3)*d[i+2]**3 - (4*np.pi/3)*d[i+1]**3
            if (d_query.value > self.target['d_Sol']) and (d[i] < d_query.value - self.target['d_Sol']):
                V_Sol[i+1] = 4*np.pi*d[i+1]**2*(d[i+2]-d[i+1])
            else:
                V_Sol[i+1] = 2*np.pi*d[i+1]**2 * (1 - (d[i+1]**2+self.target['d_Sol']**2-d_query.value**2)/
                                                  (2*d[i+1]*self.target['d_Sol']))*(d[i+2]-d[i+1])
        
        if M_G_lim is None:
            index = [True]*len(self.data)
            filesuf = ''
        else:
            index = np.where(self.data['M_G'] <= M_G_lim)
            filesuf = '_MGlim'+'{:04.1f}'.format(M_G_lim)
            
        N = ax1.hist(self.data['d_Sol'][index],bins=d)
        ax1.plot([self.target['d_Sol']]*2,ax1.get_ylim(),'--k',label=self.name_short)
        ax1.set_xlabel('distance to Sol [pc]')
        ax1.set_ylabel('# stars')
        ax1.legend()

        ax2.plot(d[1:],N[0]/V_Sol)
        ax2.plot([self.target['d_Sol']]*2,ax2.get_ylim(),'--k',label=self.name_short)
        ax2.set_xlabel('distance to Sol [pc]')
        ax2.set_ylabel('stellar density [pc^-3]')

        n_bins = int(np.floor(d_query.value)+1)
        d = np.linspace(0,80,n_bins)
        V = np.zeros(n_bins-1)
        for i in range(n_bins-1):
            V[i] = (4*np.pi/3)*d[i+1]**3 - (4*np.pi/3)*d[i]**3

        N = ax3.hist(self.data['d_target'][index],bins=d)
        ax3.set_xlabel('distance to '+self.name_short+' [pc]')
        ax3.set_ylabel('# stars')

        ax4.plot(d[1:],N[0]/V)
        ax4.set_xlabel('distance to '+self.name_short+' [pc]')
        ax4.set_ylabel('stellar density [pc$^{-3}$]')
        ax4.set_yscale('log')

        plt.savefig(self.folder+'distance_histograms_fine'+filesuf+'.pdf')
        plt.close()
        
        return

    def magnitude_histograms(self):
        plt.figure(figsize=[5,4])
        plt.hist(self.data['phot_g_mean_mag'])
        plt.plot([self.target['phot_g_mean_mag']]*2,plt.ylim(),'--k',label=self.name_short)
        plt.xlabel('G magnitude')
        plt.ylabel('# stars')
        plt.legend()
        plt.savefig(self.folder+'magnitude_histograms.pdf')
        plt.close()
        
        return

    def parallax_error_histograms(self):

        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[15,4])

        ax1.hist(self.data['parallax']/self.data['parallax_error'])
        ax1.plot([self.target['parallax']/self.target['parallax_error']]*2,ax1.get_ylim(),'--k',label=self.name_short)
        ax1.set_xlabel('parallax over error')
        ax1.set_ylabel('# stars')
        ax1.legend()

        ax2.hist(self.data['parallax']/self.data['parallax_error'],bins=np.linspace(0,1000,11))
        ax2.plot([self.target['parallax']/self.target['parallax_error']]*2,ax2.get_ylim(),'--k',label=self.name_short)
        ax2.set_xlabel('parallax over error')
        ax2.set_ylabel('# stars')
        ax2.set_xlim([0,600])

        ax3.hist(self.data['parallax']/self.data['parallax_error'],bins=np.linspace(0,100,11))
        ax3.plot([self.target['parallax']/self.target['parallax_error']]*2,ax3.get_ylim(),'--k',label=self.name_short)
        ax3.set_xlabel('parallax over error')
        ax3.set_ylabel('# stars')
        ax3.set_xlim([0,60])

        plt.savefig(self.folder+'parallax_error_histograms.pdf')
        plt.close()
        
        return

    def distance_Gmag(self):

        plt.figure(figsize=[5,4])

        plt.scatter(self.data['d_Sol'],self.data['phot_g_mean_mag'],alpha=0.1)
        plt.scatter(self.target['d_Sol'],self.target['phot_g_mean_mag'],c='k',marker='*',label=self.name_short)
        plt.xlabel('distance to Sol [pc]')
        plt.ylabel('G mag')
        plt.legend()

        plt.savefig(self.folder+'distance_Gmag.pdf')
        plt.close()

        return

    def distance_M_G(self):

        plt.figure(figsize=[5,4])

        plt.scatter(self.data['d_Sol'],self.data['M_G'],alpha=0.1)
        plt.scatter(self.target['d_Sol'],self.target['M_G'],c='k',marker='*',label=self.name_short)
        plt.xlabel('distance to Sol [pc]')
        plt.ylabel('absolute $M_G$ [mag]')
        plt.legend()

        plt.savefig(self.folder+'distance_M_G.pdf')
        plt.close()

        return

    def CMD(self):
        plt.figure(figsize=[5,4])

        plt.scatter(self.data['BP_RP'],self.data['M_G'],alpha=0.1)
        plt.scatter(self.target['BP_RP'],self.target['M_G'],c='k',marker='*',label=self.name_short)

        plt.xlabel('BP-RP')
        plt.ylabel('absolute $M_G$ [mag]')
        plt.gca().invert_yaxis()

        plt.legend()

        plt.savefig(self.folder+'CMD.pdf')
        plt.close()

        return

    def CMD_hist(self):
        
        plt.figure(figsize=[5,4])

        good = np.logical_and(~ np.isnan(self.data['BP_RP']),~ np.isnan(self.data['M_G']))
    
        hist, xedge, yedge, pcm = plt.hist2d(self.data['BP_RP'][good],self.data['M_G'][good],
                                             bins=100,norm=mpl.colors.LogNorm())
        plt.scatter(self.target['BP_RP'],self.target['M_G'],c='k',marker='*',label=self.name_short)

        plt.xlabel('BP-RP')
        plt.ylabel('absolute $M_G$ [mag]')
        plt.gca().invert_yaxis()

        plt.legend()

        plt.colorbar(pcm,label='stars per bin')

        plt.savefig(self.folder+'CMD_hist.pdf')
        plt.close()
        
        return

    def total_PM_histograms(self):
        if source_cat == 'eDR3' or source_cat == 'eDR3_all':
            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[15,4])

            ax1.hist(self.data['pm'])
            ax1.set_xlabel('total PM [mas/yr]')
            ax1.set_ylabel('# stars')

            ax2.hist(self.data['pm'],bins=np.linspace(0,1000,11))
            ax2.set_xlabel('total PM [mas/yr]')
            ax2.set_ylabel('# stars')
            ax2.set_xlim([0,1000])

            ax3.hist(self.data['pm'],bins=np.linspace(0,100,11))
            ax3.set_xlabel('total PM [mas/yr]')
            ax3.set_ylabel('# stars')
            ax3.set_xlim([0,100])

            plt.savefig(self.folder+'total_PM_histograms.pdf')
            plt.close()

        return

    def RV_histograms(self):

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=[10,4])

        ax1.hist(self.data['radial_velocity'])
        ax1.plot([self.target['radial_velocity']]*2,ax1.get_ylim(),'--k',label=self.name_short)
        ax1.set_xlabel('RV [km/s]')
        ax1.set_ylabel('# stars')
        ax1.legend()

        ax2.hist(self.data['radial_velocity'],bins=np.linspace(-100,100,11))
        ax2.plot([self.target['radial_velocity']]*2,ax2.get_ylim(),'--k',label=self.name_short)
        ax2.set_xlabel('RV [km/s]')
        ax2.set_ylabel('# stars')
        ax2.set_xlim([-100,100])

        plt.savefig(self.folder+'RV_histograms.pdf')
        plt.close()

        return

    def X_Y(self):
        plt.figure(figsize=[5,4])

        plt.plot(self.data['u'],self.data['v'],'.',alpha=0.02)
        plt.scatter(self.target['u'],self.target['v'],c='k',marker='*',label=self.name_short)
        plt.xlabel('X [pc]')
        plt.ylabel('Y [pc]')
        plt.axis('equal')
        plt.legend()

        plt.savefig(self.folder+'X_Y.pdf')
        plt.close()

        return

    def Toomre(self):

        plt.figure(figsize=[5,4])

        x = np.linspace(-400,400,1001)
        plt.plot(x,np.sqrt(100**2-x**2),'k')
        plt.plot(x,np.sqrt(200**2-x**2),'k')
        plt.plot(x,np.sqrt(300**2-x**2),'k')
        plt.plot(x,np.sqrt(400**2-x**2),'k')

        plt.plot(self.data['U'],np.sqrt(self.data['V']**2+self.data['W']**2),'.',alpha=0.1)
        plt.scatter(self.target['U'],np.sqrt(self.target['V']**2+self.target['W']**2),c='k',marker='*',
                    label=self.name_short,zorder=9)

        plt.xlabel('$U$ [km/s]')
        plt.ylabel('\sqrt{V^2+W^2}')
        plt.axis('equal')
        plt.xlim(np.min(self.data['U']),np.max(self.data['U']))
        plt.legend()

        plt.savefig(self.folder+'Toomre.pdf')
        plt.close()

        return

    def get_pos_6D(self):
        start_time = time.time()
        
        self.pos_6D = np.array([self.data['u'],self.data['v'],self.data['w'],
                                self.data['U'],self.data['V'],self.data['W']])

        self.Cov = np.cov(self.pos_6D)

        end_time = time.time()

        #print(f"get_pos_6D took {end_time-start_time} seconds")
    
        return

    def distances(self,i):
        initial_time = time.time()
        
        if self.pos_6D is None:
            self.get_pos_6D()
        
        #D_M = np.zeros(self.N_stars)

        c_inv = np.linalg.inv(self.Cov)

        start_time = time.time()
        
        #new code by Tilde:
        D_M = scipy.spatial.distance.cdist(self.pos_6D[:,i].reshape(1, -1), self.pos_6D.T, metric = 'mahalanobis', VI = c_inv).flatten()
            #this took ~0.002 seconds

        
        #for j in range(self.N_stars):
#        if i % N_stars == -1 % N_stars: # we are Sol
            #D_M[j] = scipy.spatial.distance.mahalanobis(self.pos_6D[:,i],self.pos_6D[:,j], c_inv)
            #this took ~0.4 seconds
        
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        #print("\n"+f"mahalanobis took {end_time - start_time} seconds")
        
        D_u = self.data['u'] - self.data['u'][i]
        D_v = self.data['v'] - self.data['v'][i]
        D_w = self.data['w'] - self.data['w'][i]
        D_U = self.data['U'] - self.data['U'][i]
        D_V = self.data['V'] - self.data['V'][i]
        D_W = self.data['W'] - self.data['W'][i]
    
        D_phys = np.sqrt(D_u**2 + D_v**2 + D_w**2)
        D_vel = np.sqrt(D_U**2 + D_V**2 + D_W**2)
    
        dist = {'D_M':D_M,'D_phys':D_phys,'D_vel':D_vel,'D_u':D_u,'D_v':D_v,'D_w':D_w,'D_U':D_U,'D_V':D_V,'D_W':D_W}

        final_time = time.time()

        #print(f"distances took {final_time-initial_time} seconds \n")
        
        return dist

    def get_dist_target(self):

        start_time = time.time()
        
        self.dist_target = self.distances(self.i_target)   #calculate distance from target to star within 80 pc

        end_time = time.time()

        #print(f"get_dist_target took {end_time - start_time} seconds")
        
        return
    
    
    def D_M_histograms(self):

        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=[15,4])

        ax1.hist(self.dist_target['D_M'])
        ax1.set_xlabel('Mahalanobis distance to '+self.name_short)
        ax1.set_ylabel('# stars')

        ax2.hist(self.dist_target['D_M'],bins=np.linspace(0,5,11))
        ax2.set_xlabel('Mahalanobis distance to '+self.name_short)
        ax2.set_ylabel('# stars')

        ax3.hist(self.dist_target['D_M'],bins=np.logspace(-1,2,51))
        ax3.set_xlabel('Mahalanobis distance to '+self.name_short)
        ax3.set_ylabel('# stars')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        plt.savefig(self.folder+'D_M_histograms.pdf')
        plt.close()
        
        return

    def Delta_v_histograms(self):

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

        ax1.hist(self.dist_target['D_vel'])
        ax1.set_xlabel('Delta V from '+self.name_short+' [km/s]')
        ax1.set_ylabel('# stars')

        ax2.hist(self.dist_target['D_vel'],bins=np.linspace(0,100,11))
        ax2.set_xlabel('Delta V from '+self.name_short+' [km/s]')
        ax2.set_ylabel('# stars')

        plt.savefig(self.folder+'Delta_v_histograms.pdf')
        plt.close()

        return

    def D_phys_D_M(self):
        
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[15,4])

        ax1.scatter(self.dist_target['D_phys'],self.dist_target['D_M'],alpha=0.03)
        ax1.set_xlabel('physical distance to '+self.name_short+' [pc]')
        ax1.set_ylabel('Mahalanobis distance to '+self.name_short)

        ax2.scatter(self.dist_target['D_phys'],self.dist_target['D_M'],alpha=0.01)
        ax2.set_xlabel('physical distance to '+self.name_short+' [pc]')
        ax2.set_ylabel('Mahalanobis distance to '+self.name_short)
        ax2.set_ylim([0,5])

        ax3.scatter(self.dist_target['D_phys'],self.dist_target['D_M'],alpha=0.2)
        ax3.set_xlabel('physical distance to '+self.name_short+' [pc]')
        ax3.set_ylabel('Mahalanobis distance to '+self.name_short)
        ax3.set_ylim([0,1.5])

        plt.savefig(self.folder+'D_phys_D_M.pdf')
        plt.close()
        
        return

    def Delta_v_D_M(self):
        
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[15,4])

        ax1.scatter(self.dist_target['D_vel'],self.dist_target['D_M'],alpha=0.03)
        ax1.set_xlabel('Delta v from '+self.name_short+' [km/s]')
        ax1.set_ylabel('Mahalanobis distance to '+self.name_short)

        ax2.scatter(self.dist_target['D_vel'],self.dist_target['D_M'],alpha=0.01)
        ax2.set_xlabel('Delta v from '+self.name_short+' [km/s]')
        ax2.set_ylabel('Mahalanobis distance to '+self.name_short)
        ax2.set_ylim([0,5])
        ax2.set_xlim([0,150])

        ax3.scatter(self.dist_target['D_vel'],self.dist_target['D_M'],alpha=0.2)
        ax3.set_xlabel('Delta v from '+self.name_short+' [km/s]')
        ax3.set_ylabel('Mahalanobis distance to '+self.name_short)
        ax3.set_ylim([0,1.5])
        ax3.set_xlim([0,40])

        plt.savefig(self.folder+'Delta_v_D_M.pdf')
        plt.close()

        return

    def D_phys_Delta_v(self):

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=[10,4])

        ax1.scatter(self.dist_target['D_phys'],self.dist_target['D_vel'],alpha=0.03)
        ax1.set_xlabel('physical distance to '+self.name_short+' [pc]')
        ax1.set_ylabel('Delta v from '+self.name_short+' [km/s]')

        ax2.scatter(self.dist_target['D_phys'],self.dist_target['D_vel'],alpha=0.01)
        ax2.set_xlabel('physical distance to '+self.name_short+' [pc]')
        ax2.set_ylabel('Delta v from '+self.name_short+' [km/s]')
        ax2.set_ylim([0,150])

        plt.savefig(self.folder+'D_phys_Delta_v.pdf')
        plt.close()

        return

    def get_close(self,dist,j,N=20,dump_to_file=False):
        closest = np.argsort(dist['D_M'])
    
        filename = self.folder+'20_closest.txt'
    
        if dump_to_file:
            with open(filename,'w') as f:
                
                print('Star: '+self.data['designation'][j],file=f)
                
                print('(u,v,w) =' + (' {:8.3f}'*3).format(self.data['u'][j],self.data['v'][j],self.data['w'][j]) + 
                      '  [pc]',file=f)
                print('(U,V,W) =' + (' {:8.3f}'*3).format(self.data['U'][j],self.data['V'][j],self.data['W'][j]) + 
                      '  [km/s]',file=f)
                print('\n',file=f)
                print(("{:^6s} {:^29s}" + " {:>8s}"*9).format("id","Gaia id","D_M","D_phys",
                                                              "D_u","D_v","D_w","D_vel","D_U","D_V","D_W"),file=f)
                print(("{:^6s} {:^29s}" + " {:>8s}"*9).format("","","","pc","pc","pc","pc",
                                                              "km/s","km/s","km/s","km/s"),file=f)
                print("-"*90,file=f)
                for i in closest[1:1+N]:
                    print(("{:06d} {:29s}"+" {:8.3f}"*9).format(i,self.data['designation'][i],dist['D_M'][i],
                                                                    dist['D_phys'][i],
                                                                    dist['D_u'][i],
                                                                    dist['D_v'][i],
                                                                    dist['D_w'][i],
                                                                    dist['D_vel'][i],
                                                                    dist['D_U'][i],
                                                                    dist['D_V'][i],
                                                                    dist['D_W'][i]),file=f)


        return closest

    def get_close_target(self,dump_to_file=False):
        
        self.closest_target = self.get_close(self.dist_target,self.i_target,dump_to_file=dump_to_file)
        
        return

    def get_lt_40pc(self):
        self.lt_40pc = (np.where(np.logical_and(self.dist_target['D_phys'] < d_query/2,
                                                self.data['designation'] != self.target['designation'])))[0]
  
    
        return

    def set_seed(self):
        
        self.seed_file = self.folder+'seed'
        if os.path.exists(self.seed_file):
            with open(self.seed_file,'r+') as f:
                self.timestamp = int(f.read())
            self.rng = np.random.default_rng(self.timestamp)
            self.restore_from_save = True
        else:
    # use timestamp in ms
            self.timestamp = int(time.time() * 1000)
            self.rng = np.random.default_rng(self.timestamp)
            with open(self.seed_file,'w') as f:
                f.write(str(self.timestamp))
            self.restore_from_save = False

        return

    def get_random_sample(self):
        self.N_sample = min([600,len(self.lt_40pc)])

        self.sample = self.rng.choice(self.lt_40pc,self.N_sample,replace=False)
        
        self.sample_v = np.sqrt(self.data['U'][self.sample]**2 + self.data['V'][self.sample]**2 +
                                self.data['W'][self.sample]**2)
        self.target_v = np.sqrt(self.target['U']**2 + self.target['V']**2 + self.target['W']**2)
    
        return

    def get_sample_distances(self):

        dist = []
        close = []

        self.savefile = self.folder+'densities.txt'
        if self.restore_from_save and os.path.exists(self.savefile):
    #restore from folder+'densities.txt'
            tmp = ascii.read(self.savefile,format='fixed_width_no_header',data_start=6,delimiter='|',
                             names=('Gaia id','rho','d','u','v','w','U','V','W'))
            self.rho_20_target = tmp[0]['rho']
            self.rho_20_t = tmp[1:]['rho']
            self.d_20_target = tmp[0]['d']
            self.d_20 = tmp[1:]['d']
            print(self.name_short+': restored sample from save')
        else:
            print(self.name_short+': generating MC sample')
            for i in range(self.N_sample):
                dist.append(self.distances(self.sample[i]))
#                close.append(self.get_close(dist[i],self.sample[i]))

            #calculate the mahalanobis distance for the 20th nearest neighbour for each of the 600 neighbours 
            self.d_20 = np.array([np.sort(d['D_M'])[N_thresh] if len(d['D_M']) >= N_thresh else np.inf for d in dist])
            self.rho_20 = N_thresh * self.d_20**(-6)
            self.rho_20_t = self.rho_20/np.median(self.rho_20)

            #calculate the mahalanobis distance to the 20th nearest neighbour for the target star
            if len(self.dist_target['D_M']) >= N_thresh:
                self.d_20_target = np.sort(self.dist_target['D_M'])[N_thresh]
            else:
                self.d_20_target = np.inf
            self.rho_20_target = N_thresh * self.d_20_target**(-6) / np.median(self.rho_20)
        

        return

    def sample_D_M_rho_histograms(self):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,10))

        ax1.hist(self.d_20)
        lim = ax1.get_ylim()
        ax1.plot([self.d_20_target,self.d_20_target],lim,'--k',label=self.name_short+' value')
        ax1.set_xlabel('$D_\mathrm{M}$ to 20th nearest neighbour')
        ax1.set_ylabel('# stars')
        ax1.legend()

        ax2.hist(self.d_20,bins=np.logspace(np.log10(np.min(self.d_20)),np.log10(np.max(self.d_20)),26))
        lim = ax2.get_ylim()
        ax2.plot([self.d_20_target,self.d_20_target],lim,'--k',label=self.name_short+' value')
        ax2.set_xscale('log')
        ax2.set_xlabel('$D_\mathrm{M}$ to 20th nearest neighbour')
        ax2.set_ylabel('# stars')

        ax3.hist(self.rho_20_t)
        lim = ax3.get_ylim()
        ax3.plot([self.rho_20_target,self.rho_20_target],lim,'--k',label=self.name_short+' value')
        ax3.set_xlabel(r'$\rho_{20}$ [rescaled]')
        ax3.set_ylabel('# stars')
        ax3.legend()

        ax4.hist(self.rho_20_t,bins=np.logspace(np.log10(np.min(self.rho_20_t)),np.log10(np.max(self.rho_20_t)),26))
        lim = ax4.get_ylim()
        ax4.plot([self.rho_20_target,self.rho_20_target],lim,'--k',label=self.name_short+' value')
        ax4.set_xscale('log')
        ax4.set_xlabel(r'$\rho_{20}$ [rescaled]')
        ax4.set_ylabel('# stars')

        plt.savefig(self.folder+'sample_D_M_rho_histograms.pdf')
        plt.close()
        
        return

# Gaussian mixture stuff adapted from https://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html

    def gauss(self):

        self.log_rho = np.log10(self.rho_20_t).reshape(-1,1)
        self.log_rho_target = np.log10(self.rho_20_target)

        clean = np.logical_and(np.abs(self.log_rho - np.mean(self.log_rho)) <= 2*np.std(self.log_rho),
                               self.log_rho <= np.log10(rho_thr))

        models = [None] * N_models
        self.x_rho = np.linspace(np.min(self.log_rho),np.max(self.log_rho),101).reshape(-1,1)
        self.pdf = [None] * N_models
        self.AIC = np.zeros(N_models) * np.nan
        self.BIC = np.zeros(N_models) * np.nan

        self.max_comp_rho = min([N_models,np.sum(clean)])
        
        for i in range(self.max_comp_rho):
            models[i] = mixture.GaussianMixture(n_components=i+1,
                                                random_state=self.timestamp%(int(2**32))).fit(self.log_rho[clean].reshape(-1,1))
            self.pdf[i] = np.exp(models[i].score_samples(self.x_rho)).reshape(-1,1)
            self.AIC[i] = models[i].aic(self.log_rho[clean].reshape(-1,1))
            self.BIC[i] = models[i].bic(self.log_rho[clean].reshape(-1,1))

        if models[1] is not None:
            order = np.argsort(models[1].means_[:,0])

            responsibilities_smooth = (models[1].predict_proba(self.x_rho.reshape(-1, 1)))[:,order]
            self.pdf_individual = responsibilities_smooth * self.pdf[1]
            responsibilities_data = (models[1].predict_proba(self.log_rho.reshape(-1, 1)))[:,order]
            responsibilities_target = (models[1].predict_proba(self.log_rho_target.reshape(-1, 1)))[:,order]

            self.P_high = np.array(responsibilities_data[:,1]/(responsibilities_data[:,0]+responsibilities_data[:,1]))
            self.P_high[self.rho_20_t > rho_thr] = 1 # if rho>50 auto in high pop
            self.P_target = np.array(responsibilities_target[:,1]/(responsibilities_target[:,0]+
                                                                   responsibilities_target[:,1]))
            if self.rho_20_target > rho_thr:
                self.P_target = 1.0

            self.is_high = self.P_high > 0.84
            self.is_low = self.P_high < 0.16
            self.is_ind = np.logical_and(self.P_high <= 0.84,self.P_high >= 0.16)

            self.P_1comp = scipy.stats.kstest(self.log_rho[clean],'norm',
                                              args=(models[0].means_[0,0],models[0].covariances_[0,0,0]))
        else:
            self.P_high = np.nan
            self.P_target = np.nan
            self.is_high = np.nan
            self.is_ind = np.nan
            self.is_low = np.nan
            self.P_1comp = np.nan

            
        return

    def plot_gaussian_mixture(self,N_comps_to_plot=4):


        if self.P_target != np.nan:
    
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=[15,5])

            for i in range(min([self.max_comp_rho,N_comps_to_plot])):
                label = 'N={:2d}, ΔAIC = {:>8.2f},ΔBIC = {:>8.2f}'.format(i+1,self.AIC[i]-np.min(self.AIC),
                                                                          self.BIC[i]-np.min(self.BIC))
                ax1.plot(self.x_rho,self.pdf[i],label=label)
            ax1.hist(self.log_rho,density=True,color='k',alpha=0.5,bins=np.linspace(np.min(self.log_rho),
                                                                                    np.max(self.log_rho),26))
            ax1.plot([self.log_rho_target]*2,ax1.get_ylim(),'--k')
            ax1.legend(loc='upper left',fontsize='small')
            ax1.set_xlabel(r'$\log_{10} \rho$ [rescaled]')
            ax1.set_ylabel('pdf')

            ax2.plot(self.x_rho,self.pdf[0],label='N=1')
            ax2.plot(self.x_rho,self.pdf[1],label='N=2')
            ax2.plot(self.x_rho,self.pdf_individual[:,0],'b:',label='N=2 components')
            ax2.plot(self.x_rho,self.pdf_individual[:,1],'b:')
            ax2.hist(self.log_rho,density=True,color='k',alpha=0.5,bins=np.linspace(np.min(self.log_rho),
                                                                                    np.max(self.log_rho),26))
            ax2.plot([self.log_rho_target]*2,ax2.get_ylim(),'--k',label=self.name_short)
            ax2.set_xlabel(r'$\log_{10} \rho$ [rescaled]')
            ax2.set_ylabel('pdf')
            ax2.legend(fontsize='small')

            ax3.plot(self.log_rho,self.P_high,'o')
            ax3.plot([self.log_rho_target]*2,ax3.get_ylim(),'--k',label=self.name_short)
            ax3.set_xlabel(r'$\log_{10} \rho$ [rescaled]')
            ax3.set_ylabel('$P_\mathrm{high}$')

            plt.savefig(self.folder+'gaussian_mixture.pdf')
            plt.close()
        
        return

    def gauss_v(self):

        
        self.log_v = np.log10(self.sample_v).reshape(-1,1)
        self.log_v_target = np.log10(self.target_v)

        clean = np.abs(self.log_v - np.mean(self.log_v)) <= 2*np.std(self.log_v)

        models = [None] * N_models
        self.x_vel = np.linspace(np.min(self.log_v),np.max(self.log_v),101).reshape(-1,1)
        self.pdf_v = [None] * N_models
        self.AIC_v = np.zeros(N_models) * np.nan
        self.BIC_v = np.zeros(N_models) * np.nan

        self.max_comp_v = min([N_models,np.sum(clean)])
        
        for i in range(self.max_comp_v):
            models[i] = mixture.GaussianMixture(n_components=i+1,
                                                random_state=self.timestamp%(int(2**32))).fit(self.log_v[clean].reshape(-1,1))
            self.pdf_v[i] = np.exp(models[i].score_samples(self.x_vel)).reshape(-1,1)
            self.AIC_v[i] = models[i].aic(self.log_v[clean].reshape(-1,1))
            self.BIC_v[i] = models[i].bic(self.log_v[clean].reshape(-1,1))

        if models[1] is not None:

            order = np.argsort(models[1].means_[:,0])

            responsibilities_smooth = (models[1].predict_proba(self.x_vel.reshape(-1, 1)))[:,order]
            self.pdf_individual_v = responsibilities_smooth * self.pdf_v[1]
            responsibilities_data = (models[1].predict_proba(self.log_v.reshape(-1, 1)))[:,order]
            responsibilities_target = (models[1].predict_proba(self.log_v_target.reshape(-1, 1)))[:,order]

            self.P_high_v = np.array(responsibilities_data[:,1]/(responsibilities_data[:,0]+responsibilities_data[:,1]))
            self.P_target_v = np.array(responsibilities_target[:,1]/(responsibilities_target[:,0]+responsibilities_target[:,1]))

            self.is_high_v = self.P_high_v > 0.84
            self.is_low_v = self.P_high_v < 0.16
            self.is_ind_v = np.logical_and(self.P_high_v <= 0.84,self.P_high_v >= 0.16)
            self.P_1comp_v = scipy.stats.kstest(self.log_v[clean],'norm',
                                                args=(models[0].means_[0,0],models[0].covariances_[0,0,0]))

        else:
            self.P_high_v = np.nan
            self.P_target_v = np.nan
            self.is_high_v = np.nan
            self.is_ind_v = np.nan
            self.is_low_v = np.nan
            self.P_1comp_v = np.nan

    def save_densities(self):

        if source_cat == 'DR3_1kpc':
            densities_filename = 'results/data_densities_DR3.txt'
            densities_full_filename = 'results/data_densities_full_DR3.txt'
            
        if source_cat == 'DR2_all':
            densities_filename = 'results/data_densities_DR2.txt'
            densities_full_filename = 'results/data_densities_full_DR2.txt'
            

        if not os.path.exists(densities_filename):
            with open(densities_filename,'w') as f:
                print('name,    designation,    rho,    P_target,    P_1comp,    P_1compv,    N_sample', file=f)
        
        with open(densities_filename, "a") as f:
            print(f"{self.name_short},{self.data["designation"][self.i_target]},{self.rho_20_target},{self.P_target},{np.array([float(self.P_1comp.statistic), float(self.P_1comp.pvalue)])},{np.array([float(self.P_1comp_v.statistic), float(self.P_1comp_v.pvalue)])},{self.N_sample}", file=f)
        
        if not os.path.exists(densities_full_filename):
            with open(densities_full_filename,'w') as f:
                print('name,    designation,    rho,    P_high', file=f)

        
        with open(densities_full_filename, 'a') as f:
            for i in range(self.N_sample):
                print(f"{self.name_short},{self.data["designation"][self.sample[i]]},{self.rho_20[i]},{self.P_high[i]}", file=f)
        
        
        return

    def plot_gaussian_mixture_v(self,N_comps_to_plot=4):

        if N_comps_to_plot is None:
            N_comps_to_plot = 4
            
        if self.P_target != np.nan:
            
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=[15,5])

            for i in range(min([self.max_comp_rho,N_comps_to_plot])):
                label = 'N={:2d}, ΔAIC = {:>8.2f}, ΔBIC = {:>8.2f}'.format(i+1,self.AIC_v[i]-np.min(self.AIC_v),
                                                                           self.BIC_v[i]-np.min(self.BIC_v))
                ax1.plot(self.x_vel,self.pdf_v[i],label=label)
            ax1.hist(self.log_v,density=True,color='k',alpha=0.5,bins=np.linspace(np.min(self.log_v),
                                                                                  np.max(self.log_v),26))
            ax1.plot([self.log_v_target]*2,ax1.get_ylim(),'--k')
            ax1.legend(loc='upper left',fontsize='small')
            ax1.set_xlabel(r'$\log_{10} |\mathbf{v}|$ [km/s]')
            ax1.set_ylabel('pdf')

            ax2.plot(self.x_vel,self.pdf_v[0],label='N=1')
            ax2.plot(self.x_vel,self.pdf_v[1],label='N=2')
            ax2.plot(self.x_vel,self.pdf_individual_v[:,0],'b:',label='N=2 components')
            ax2.plot(self.x_vel,self.pdf_individual_v[:,1],'b:')
            ax2.hist(self.log_v,density=True,color='k',alpha=0.5,bins=np.linspace(np.min(self.log_v),
                                                                                  np.max(self.log_v),26))
            ax2.plot([self.log_v_target]*2,ax2.get_ylim(),'--k',label=self.name_short)
            ax2.set_xlabel(r'$\log_{10} |\mathbf{v}|$ [km/s]')
            ax2.set_ylabel('pdf')
            ax2.legend(fontsize='small')

            ax3.plot(self.log_v,self.P_high_v,'o')
            ax3.plot([self.log_v_target]*2,ax3.get_ylim(),'--k',label=self.name_short)
            ax3.set_xlabel(r'$\log_{10} |\mathbf{v}|$ [km/s]')
            ax3.set_ylabel('$P_\mathrm{high}$')

            plt.savefig(self.folder+'gaussian_mixture_v.pdf')
            plt.close()
        
        return

    def sample_D_phys_D_M_rho(self):

        x = self.dist_target['D_phys'][self.sample]
        y = self.d_20
        total_bins = 20

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=[15,8])

        ax1.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax1.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax1.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax1.plot(bins-delta/2,running_median,c='k',label='running median')
        ax1.plot([0,40],[self.d_20_target,self.d_20_target],'k--',label=self.name_short+' value')
        ax1.set_xlabel('distance to '+self.name_short+' [pc]')
        ax1.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')
        ax1.legend()

        ax2.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax2.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax2.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax2.set_ylim([0.4,1.1])
        ax2.plot(bins-delta/2,running_median,c='k')
        ax2.plot([0,40],[self.d_20_target,self.d_20_target],'k--')
        ax2.set_xlabel('distance to '+self.name_short+' [pc]')
        ax2.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        ax3.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax3.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax3.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax3.set_yscale('log')
        ax3.plot(bins-delta/2,running_median,c='k')
        ax3.plot([0,40],[self.d_20_target,self.d_20_target],'k--')
        ax3.set_xlabel('distance to '+self.name_short+' [pc]')
        ax3.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        y = self.rho_20_t
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax4.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax4.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax4.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax4.plot(bins-delta/2,running_median,c='k',label='running median')
        ax4.plot([0,40],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax4.set_xlabel('distance to '+self.name_short+' [pc]')
        ax4.set_ylabel(r'$\rho_{20}$ [rescaled]')
        ax4.legend()

        ax5.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax5.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax5.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax5.set_ylim([0.,10.])
        ax5.plot(bins-delta/2,running_median,c='k')
        ax5.plot([0,40],[self.rho_20_target,self.rho_20_target],'k--')
        ax5.set_xlabel('distance to '+self.name_short+' [pc]')
        ax5.set_ylabel(r'$\rho_{20}$ [rescaled]')

        ax6.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax6.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax6.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax6.set_yscale('log')
        ax6.plot(bins-delta/2,running_median,c='k')
        ax6.plot([0,40],[self.rho_20_target,self.rho_20_target],'k--')
        ax6.set_xlabel('distance to '+self.name_short+' [pc]')
        ax6.set_ylabel(r'$\rho_{20}$ [rescaled]')

        plt.savefig(self.folder+'sample_D_phys_D_M_rho.pdf')
        plt.close()
        
        return

    def sample_D_phys_D_M_rho_1panel(self):
        
        if self.dist_target is not None:
            x = self.dist_target['D_phys'][self.sample]
        else:
            self.get_dist_target()
            x = self.dist_target['D_phys'][self.sample]
            
        total_bins = 20

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)

        plt.figure(figsize=[5,4])

        y = self.rho_20_t
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        plt.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='$P_\mathrm{high}>0.84$')
        plt.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='$0.16\leq P_\mathrm{high}\leq0.84$')
        plt.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='$P_\mathrm{high}<0.16$')
        plt.yscale('log')
        plt.plot(bins-delta/2,running_median,c='k',label='Running median')
        plt.plot([0,40],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        plt.xlabel('distance to '+self.name_short+' [pc]')
        plt.ylabel(r'$\rho_{20}$ [rescaled]')
        plt.legend()

        plt.savefig(self.folder+'sample_D_phys_D_M_rho_1panel.pdf')
        plt.close()
        
        return

    def sample_Delta_v_D_M_rho(self):
        
        x = self.dist_target['D_vel'][self.sample]
        y = self.d_20
        total_bins = 40

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=[15,8])

        ax1.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax1.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax1.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax1.plot(bins-delta/2,running_median,c='k',label='running median')
        ax1.plot([0,300],[self.d_20_target,self.d_20_target],'k--',label=self.name_short+' value')
        ax1.set_xlabel('$|\Delta v|$ from '+self.name_short+' [km/s]')
        ax1.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')
        ax1.legend()

        ax2.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax2.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax2.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax2.set_xlim([0,100])
        ax2.set_ylim([0.4,1.1])
        ax2.plot(bins-delta/2,running_median,c='k')
        ax2.plot([0,100],[self.d_20_target,self.d_20_target],'k--')
        ax2.set_xlabel('$|\Delta v|$ from '+self.name_short+' [km/s]')
        ax2.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        ax3.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax3.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax3.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.plot(bins-delta/2,running_median,c='k')
        ax3.plot([0,300],[self.d_20_target,self.d_20_target],'k--')
        ax3.set_xlabel('$|\Delta v|$ from '+self.name_short+' [km/s]')
        ax3.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        y = self.rho_20_t
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax4.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax4.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax4.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax4.plot(bins-delta/2,running_median,c='k',label='running median')
        ax4.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax4.set_xlabel('$|\Delta v|$ from '+self.name_short+' [km/s]')
        ax4.set_ylabel(r'$\rho_{20}$ [rescaled]')
        ax4.legend()

        ax5.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax5.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax5.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax5.set_xlim([0,100])
        ax5.set_ylim([0,10])
        ax5.plot(bins-delta/2,running_median,c='k')
        ax5.plot([0,100],[self.rho_20_target,self.rho_20_target],'k--')
        ax5.set_xlabel('$|\Delta v|$ from '+self.name_short+' [km/s]')
        ax5.set_ylabel(r'$\rho_{20}$ [rescaled]')

        ax6.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax6.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax6.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.plot(bins-delta/2,running_median,c='k')
        ax6.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--')
        ax6.set_xlabel('$|\Delta v|$ from '+self.name_short+' [km/s]')
        ax6.set_ylabel(r'$\rho_{20}$ [rescaled]')

        plt.savefig(self.folder+'sample_Delta_v_D_M_rho.pdf')
        plt.close()
        
        return

    def sample_abs_v_D_M_rho(self):

        x = self.sample_v
        x_t = self.target_v
        y = self.d_20
        total_bins = 40
        if self.P_target < 0.16:
            c_t = 'b'
        if self.P_target > 0.84:
            c_t = 'r'
        if self.P_target >= 0.16 and self.P_target <= 0.84:
            c_t = 'k'

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=[15,8])

        ax1.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax1.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax1.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax1.scatter([x_t],[self.d_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax1.plot(bins-delta/2,running_median,c='k',label='running median')
        ax1.plot([0,300],[self.d_20_target,self.d_20_target],'k--',label=self.name_short+' value')
        ax1.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax1.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        xmin = np.max((ax1.get_xlim()[0],0))
        xmax = ax1.get_xlim()[1]
        ymin = ax1.get_ylim()[0]
        ymax = ax1.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),v_thick_max-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        halo = Rectangle((v_halo,ymin),xmax-v_halo,ymax-ymin,facecolor='g',alpha=0.1,zorder=-10)
        ax1.add_patch(thin)
        ax1.add_patch(thick)
        ax1.add_patch(halo)

        ax1.legend()

        ax2.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax2.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax2.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax2.scatter([x_t],[self.d_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax2.set_xlim([0,100])
        ax2.set_ylim([0.4,1.1])
        ax2.plot(bins-delta/2,running_median,c='k')
        ax2.plot([0,100],[self.d_20_target,self.d_20_target],'k--')
        ax2.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax2.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        xmin = np.max((ax2.get_xlim()[0],0))
        xmax = ax2.get_xlim()[1]
        ymin = ax2.get_ylim()[0]
        ymax = ax2.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),xmax-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        ax2.add_patch(thin)
        ax2.add_patch(thick)

        ax3.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax3.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax3.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax3.scatter([x_t],[self.d_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.plot(bins-delta/2,running_median,c='k')
        ax3.plot([0,300],[self.d_20_target,self.d_20_target],'k--')
        ax3.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax3.set_ylabel('$D_\mathrm{M}$ to 20th nearest neighbour')

        xmin = np.max((ax3.get_xlim()[0],0))
        xmax = ax3.get_xlim()[1]
        ymin = ax3.get_ylim()[0]
        ymax = ax3.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),v_thick_max-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        halo = Rectangle((v_halo,ymin),xmax-v_halo,ymax-ymin,facecolor='g',alpha=0.1,zorder=-10)
        ax3.add_patch(thin)
        ax3.add_patch(thick)
        ax3.add_patch(halo)

        y = self.rho_20_t
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax4.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax4.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax4.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax4.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax4.plot(bins-delta/2,running_median,c='k',label='running median')
        ax4.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax4.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax4.set_ylabel(r'$\rho_{20}$ [rescaled]')


        xmin = np.max((ax4.get_xlim()[0],0))
        xmax = ax4.get_xlim()[1]
        ymin = ax4.get_ylim()[0]
        ymax = ax4.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),v_thick_max-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        halo = Rectangle((v_halo,ymin),xmax-v_halo,ymax-ymin,facecolor='g',alpha=0.1,zorder=-10)
        ax4.add_patch(thin)
        ax4.add_patch(thick)
        ax4.add_patch(halo)

        ax4.legend()

        ax5.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax5.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax5.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax5.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax5.set_xlim([0,100])
        ax5.set_ylim([0,10])
        ax5.plot(bins-delta/2,running_median,c='k')
        ax5.plot([0,100],[self.rho_20_target,self.rho_20_target],'k--')
        ax5.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax5.set_ylabel(r'$\rho_{20}$ [rescaled]')

        xmin = np.max((ax5.get_xlim()[0],0))
        xmax = ax5.get_xlim()[1]
        ymin = ax5.get_ylim()[0]
        ymax = ax5.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),xmax-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        ax5.add_patch(thin)
        ax5.add_patch(thick)

        ax6.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax6.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax6.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax6.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.plot(bins-delta/2,running_median,c='k')
        ax6.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--')
        ax6.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax6.set_ylabel(r'$\rho_{20}$ [rescaled]')

        xmin = np.max((ax6.get_xlim()[0],0))
        xmax = ax6.get_xlim()[1]
        ymin = ax6.get_ylim()[0]
        ymax = ax6.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),v_thick_max-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        halo = Rectangle((v_halo,ymin),xmax-v_halo,ymax-ymin,facecolor='g',alpha=0.1,zorder=-10)
        ax6.add_patch(thin)
        ax6.add_patch(thick)
        ax6.add_patch(halo)

        plt.savefig(self.folder+'sample_abs_v_D_M_rho.pdf')
        plt.close()
        
        return

    def sample_abs_v_D_M_rho_1panel(self):


        x = self.sample_v
        x_t = self.target_v
        total_bins = 40
        if self.P_target < 0.16:
            c_t = 'b'
        if self.P_target > 0.84:
            c_t = 'r'
        if self.P_target >= 0.16 and self.P_target <= 0.84:
            c_t = 'k'

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)

        fig = plt.figure(figsize=[5,4])

        ax = fig.gca()

        y = self.rho_20_t
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='$P_\mathrm{high}>0.84$')
        ax.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='$0.16\leq P_\mathrm{high}\leq0.84$')
        ax.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='$P_\mathrm{high}<0.16$')
        ax.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(bins-delta/2,running_median,c='k',label='Running median')
        ax.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax.set_xlabel('$\sqrt{U^2 + V^2 + W^2}$ [km/s]')
        ax.set_ylabel(r'$\rho_{20}$ [rescaled]')

        xmin = np.max((ax.get_xlim()[0],0))
        xmax = ax.get_xlim()[1]
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        thin = Rectangle((xmin,ymin),v_thin-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((v_thick_min,ymin),v_thick_max-v_thick_min,ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        halo = Rectangle((v_halo,ymin),xmax-v_halo,ymax-ymin,facecolor='g',alpha=0.1,zorder=-10)
        ax.add_patch(thin)
        ax.add_patch(thick)
        ax.add_patch(halo)
        ax.text(xmin+5,ymax/4,'thin disc',color='g')
        ax.text(v_thick_min,ymax/4,'thick disc',color='g')
        ax.text(v_halo,ymax/4,'halo',color='g')

        ax.legend(fontsize='small')

        plt.savefig(self.folder+'sample_abs_v_D_M_rho_1panel.pdf')
        plt.close()
        
        return

    def UVW_rho(self):

        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=[15,4])

        if self.P_target < 0.16:
            c_t = 'b'
        if self.P_target > 0.84:
            c_t = 'r'
        if self.P_target >= 0.16 and self.P_target <= 0.84:
            c_t = 'k'

        x = np.abs(self.data['U'][self.sample])
        x_t = np.abs(self.target['U'])
        y = self.rho_20_t
        total_bins = 40

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax1.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax1.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax1.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax1.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax1.plot(bins-delta/2,running_median,c='k',label='running median')
        ax1.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax1.set_xlabel('|U| [km/s]')
        ax1.set_ylabel('rho_20 [rescaled]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()

        x = np.abs(self.data['V'][self.sample])
        x_t = np.abs(self.target['V'])
        y = self.rho_20_t
        total_bins = 40

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax2.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax2.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax2.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax2.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax2.plot(bins-delta/2,running_median,c='k',label='running median')
        ax2.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax2.set_xlabel('|V| [km/s]')
        ax2.set_ylabel('rho_20 [rescaled]')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        x = np.abs(self.data['W'][self.sample])
        x_t = np.abs(self.target['W'])
        y = self.rho_20_t
        total_bins = 40

        bins = np.linspace(x.min(),x.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(x,bins)
        running_median = [np.median(y[idx==k]) for k in range(total_bins)]

        ax3.scatter(x[self.is_high],y[self.is_high],c='r',alpha=0.2,label='P_high>0.84')
        ax3.scatter(x[self.is_ind],y[self.is_ind],c='k',alpha=0.2,label='0.16<=P_high<=0.84')
        ax3.scatter(x[self.is_low],y[self.is_low],c='b',alpha=0.2,label='P_high<0.16')
        ax3.scatter([x_t],[self.rho_20_target],c=c_t,edgecolor='yellow',marker='*',label=self.name_short,zorder=9)
        ax3.plot(bins-delta/2,running_median,c='k',label='running median')
        ax3.plot([0,300],[self.rho_20_target,self.rho_20_target],'k--',label=self.name_short+' value')
        ax3.set_xlabel('|W| [km/s]')
        ax3.set_ylabel('rho_20 [rescaled]')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        plt.savefig(self.folder+'_sample_UVW_rho.pdf')
        plt.close()
        
        return


    def sample_position_D_M_rho(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=[15,4])

        if self.P_target < 0.16:
            c_t = 'b'
        if self.P_target > 0.84:
            c_t = 'r'
        if self.P_target >= 0.16 and self.P_target <= 0.84:
            c_t = 'k'
        
        points = ax1.scatter(self.data['u'][self.sample],self.data['v'][self.sample],c=np.log10(self.d_20),alpha=0.5)
        ax1.scatter(self.target['u'],self.target['v'],c=np.log10(self.d_20_target),edgecolor='k',marker='*')
        ax1.axis('equal')
        ax1.set_xlabel('u [pc]')
        ax1.set_ylabel('v [pc]')
        cbar = fig.colorbar(points,ax=ax1)
        cbar.ax.set_ylabel('log D_M to 20th nearest neighbour')

        points = ax2.scatter(self.data['u'][self.sample],self.data['v'][self.sample],c=np.log10(self.rho_20_t),
                             alpha=0.5)
        ax2.scatter(self.target['u'],self.target['v'],c=np.log10(self.rho_20_target),edgecolor='k',marker='*')
        ax2.axis('equal')
        ax2.set_xlabel('u [pc]')
        ax2.set_ylabel('v [pc]')
        cbar = fig.colorbar(points,ax=ax2)
        cbar.ax.set_ylabel('log rho_20 [rescaled]')

        ax3.scatter((self.data['u'][self.sample])[self.is_high],(self.data['v'][self.sample])[self.is_high],
                    c='r',label='P_high>0.84',alpha=0.5)
        ax3.scatter((self.data['u'][self.sample])[self.is_ind],(self.data['v'][self.sample])[self.is_ind],
                    c='k',label='0.16<=P_high<=0.84',alpha=0.5)
        ax3.scatter((self.data['u'][self.sample])[self.is_low],(self.data['v'][self.sample])[self.is_low],
                    c='b',label='P_high<0.16',alpha=0.5)
        ax3.scatter(self.target['u'],self.target['v'],c=c_t,edgecolor='yellow',marker='*')
        ax3.axis('equal')
        ax3.set_xlabel('u [pc]')
        ax3.set_ylabel('v [pc]')
        ax3.legend()

        plt.savefig(self.folder+'sample_position_D_M_rho.pdf')
        plt.close()
        
        return

    def sample_Toomre(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=[15,4])


        if self.P_target < 0.16:
            c_t = 'b'
        if self.P_target > 0.84:
            c_t = 'r'
        if self.P_target >= 0.16 and self.P_target <= 0.84:
            c_t = 'k'

        x = np.linspace(-400,400,1001)
        ax1.plot(x,np.sqrt(100**2-x**2),'k')
        ax1.plot(x,np.sqrt(200**2-x**2),'k')
        ax1.plot(x,np.sqrt(300**2-x**2),'k')
        ax1.plot(x,np.sqrt(400**2-x**2),'k')

        points = ax1.scatter(self.data['U'][self.sample],np.sqrt(self.data['V'][self.sample]**2+
                                                                 self.data['W'][self.sample]**2),
                             c=np.log10(self.d_20),alpha=0.5)
        ax1.scatter(self.target['U'],np.sqrt(self.target['V']**2+self.target['W']**2),
                    c=np.log10(self.d_20_target),edgecolor='yellow',
                    marker='*')
        ax1.set_xlabel('$U$ [km/s]')
        ax1.set_ylabel('$\sqrt{V^2+W^2} [km/s]')
        ax1.set_xlim(np.min(self.data['U'][self.sample])-10,np.max(self.data['U'][self.sample])+10)
        ax1.set_ylim([0,np.max(np.abs(ax1.get_xlim()))+50])
        ax1.set_aspect('equal')

        cbar = plt.colorbar(points, ax=ax1)
        cbar.ax.set_ylabel('$\log_{10} D_\mathrm{M}$ to 20th nearest neighbour')

        ax2.plot(x,np.sqrt(100**2-x**2),'k')
        ax2.plot(x,np.sqrt(200**2-x**2),'k')
        ax2.plot(x,np.sqrt(300**2-x**2),'k')
        ax2.plot(x,np.sqrt(400**2-x**2),'k')

        points = ax2.scatter(self.data['U'][self.sample],np.sqrt(self.data['V'][self.sample]**2+
                                                                 self.data['W'][self.sample]**2),
                             c=np.log10(self.rho_20_t),alpha=0.5)
        ax2.scatter(self.target['U'],np.sqrt(self.target['V']**2+self.target['W']**2),
                    c=np.log10(self.rho_20_target),edgecolor='yellow',
                    marker='*')
        ax2.set_xlabel('$U$ [km/s]')
        ax2.set_ylabel('$\sqrt{V^2+W^2}$ [km/s]')
        ax2.set_xlim(np.min(self.data['U'][self.sample])-10,np.max(self.data['U'][self.sample])+10)
        ax2.set_ylim([0,np.max(np.abs(ax2.get_xlim()))+50])
        ax2.set_aspect('equal')

        cbar = plt.colorbar(points, ax=ax2)
        cbar.ax.set_ylabel(r'$\log_{10} \rho_{20}$ [rescaled]')

        ax3.plot(x,np.sqrt(100**2-x**2),'k')
        ax3.plot(x,np.sqrt(200**2-x**2),'k')
        ax3.plot(x,np.sqrt(300**2-x**2),'k')
        ax3.plot(x,np.sqrt(400**2-x**2),'k')

        ax3.scatter((self.data['U'][self.sample])[self.is_low],
                    np.sqrt((self.data['V'][self.sample])[self.is_low]**2+
                            (self.data['W'][self.sample])[self.is_low]**2),
                    c='b',alpha=0.2,label='$P_\mathrm{high}<0.16$')
        ax3.scatter((self.data['U'][self.sample])[self.is_ind],
                    np.sqrt((self.data['V'][self.sample])[self.is_ind]**2+
                            (self.data['W'][self.sample])[self.is_ind]**2),
                    c='k',alpha=0.2,label='$0.16\leq P_\mathrm{high}\leq0.84$')
        ax3.scatter((self.data['U'][self.sample])[self.is_high],
                    np.sqrt((self.data['V'][self.sample])[self.is_high]**2+
                            (self.data['W'][self.sample])[self.is_high]**2),
                    c='r',alpha=0.2,label='$P_\mathrm{high}>0.84$')
        ax3.scatter(self.target['U'],np.sqrt(self.target['V']**2+self.target['W']**2),
                    c=c_t,edgecolor='yellow',marker='*')
        ax3.set_xlabel('$U$ [km/s]')
        ax3.set_ylabel('$\sqrt{V^2+W^2}$ [km/s]')
        ax3.set_xlim(np.min(self.data['U'][self.sample])-10,np.max(self.data['U'][self.sample])+10)
        ax3.set_ylim([0,np.max(np.abs(ax3.get_xlim()))+50])
        ax3.set_aspect('equal')
        ax3.legend()


        plt.savefig(self.folder+'sample_Toomre.pdf')
        plt.close()
        
        return

    def detrend_v(self):

        try:
            degree = 4
            x = np.log10(self.sample_v)
            x_t = np.log10(self.target_v)

            log_rho = np.log10(self.rho_20_t)
            log_rho_t = np.log10(self.rho_20_target)

        # ignore rho_t > 50 in the fit, so not biased by clusters    
            self.fit = np.polynomial.Polynomial.fit(x[self.rho_20_t < 50],log_rho[self.rho_20_t < 50],degree)
            self.residuals = log_rho - self.fit.__call__(x)

            self.residuals_t = log_rho_t - self.fit.__call__(x_t)
        except:
            self.residuals = np.zeros(self.N_sample) * np.nan
            self.residuals_t = np.nan
            
        return

    def plot_trend(self):
        
        plt.figure(figsize=[5,4])
        if self.P_target < 0.16:
            c_t = 'b'
        if self.P_target > 0.84:
            c_t = 'r'
        if self.P_target >= 0.16 and self.P_target <= 0.84:
            c_t = 'k'

        x = np.log10(self.sample_v)
        x_t = np.log10(self.target_v)
        log_rho = np.log10(self.rho_20_t)
        log_rho_t = np.log10(self.rho_20_target)

        xfit,yfit = self.fit.linspace(101,[np.min(x),np.max(x)])

        plt.plot(xfit,yfit,label='quartic trend',c='k')
        plt.scatter(x[self.is_high],log_rho[self.is_high],c='r',alpha=0.2,label='$P_\mathrm{high}>0.84$')
        plt.scatter(x[self.is_ind],log_rho[self.is_ind],c='k',alpha=0.2,
                     label='$0.16\leq P_\mathrm{high}\leq0.84$')
        plt.scatter(x[self.is_low],log_rho[self.is_low],c='b',alpha=0.2,label='$P_\mathrm{high}<0.16$')
        plt.scatter(x_t,log_rho_t,c=c_t,edgecolor='yellow',marker='*',zorder=9,label=self.name_short)
        plt.xlabel('$\log_{10} |\mathbf{v}|$ [km/s]')
        plt.ylabel(r'$\log_{10} \rho_{20}$')

        ax1 = plt.gca()
        xmin = np.max((ax1.get_xlim()[0],0))
        xmax = ax1.get_xlim()[1]
        ymin = ax1.get_ylim()[0]
        ymax = ax1.get_ylim()[1]
        thin = Rectangle((xmin,ymin),np.log10(v_thin)-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
        thick = Rectangle((np.log10(v_thick_min),ymin),np.log10(v_thick_max)-np.log10(v_thick_min),
                          ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
        halo = Rectangle((np.log10(v_halo),ymin),xmax-np.log10(v_halo),ymax-ymin,facecolor='g',alpha=0.1,zorder=-10)
        ax1.add_patch(thin)
        ax1.add_patch(thick)
        ax1.add_patch(halo)
        ax1.text(xmin+(xmax-xmin)*0.1,ymin+(ymax-ymin)*0.9,'thin disc',color='g')
        ax1.text(np.log10(v_thick_min),ymin+(ymax-ymin)*0.9,'thick disc',color='g')
        ax1.text(np.log10(v_halo),ymin+(ymax-ymin)*0.9,'halo',color='g')
        ax1.legend(fontsize='small',loc='lower left')

        plt.savefig(self.folder+'trend.pdf')
        plt.close()
        
    def plot_residuals(self):

        try:
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=[10,4])

            if self.P_target < 0.16:
                c_t = 'b'
            if self.P_target > 0.84:
                c_t = 'r'
            if self.P_target >= 0.16 and self.P_target <= 0.84:
                c_t = 'k'

            x = np.log10(self.sample_v)
            x_t = np.log10(self.target_v)
            log_rho = np.log10(self.rho_20_t)
            log_rho_t = np.log10(self.rho_20_target)

            xfit,yfit = self.fit.linspace(101,[np.min(x),np.max(x)])

            ax1.plot(xfit,yfit,label='quartic trend',c='k')
            ax1.scatter(x[self.is_high],log_rho[self.is_high],c='r',alpha=0.2,label='$P_\mathrm{high}>0.84$')
            ax1.scatter(x[self.is_ind],log_rho[self.is_ind],c='k',alpha=0.2,
                        label='$0.16\leq P_\mathrm{high}\leq0.84$')
            ax1.scatter(x[self.is_low],log_rho[self.is_low],c='b',alpha=0.2,label='$P_\mathrm{high}<0.16$')
            ax1.scatter(x_t,log_rho_t,c=c_t,edgecolor='yellow',marker='*',zorder=9,label=self.name_short)
            ax1.set_xlabel('$\log_{10} |\mathbf{v}|$ [km/s]')
            ax1.set_ylabel(r'$\log_{10} \rho_{20}$')

            xmin = np.max((ax1.get_xlim()[0],0))
            xmax = ax1.get_xlim()[1]
            ymin = ax1.get_ylim()[0]
            ymax = ax1.get_ylim()[1]
            thin = Rectangle((xmin,ymin),np.log10(v_thin)-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
            thick = Rectangle((np.log10(v_thick_min),ymin),np.log10(v_thick_max)-np.log10(v_thick_min),
                              ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
            halo = Rectangle((np.log10(v_halo),ymin),xmax-np.log10(v_halo),ymax-ymin,facecolor='g',alpha=0.1,
                             zorder=-10)
            ax1.add_patch(thin)
            ax1.add_patch(thick)
            ax1.add_patch(halo)
            ax1.text(xmin+(xmax-xmin)*0.1,ymin+(ymax-ymin)*0.9,'thin disc',color='g')
            ax1.text(np.log10(v_thick_min),ymin+(ymax-ymin)*0.9,'thick disc',color='g')
            ax1.text(np.log10(v_halo),ymin+(ymax-ymin)*0.9,'halo',color='g')
            ax1.legend(fontsize='small')

            ax2.scatter(x[self.is_high],self.residuals[self.is_high],c='r',alpha=0.2)
            ax2.scatter(x[self.is_ind],self.residuals[self.is_ind],c='k',alpha=0.2)
            ax2.scatter(x[self.is_low],self.residuals[self.is_low],c='b',alpha=0.2)
            ax2.scatter(x_t,self.residuals_t,c=c_t,edgecolor='yellow',marker='*',label=self.name_short)
            ax2.set_xlabel('$\log_{10} |\mathbf{v}|$ [km/s]')
            ax2.set_ylabel('residuals')

            xmin = np.max((ax2.get_xlim()[0],0))
            xmax = ax2.get_xlim()[1]
            ymin = ax2.get_ylim()[0]
            ymax = ax2.get_ylim()[1]
            thin = Rectangle((xmin,ymin),np.log10(v_thin)-xmin,ymax-ymin,facecolor='g',alpha = 0.3,zorder=-10)
            thick = Rectangle((np.log10(v_thick_min),ymin),np.log10(v_thick_max)-np.log10(v_thick_min),
                              ymax-ymin,facecolor='g',alpha=0.2,zorder=-10)
            halo = Rectangle((np.log10(v_halo),ymin),xmax-np.log10(v_halo),ymax-ymin,facecolor='g',
                             alpha=0.1,zorder=-10)
            ax2.add_patch(thin)
            ax2.add_patch(thick)
            ax2.add_patch(halo)
            ax2.text(xmin+(xmax-xmin)*0.1,ymin+(ymax-ymin)*0.9,'thin disc',color='g')
            ax2.text(np.log10(v_thick_min),ymin+(ymax-ymin)*0.9,'thick disc',color='g')
            ax2.text(np.log10(v_halo),ymin+(ymax-ymin)*0.9,'halo',color='g')
            ax2.legend(fontsize='small',loc='lower left')

            plt.savefig(self.folder+'residuals.pdf')
            plt.close()

        except:
            pass
        
        return

    def get_ranks(self):

        v_abs = self.sample_v
        v_t_abs = self.target_v

        similar = np.logical_and(v_abs < v_t_abs*v_factor,v_abs > v_t_abs/v_factor)
        self.N_sim = sum(similar)

        self.rank = sum(self.rho_20_t > self.rho_20_target) + 1

        try:
            self.rank_sim = sum(self.rho_20_t[similar] > self.rho_20_target) + 1
        except: 
            print("len(similar):",len(similar))
            print("len(self.rho_20_t):", len(self.rho_20_t))
            self.rank_sim = None
        
        self.rank_detrended = sum(self.residuals > self.residuals_t) + 1
        
        
        temp = np.argsort(self.rho_20_t)[::-1]
        self.rank_all = np.empty_like(temp)
        self.rank_all[temp] = np.arange(len(self.rho_20_t))

        temp = np.argsort(self.residuals)[::-1]
        self.rank_detrended_all = np.empty_like(temp)

        try:
            self.rank_detrended_all[temp] = np.arange(len(self.rho_20_t))
        except:
            print("len(temp):",len(temp))
            print("len(self.rank_detrended_all):", len(self.rank_detrended_all))
            print("len(np.arange(len(self.rho_20_t))):",len(np.arange(len(self.rho_20_t))))
            
        
        return

    def plot_ranks(self):
        
        plt.figure(figsize=[5,4])
        plt.scatter(self.rank_all/self.N_sample,self.rank_detrended_all/self.N_sample)
        plt.xlabel('density fractional rank')
        plt.ylabel('residuals fractional rank')
        plt.title('Neighbours of '+self.name_short)
        plt.savefig(self.folder+'ranks.pdf')
        plt.close()
        
        return

    def write_densities(self):

        print("{:<20s} ranks {:>4d} of {:>4d} stars in decreasing density".format(self.name_short,
                                                                                  self.rank,self.N_sample+1))
        print("{:<20s} ranks {:>4d} of {:>4d} stars in decreasing residuals".format(self.name_short,
                                                                                    self.rank_detrended,
                                                                                    self.N_sample+1))
        if self.rank_sim != None:
            print("{:<20s} ranks {:>4d} of {:>4d} stars with |v| within {:5f}".format(self.name_short,
                                                                                      self.rank_sim,self.N_sim+1,v_factor))
        else:
            print("{:<20s} ranks {} of {:>4d} stars with |v| within {:5f}".format(self.name_short,
                                                                                      self.rank_sim,self.N_sim+1,v_factor))
            
            with open("errors.txt", "a") as file:
                    file.write(f"{self.name_short}\t{self.gaia_id}\n")
        
        with open(self.folder+'densities.txt','w') as f:
            print("Saving...")
            print("{:<20s} ranks {:>4d} of {:>4d} stars in decreasing density".format(self.name_short,
                                                                                      self.rank,self.N_sample+1),
                  file=f)
            print("{:<20s} ranks {:>4d} of {:>4d} stars in decreasing residuals".format(self.name_short,
                                                                                        self.rank_detrended,
                                                                                        self.N_sample+1),file=f)
            if self.rank_sim != None:
                print("{:<20s} ranks {:>4d} of {:>4d} stars with |v| within {:5f}".format(self.name_short,
                                                                                          self.rank_sim,self.N_sim+1,
                                                                                          v_factor),file=f)
            else:
                print("{:<20s} ranks {} of {:>4d} stars with |v| within {:5f}".format(self.name_short,
                                                                                          self.rank_sim,self.N_sim+1,
                                                                                          v_factor),file=f)
                
                
            print("\n",file=f)
            print(("{:^30s}|{:^9s}|{:^9s}"+"|{:^10s}"*6).format("Gaia id","rho","D","u","v","w","U","V","W"),file=f)
            print(("{:^30s}|{:^9s}|{:^9s}"+"|{:^10s}"*6).format("","","","[pc]","[pc]","[pc]",
                                                                 "[km/s]","[km/s]","[km/s]"),file=f)
            print('-'*120,file=f)

            print(("{:<30s}|{:^9.3e}|{:^9.3e}"+"|{:>10.3e}"*6).format(self.data["designation"][self.i_target],
                                                                          self.rho_20_target,self.d_20_target,
                                                                          self.data["u"][self.i_target],
                                                                          self.data["v"][self.i_target],
                                                                          self.data["w"][self.i_target],
                                                                          self.data["U"][self.i_target],
                                                                          self.data["V"][self.i_target],
                                                                          self.data["W"][self.i_target]),
                      file=f)
        
                
            for i in range(self.N_sample):
                try:
                    print(("{:<30s}|{:^9.3e}|{:^9.3e}"+"|{:>10.3e}"*6).format(self.data["designation"][self.sample[i]],
                                                                              self.rho_20_t[i],self.d_20[i],
                                                                              self.data["u"][self.sample[i]],
                                                                              self.data["v"][self.sample[i]],
                                                                              self.data["w"][self.sample[i]],
                                                                              self.data["U"][self.sample[i]],
                                                                              self.data["V"][self.sample[i]],
                                                                              self.data["W"][self.sample[i]]),
                                                                            file=f)
                except:
                    pass
                    #print("Error")               
                
        return
    
    def free_mem(self):
        self.dist_target = None
        self.pos_6D = None
        self.closest_target = None

# +
include_Sol = True

#planets_table = ascii.read('PS_2021.02.24_07.17.03.csv')
#planets_table = ascii.read('PS_2021.03.08_07.22.24.csv')
#planets_table = ascii.read('PS_2021.03.11_04.56.35.csv')
#planets_table = ascii.read('PS_2025.03.15_04.37.59.csv')
#planets_table = ascii.read('PS_2025.04.18_08.23.37.csv')
planets_table = ascii.read('PSCompPars_2025.05.16_03.09.22.csv')


#fix "Qatar-n" -> "Qatar n"
planets_table['hostname'] = [t.replace("Qatar-","Qatar ") for t in planets_table['hostname']]
#Praesepe: "Prnnnn" isn't a catalogue in Simbad, and I can't find it's published anywhere
#exoplanet.eu ids Pr0201 as BD+20 2184 but no id for Pr0211...
planets_table['hostname'] = [t.replace("Pr0201","BD+20 2184") for t in planets_table['hostname']]
#HIP 65A is just HIP 65 in Simbad

targets = [p['hostname'] for p in planets_table]
planets = [p['pl_name'] for p in planets_table]


#remove duplicate hosts
#might change order at this point
targets = list(set(targets))
planets = list(set(planets))

print(f"nr of exoplanet systems: {len(targets)}")
print(f"nr of exoplanets: {len(planets)}")

#restored_from_file = False
restore_from_file=False

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
            dr2id.append(xmatch['dr2id'][index])
            dr3id.append(xmatch['dr3id'][index])
        
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


for i in range(10):
    print(targets[i],dr2id[i], dr3id[i])




# +
n_targets = len(targets)

stars = []

# loop over targets. Functions making plots are commented out

for i in range(n_targets):
    print(str(i)+' of '+str(n_targets))
    if source_cat == 'DR3_1kpc':
        target = Target(targets[i],dr3id[i]) 
    if source_cat == 'DR2_all':
        target = Target(targets[i],dr2id[i])
        
    target.get_neighbours()
    if target.data is not None and target.N_stars >= N_stars_min:
        #target.distance_histograms()
        #target.distance_histograms_fine()
        #target.magnitude_histograms()
        #target.parallax_error_histograms()
        #target.distance_Gmag()
        #target.distance_M_G()
        #target.CMD()
        #target.CMD_hist()
        #target.RV_histograms()
        #target.X_Y()
        #target.Toomre()
        target.get_pos_6D()
        target.get_dist_target()
        #target.D_M_histograms()
        #target.Delta_v_histograms()
        #target.D_phys_D_M()
        #target.Delta_v_D_M()
        #target.D_phys_Delta_v()
        target.get_close_target()
        target.get_lt_40pc()
        target.set_seed()
        target.get_random_sample()
        target.get_sample_distances()
        #target.sample_D_M_rho_histograms()
        target.gauss()
#        target.plot_gaussian_mixture()
        target.gauss_v()
        #target.plot_gaussian_mixture_v()
        #target.sample_D_phys_D_M_rho()
        #target.sample_D_phys_D_M_rho_1panel()
        #target.sample_Delta_v_D_M_rho()
        #target.sample_abs_v_D_M_rho()
        #target.sample_abs_v_D_M_rho_1panel()
        #target.UVW_rho()
        #target.sample_position_D_M_rho()
        #target.sample_Toomre()
        target.detrend_v()
        #target.plot_residuals()
        target.get_ranks()
        target.write_densities()
        target.save_den