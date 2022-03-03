# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:20:52 2022

@author: filip
"""
import os
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
import topsis_FG as top
import copras as cop

def par_mean(plik2, name, date):
    x = plik2.stack().T.loc[:, (slice(None), slice(None), name)]
    m = pd.DataFrame(x.T.mean())
    m.columns = [date]
    return m, x

def how_empty(file, IGSNetwork):
    for z in file.columns:
        if file[z].sum() == 0:
            try:
                IGSNetwork.loc[z,'stats'] = IGSNetwork.loc[z,'stats'] + 1
            except(KeyError):
                continue
    return IGSNetwork

def dir_to_pick_file(directory, date, endpoint):
    
    year = date.strftime('%Y')
    doy = date.strftime('%j')
    list_of_file = os.listdir(directory)
    s = []
    for k in list_of_file:
        if k.endswith(f"{year}_{doy}{endpoint}"):
            s.append(os.path.join(directory, k))
    return s[0]

def map_generate(longitude, latitude, labels, num_points):
    #generacja mapy
    import geopandas
    path = geopandas.datasets.get_path('naturalearth_lowres')
    df = geopandas.read_file(path)
    df.plot(figsize=(15,10), color="white", edgecolor = "grey")
    plt.grid()
    plt.scatter(longitude, latitude, c=labels, s=30)
    plt.title(f"Dla podziału na {num_points} segmentów")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Longitude [°]", fontsize=15)
    plt.ylabel("Latitude [°]", fontsize=15)
    plt.legend(labels)
    plt.show()
    
# grupowanie pozostałych stacji na segmenty

def dividing_stations(IGSNetwork, alg, num_points):
    if alg == "KMeans":
        algorytm = KMeans(n_clusters=num_points, random_state=0)
        
    elif alg == "AgglomerativeClustering":
        algorytm = AgglomerativeClustering(n_clusters=num_points)
    
    labels = algorytm.fit(np.array(IGSNetwork.iloc[:,:3])).labels_
    labels = pd.DataFrame(labels, index=IGSNetwork.index,columns=['segment'])
    IGSNetwork = pd.concat([IGSNetwork, labels], axis=1)
    
    return IGSNetwork

#Multiple-criteria decision analysis
def MDCA(IGSNetwork, alg, weights, num_points, num_hz):
    mat = pd.DataFrame()
    
    crit = [1, 1, -1, -1]
    criterias = np.array([-1])
    
    crit2 = np.array(['max', 'max', 'min', 'min'])
    criterias2 = np.array(['min'])
    
    for h in range(num_hz):
        criterias = np.hstack((criterias,crit))
        criterias2 = np.hstack((criterias2,crit2))
        
    for segment in range(0,num_points):
        seg = IGSNetwork[IGSNetwork.loc[:,"segment"]==segment].drop("stats", axis=1)
        evaluation_matrix = np.array(seg.iloc[:,6:-1])

        if alg == "TOPSIS":
            rank = top.topsis(evaluation_matrix, weights, criterias)
            df = pd.DataFrame(rank, index=seg.index, columns=["TOPSIS"])
            
        elif alg == "COPRAS":
            rank = cop.copras_method(evaluation_matrix, weights, criterias2)
            df = pd.DataFrame(rank, index=seg.index, columns=["COPRAS"])
        
        mat = pd.concat([mat, df])
    IGSNetwork = pd.concat([IGSNetwork, mat], axis=1)

    return IGSNetwork

def process_file(IGSNetwork, folder_name, date, lastDate):
    print("Przetwarzam pliki...")
    snr_mean = pd.DataFrame()
    gaps_mean = pd.DataFrame()
    obs_mean = pd.DataFrame()
    mp_mean = pd.DataFrame()
    hz = []
    sys = []
    new=[]
    i=0
    while date <= lastDate:
        dirname = os.path.dirname(os.path.abspath("__file__"))
        directory = os.path.join(dirname,folder_name)
        
        file_to_av = dir_to_pick_file(directory, date, ".csv")
        file_to_q = dir_to_pick_file(directory, date, "_q.csv")
        
        plik = pd.read_csv(file_to_av, index_col=0)
        plik2 = pd.read_csv(file_to_q, index_col=[0,1], header=[0,1])
        plik3 = pd.read_csv(file_to_q).iloc[:,1].to_list()
        plik4 = pd.read_csv(file_to_q).iloc[:,0].to_list()
        for g in plik2.index:
            new.append(g)

        for y in plik4:
            if type(y) is str:
                sys.append(y)
        
        IGSNetwork = how_empty(plik, IGSNetwork)
        
        ms, snr = par_mean(plik2, 'snr', date)
        snr_mean = pd.concat([snr_mean, ms], axis=1).fillna(0)
        
        mg, gaps = par_mean(plik2, 'gaps', date)
        gaps_mean = pd.concat([gaps_mean, mg], axis=1).fillna(0)
        
        mo, obs = par_mean(plik2, 'obs', date)
        obs_mean = pd.concat([obs_mean, mo], axis=1).fillna(0)
        
        mp, multipath = par_mean(plik2, 'multipath', date)
        mp_mean = pd.concat([mp_mean, mp], axis=1).fillna(0)
        
        i+=1
        date += datetime.timedelta(days=1)

    new = list(sorted(set(new)))
    multi = pd.MultiIndex.from_tuples(new)
    new2 = pd.DataFrame(index=multi)
    sys = list(set(sys))
    df4_2 = pd.concat([snr, obs, gaps, multipath], axis=1)
    return df4_2, i, sys, new2

def by_par(statsy, system1, system2, hz1, hz2):
    par1 = statsy.loc[:, (system1, hz1, slice(None))]
    par2 = statsy.loc[:, (system2, hz2, slice(None))]
    
    out = pd.concat([par1, par2], axis=1).dropna()
    return out

def only_ones(IGSNetwork, method):
    IGSNetwork = IGSNetwork[IGSNetwork.loc[:,method]==1]
    return IGSNetwork