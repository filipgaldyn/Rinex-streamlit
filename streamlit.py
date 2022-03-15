import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import base64
import datetime
import os
from functions import *
import branca.colormap as cm

def filedownload(df, name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download CSV File</a>'
    return href

st.set_page_config(
    page_title="RINEX-AV",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h2 style='text-align: center; color: black;'>GNSS permanent MGEX station selection system based on qualitative analysis of RINEX files</h2>", unsafe_allow_html=True)

st.subheader("The application selects stations with the best quality associated with RINEX files. \n Priority level is determined using the LIST OF STATION PROPOSED TO IGS REPRO3, which can be found [here](http://acc.igs.org/repro3/repro3_station_priority_list_060819.pdf).\n")

col1 = st.sidebar
col2, col3 = st.columns((2,1))

dirname = os.path.dirname(os.path.abspath("__file__"))
directory = os.path.join(dirname,"rv3_stat")

dirname = os.path.dirname(os.path.abspath("__file__"))
file = os.path.join(dirname, "MGEX_wsp.csv")
IGSNetwork = pd.read_csv(file).set_index("#StationName")
IGSNetwork['stats'] = 0
st.sidebar.markdown("<h3 style='text-align: center; color: black;'>Data Availability:<br> <u>2021/01/01 - 2021/02/16</u></h3>", unsafe_allow_html=True)
date = st.sidebar.date_input('Start date', datetime.date(2021,1,1))
lastDate = st.sidebar.date_input('End date', datetime.date(2021,1,3))

ileprocent = st.sidebar.slider('Avarage Percent of RINEX availability', 50, 100, value=90, step=5)
clustering_method = st.sidebar.selectbox('Clustering Method', ["KMeans", "AgglomerativeClustering"])

method = st.sidebar.selectbox('Method of decision making', ["TOPSIS", "COPRAS"])
num_points = st.sidebar.slider('Number of Stations', 5, IGSNetwork.shape[0], value=100, step=5)

try:
    df, i, sys = process_file(IGSNetwork, "rv3_stat", date, lastDate)
except:
    st.error("ENTER THE RELEVANT DATE RANGE")
    st.stop()

with col1:
    sys_bar1 = st.sidebar.multiselect('Systems', sys, default=sys[0])
    freq1 = df.loc[:, (sys_bar1, slice(None), slice(None))].columns.to_list()
    
hz = []
for par in freq1:
    hz.append(tuple(list(par)[:2]))
    
freq1 = sorted(list(set(hz)))

if freq1:
    freq_done1 =  st.sidebar.multiselect('Frequencies', freq1, default=freq1[0])
    
    par = ['SNR', 'N_obs', 'gaps', 'multipath']
    
    k = int(st.sidebar.number_input('Weight Priority Level', 1, 100))
    ich_all=[k]
    
    for k in freq_done1:
        for h in par:
            ich = int(st.sidebar.number_input(f'Weight {h} ({k[0]},{k[1]})', 1, 100))
            ich_all.append(ich)
    
if st.sidebar.button('Submit'):
    outer=pd.DataFrame()
    for x in freq_done1:
        out = df.loc[:, (x[0], x[1], slice(None))]
        outer = pd.concat([outer, out], axis=1)
    outer = outer.dropna()
    full_mean = mean_all(outer, freq_done1)
    IGSNetwork = pd.concat([IGSNetwork, full_mean], axis=1, join='inner')
    IGSNetwork.iloc[:,7:8] = ((i-IGSNetwork.iloc[:,7:8])/i) *100
    IGSNetwork = IGSNetwork[IGSNetwork.loc[:,"stats"]>=ileprocent]
    
    if IGSNetwork.shape[0] >= num_points:
        IGSNetwork = dividing_stations(IGSNetwork, clustering_method, num_points)
        IGSNetwork = MDCA(IGSNetwork, method, ich_all, num_points, len(freq_done1))
        wybor = only_ones(IGSNetwork, method)
        wybor = wybor.iloc[:,3:]
        IGSNetwork = IGSNetwork.iloc[:,3:]
        
        col2.markdown(f"<h4 color: black;'>All stations that have RINEX file availability above {ileprocent}% and have observations on selected frequencies.</h4>", unsafe_allow_html=True)
        col2.dataframe(IGSNetwork.iloc[:,3:].style.format("{:.1f}"), height=380)
        col2.write('Data Dimension: ' + str(IGSNetwork.shape[0]) + ' rows and ' + str(IGSNetwork.shape[1]) + ' columns.')
        col2.markdown(filedownload(IGSNetwork.reset_index(), 'all_stations'), unsafe_allow_html=True)
        col2.markdown(f"<h4 color: black;'>Selected {num_points} stations with the best RINEX file quality.</h4>", unsafe_allow_html=True)
        col2.dataframe(wybor.style.format("{:.1f}"), height=380)
        col2.write('Data Dimension: ' + str(wybor.shape[0]) + ' rows and ' + str(wybor.shape[1]) + ' columns.')
        col2.markdown(filedownload(wybor.reset_index(), "selected"), unsafe_allow_html=True)
        
        df = IGSNetwork.loc[:,["Longitude", "Latitude"]]
        df = df.rename(columns={"Longitude": "lon", "Latitude": "lat"})
        
        df2 = wybor.loc[:,["Longitude", "Latitude"]]
        df2 = df2.rename(columns={"Longitude": "lon", "Latitude": "lat"})
  
        m = folium.Map()
        m2 = folium.Map()
        
        df3 =  IGSNetwork.loc[:,["Longitude", "Latitude", "segment"]]
        df3 = df3.rename(columns={"Longitude": "lon", "Latitude": "lat"})
        linear = cm.LinearColormap(["r", "y", "g", "c", "b", "m","white", "black", "red"], vmin=df3.loc[:,'segment'].min(), vmax=df3.loc[:,'segment'].max())

        for _, row in df3.iterrows():
            pos = [row.lat, row.lon]

            folium.CircleMarker(pos,
                          popup=_,
                          radius=3,
                          color = linear(row.segment)
                          ).add_to(m)
            
        for x in df2.index:
            pos = list(np.flip(np.array(df2.loc[x])))
            folium.CircleMarker(pos,
                          popup=x,
                          radius=2,
                          fill_color="red", 
                          color = "red",
                          ).add_to(m2)

        with col3:
            folium_static(m)
            folium_static(m2)
    elif IGSNetwork.shape[0] == 0:
        st.markdown(f"<h3 style='text-align: center; color: red;'>None of the stations meet the conditions entered.</h3>", unsafe_allow_html=True)
        
    else:
        st.dataframe(IGSNetwork)
        st.markdown(f"<h3 style='text-align: center; color: red;'>Only {IGSNetwork.shape[0]} stations received a signal on selected frequencies. To display the result, decrease the number of stations.</h3>", unsafe_allow_html=True)
        

st.sidebar.title("About")
st.sidebar.info(
    """
    This web [app](##) is maintained by [Filip Gałdyn](##). You can follow me on social media:
     [GitHub](https://github.com/filipgaldyn) | [Twitter](https://twitter.com/FilipGaldyn) | [LinkedIn](https://www.linkedin.com/in/filip-ga%C5%82dyn/).
    This web app URL: <https://rinexav.herokuapp.com/>
""")
