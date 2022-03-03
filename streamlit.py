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
st.markdown("<h1 style='text-align: center; color: black;'>GNSS permanent MGEX station selection system based on qualitative analysis of RINEX files</h1>", unsafe_allow_html=True)

st.subheader("The application selects stations with the best quality associated with RINEX files. These are priority level, SNR, number of observations, multipath and gaps. Each parameter has the same weight. Priority level is determined using the LIST OF STATION PROPOSED TO IGS REPRO3, which can be found [here](http://acc.igs.org/repro3/repro3_station_priority_list_060819.pdf).\n")

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
lastDate = st.sidebar.date_input('End date', datetime.date(2021,1,2))
num_points = st.sidebar.slider('Number of Stations', 10, IGSNetwork.shape[0], value=100, step=5)
ileprocent = st.sidebar.slider('Avarage Percent of RINEX availability', 50, 100, value=90, step=5)
clustering_method = st.sidebar.selectbox('Clustering Method', ["KMeans", "AgglomerativeClustering"])

method = st.sidebar.selectbox('Method of decision making', ["TOPSIS", "COPRAS"])

try:
    statsy, i, sys, new2 = process_file(IGSNetwork, "rv3_stat", date, lastDate)
except:
    st.error("ENTER THE RELEVANT DATE RANGE")
    st.stop()
    
sys_bar1 = st.sidebar.multiselect('Systems', sys, default=sys[0])
freq1 = new2.loc[(sys_bar1, slice(None))].index.to_list()
freq_done1 =  st.sidebar.multiselect('Frequencies', freq1, default=freq1[0])
par = ['SNR', 'N_obs', 'gaps', 'multipath']

k = int(st.sidebar.number_input('Weight Priority Level', 1, 100))
ich_all=[k]
for k in freq_done1:
    for h in par:
        ich = int(st.sidebar.number_input(f'Weight {h} ({k[0]},{k[1]})', 1, 100))
        ich_all.append(ich)

freq=[]
for x in freq_done1:
    freq.append(x[1])

# sys_bar2 = st.sidebar.selectbox('System 2', sys)
# freq2 = new2.loc[(sys_bar2, slice(None))].index.to_list()
# freq_done2 =  st.sidebar.selectbox('Frequency 2', freq2)

war1 = False
war2 = False
if st.sidebar.button('Submit'):
    if (war1 and war2) == False:   
        out = statsy.loc[:, (sys_bar1, freq, slice(None))]
        out = out.dropna()
        IGSNetwork = pd.concat([IGSNetwork, out], axis=1, join='inner')
        
        IGSNetwork.iloc[:,7:8] = ((i-IGSNetwork.iloc[:,7:8])/i) *100
        IGSNetwork = IGSNetwork[IGSNetwork.loc[:,"stats"]>=ileprocent]
        
        if IGSNetwork.shape[0] >= num_points:
            IGSNetwork = dividing_stations(IGSNetwork, clustering_method, num_points)
            IGSNetwork = MDCA(IGSNetwork, method, ich_all, num_points, len(freq))
            wybor = only_ones(IGSNetwork, method)
            wybor = wybor.iloc[:,3:]
            IGSNetwork = IGSNetwork.iloc[:,3:]
            
            col2.markdown(f"<h4 color: black;'>All stations that have RINEX file availability above {ileprocent}% and have observations on selected frequencies.</h4>", unsafe_allow_html=True)
            col2.dataframe(IGSNetwork.iloc[:,3:], height=320)
            col2.write('Data Dimension: ' + str(IGSNetwork.shape[0]) + ' rows and ' + str(IGSNetwork.shape[1]) + ' columns.')
            col2.markdown(filedownload(IGSNetwork.reset_index(), 'all_stations'), unsafe_allow_html=True)
            col2.markdown(f"<h4 color: black;'>Selected {num_points} stations with the best RINEX file quality.</h4>", unsafe_allow_html=True)
            col2.dataframe(wybor, height=320)
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
            linear = cm.LinearColormap(["r", "y", "g", "c", "b", "m", "black"], vmin=df3.loc[:,'segment'].min(), vmax=df3.loc[:,'segment'].max())

            for _, row in df3.iterrows():
                pos = [row.lat, row.lon]
                print(row)
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
        else:
            st.markdown(f"<h3 style='text-align: center; color: red;'>Only {IGSNetwork.shape[0]} stations received a signal on selected frequencies. To display the result, decrease the number of stations.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: red;'>Select another frequency pair.</h3>", unsafe_allow_html=True)
st.sidebar.title("About")
st.sidebar.info(
    """
    This web [app](##) is maintained by [Filip Gałdyn](##). You can follow me on social media:
     [GitHub](https://github.com/filipgaldyn) | [Twitter](https://twitter.com/FilipGaldyn) | [LinkedIn](https://www.linkedin.com/in/filip-ga%C5%82dyn/).
    This web app URL: <https://rinexav.herokuapp.com/>
""")
