import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import base64
import datetime
import os
from functions import *

def filedownload(df, name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download CSV File</a>'
    return href


st.set_page_config(layout="wide")
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
st.sidebar.header('User Input Features')
date = st.sidebar.date_input('Start date', datetime.date(2021,1,1))
lastDate = st.sidebar.date_input('End date', datetime.date(2021,1,2))
num_points = st.sidebar.slider('Number of Stations', 10, IGSNetwork.shape[0], value=100, step=10)
ileprocent = st.sidebar.slider('Avarage Percent of RINEX availability', 50, 100, value=90, step=5)
clustering_method = st.sidebar.selectbox('Clustering Method', ["KMeans", "AgglomerativeClustering"])

method = st.sidebar.selectbox('Method of decision making', ["TOPSIS", "COPRAS"])

if date <= lastDate:
    statsy, i, sys, new2 = process_file(IGSNetwork, "rv3_stat", date, lastDate)
else:
    st.markdown("Wprowadź odpowiedni przedział dat")
  
sys_bar1 = st.sidebar.selectbox('System 1', sys)
freq1 = new2.loc[(sys_bar1, slice(None))].index.to_list()
freq_done1 =  st.sidebar.selectbox('Frequency 1', freq1)


sys_bar2 = st.sidebar.selectbox('System 2', sys)
freq2 = new2.loc[(sys_bar2, slice(None))].index.to_list()
freq_done2 =  st.sidebar.selectbox('Frequency 2', freq2)

war1 = sys_bar1 == sys_bar2
war2 = freq_done1 == freq_done2 
if st.sidebar.button('Submit'):
    st.empty()
    if (war1 and war2) == False:   
        out = by_par(statsy, sys_bar1, sys_bar2, freq_done1, freq_done2)
        IGSNetwork = pd.concat([IGSNetwork, out], axis=1, join='inner')
        IGSNetwork.iloc[:,7:8] = ((i-IGSNetwork.iloc[:,7:8])/i) *100
        IGSNetwork = IGSNetwork[IGSNetwork.loc[:,"stats"]>=ileprocent]
        st.write(IGSNetwork.shape[0])
        
        if IGSNetwork.shape[0] >= num_points:
            IGSNetwork = dividing_stations(IGSNetwork, clustering_method, num_points)
            IGSNetwork = MDCA(IGSNetwork, method, [1, 1, 1, 1, 1, 1, 1, 1, 1], num_points)
            wybor = only_ones(IGSNetwork, method)
            wybor = wybor.iloc[:,3:]
            
            col2.markdown(f"<h4 color: black;'>All stations that have RINEX file availability above {ileprocent}% and have observations on selected frequencies.</h4>", unsafe_allow_html=True)
            col2.dataframe(IGSNetwork.iloc[:,3:], height=320)
            col2.write('Data Dimension: ' + str(IGSNetwork.shape[0]) + ' rows and ' + str(IGSNetwork.shape[1]) + ' columns.')
            col2.markdown(filedownload(IGSNetwork.reset_index().iloc[:,3:], 'all_stations'), unsafe_allow_html=True)
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

            for x in df.index:
                pos = list(np.flip(np.array(df.loc[x])))

                folium.CircleMarker(pos,
                              popup=x,
                              radius=3,
                              fill_color="red", 
                              color = 'red',
                              ).add_to(m)
            for x in df2.index:
                pos = list(np.flip(np.array(df2.loc[x])))

                folium.CircleMarker(pos,
                              popup=x,
                              radius=3,
                              fill_color="green", 
                              color = 'green'
                              ).add_to(m2)

            with col3:
                folium_static(m)
                folium_static(m2)
        else:
            st.write("Zbyt mało stacji spełnia wskazane warunki")
    else:
        "Wybierz inną parę"
st.sidebar.title("About")
st.sidebar.info(
    """
    This web [app](##) is maintained by [Filip Gałdyn](##). You can follow me on social media:
     [GitHub](https://github.com/filipgaldyn) | [Twitter](https://twitter.com/FilipGaldyn) | [LinkedIn](https://www.linkedin.com/in/filip-ga%C5%82dyn/).
    This web app URL: <##>
""")
