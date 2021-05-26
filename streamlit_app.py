import streamlit as st
import datetime
import json
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
# sns.set_style("darkgrid")

# configuration

with open("config.yaml", "r") as yamlfile:
    cfg = yaml.safe_load(yamlfile)

jaydata_filenme = 'jaydata.parquet'
vaccination_filename = 'vaccination.parquet'
# begin_date  = cfg["data"].get('begin_date')
hospitalized_breakdown_start = '2021-04-24'

@st.cache
def load_data():
    jay_data = pd.read_parquet('jaydata.parquet')
    vac_data = pd.read_parquet('vaccination.parquet')
    return jay_data, vac_data

begin_date = st.sidebar.slider("Chart start date", 
                        value=datetime.date(2021, 3, 1),
                        format="DD-MM-YYYY", 
                        min_value=datetime.date(2020, 1, 1),
                        max_value=datetime.date.today()).strftime("%Y-%m-%d")

vac_date = st.sidebar.slider("Vaccination status date", 
                        value=datetime.date(2021, 5, 2),
                        format="DD-MM-YYYY", 
                        min_value=datetime.date(2021, 3, 6),
                        max_value=datetime.date.today()).strftime("%Y-%m-%d")

st.title('Thailand COVID Dashboard\n Data from https://github.com/djay/covidthailand')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df, vac_df = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data... Done!")

# Data Processing

datemask = (df['Date'] >= begin_date)
df = df.loc[datemask]
# df.set_index("Date", drop=False, inplace=True)

# df['Hospitalized'] = np.where(df['Date'] >= hospitalized_breakdown_start, 0, df['Hospitalized']) #only for visualization

df.loc[df['Date'] >= hospitalized_breakdown_start, 'Hospitalized'] = 0

# Forward fill some cumulative numbers that have gaps in data

col_to_ffill = ['Vac Group Medical Staff 1 Cum', 
                'Vac Group Other Frontline Staff 1 Cum',
                'Vac Group Over 60 1 Cum',
                'Vac Group Risk: Disease 1 Cum',
                'Vac Group Risk: Location 1 Cum',
                'Vac Given 1 Cum',
                'Vac Given 2 Cum']
df.loc[:,col_to_ffill] = df.loc[:,col_to_ffill].ffill()

# Calculate daily vaccination administered

df['Shot 1 Administered Daily'] = df['Vac Given 1 Cum'].diff()
df['Shot 2 Administered Daily'] = df['Vac Given 2 Cum'].diff()

province_df = pd.read_html("https://en.wikipedia.org/wiki/Provinces_of_Thailand#The_provinces_and_Administrative_Areas")[2]
province_df.columns = ['Seal', 'Province', 'ProvinceTh', 'Population', 'Area', 'PopDensity', 'Namesake', 'HS', 'ISO', 'FIPS']
province_df.at[0, 'Province'] = 'Bangkok'
vac_df = vac_df.merge(province_df[['Province', 'Population']], on='Province', how='left')
vac_df['Vac Given 1 %'] = 100 * vac_df['Vac Given 1 Cum'] / vac_df['Population']
df = df.rename(columns={'Date':'index'}).set_index('index')

# original data from https://github.com/apisit/thailand.json
# read geojson definition of provinces and geometries
with open('thailand.json') as json_file:
    thailand_json = json.load(json_file)

# Fixing to make it work with Plotly Chroropleth expectation
removed_first_element = thailand_json['features'].pop(0)
# Fixing Province names to match those used on Wikipedia
thailand_json['features'][3]['properties']['name'] = 'Phang Nga'
thailand_json['features'][31]['properties']['name'] = 'Lopburi'
thailand_json['features'][33]['properties']['name'] = 'Prachinburi'
thailand_json['features'][39]['properties']['name'] = 'Bangkok'
thailand_json['features'][48]['properties']['name'] = 'Sisaket'
thailand_json['features'][52]['properties']['name'] = 'Chonburi'
thailand_json['features'][58]['properties']['name'] = 'Buriram'
thailand_json['features'][70]['properties']['name'] = 'Nong Bua Lamphu'

st.subheader('Daily New Cases')
st.bar_chart(df['Cases'])

st.subheader('Daily Tests Administered')
st.bar_chart(df['Tested'])

st.subheader('Number of people hospitalized')
st.bar_chart(df[["Hospitalized", "Hospitalized Hospital", "Hospitalized Field"]].fillna(0))

st.subheader('Number of Serious Cases and Deaths')
st.line_chart(df[["Hospitalized Respirator", "Hospitalized Severe", "Deaths"]])

st.subheader('1st shot vaccine administered (daily)')
st.bar_chart(df['Shot 1 Administered Daily'])


st.subheader('Bangkok 1st shot vaccine administered (cumulative)')

st.bar_chart(df[['Vac Group Risk: Location 1 Cum', 
                'Vac Group Risk: Disease 1 Cum',
                'Vac Group Over 60 1 Cum',
                'Vac Group Other Frontline Staff 1 Cum',
                'Vac Group Medical Staff 1 Cum']])

st.subheader('% of population having got 1st shot vaccine')

vac_oneday_df = vac_df[vac_df['Date'] == vac_date]

fig = px.choropleth(vac_oneday_df, geojson=thailand_json, color="Vac Given 1 %",
                    locations="Province", featureidkey="properties.name",
                    projection="mercator",
                    color_continuous_scale = 'Greens'
                   )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig)