import requests
import io
import os.path as path
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("darkgrid")
import boto3

# configuration

with open("config.yaml", "r") as yamlfile:
    cfg = yaml.safe_load(yamlfile)


begin_date  = cfg["data"].get('begin_date')
plot_save = cfg["plot"].get('plot_save')
plot_upload = cfg["plot"].get('plot_upload')
local_max_age = cfg["freshness"].get('local_max_age') # To  always download, set this to zero in config.yaml
file_name = 'jaydata.parquet'
hospitalized_breakdown_start = '2021-04-24'

last_updated = 0

session = boto3.Session(profile_name='default')
s3 = session.resource('s3')

def download_data_and_save(): 
  """Download latest data from djay's repo and save as CSV and parquet
  """
  url = 'https://github.com/djay/covidthailand/wiki/combined.csv'
  s=requests.get(url).content
  global df
  global last_updated
  df=pd.read_csv(io.StringIO(s.decode('utf-8')), parse_dates= ['Date'])
  df.to_parquet(file_name, compression='UNCOMPRESSED')
  df.to_csv('jaydata.csv')
  last_updated = df['Date'][df.index[-1]].strftime("%d %B %Y")

  url = 'https://raw.githubusercontent.com/wiki/djay/covidthailand/vaccinations.csv'
  s=requests.get(url).content
  global vac_df
  vac_df=pd.read_csv(io.StringIO(s.decode('utf-8')), parse_dates= ['Date'])
  vac_df.to_parquet('vaccination.parquet', compression='UNCOMPRESSED')

  print("Data downloaded and saved successfully. Data up to " + last_updated)

try:
  file_time = path.getmtime(file_name)
  file_age = ((time.time() - file_time) / 3600) # in hours
  print("file age: " + f'{file_age:9.1f}')
  if file_age > local_max_age:
    print(f"File older than {local_max_age} hours. Try to download newer data.")
    download_data_and_save()
  else:
    print(f"File less than {local_max_age} hours old. Load from local copy")
    df = pd.read_parquet('jaydata.parquet')
except FileNotFoundError:
  download_data_and_save()

# Data Processing

datemask = (df['Date'] >= begin_date)
df = df.loc[datemask]
df.set_index("Date", drop=False, inplace=True)

last_updated = df['Date'][df.index[-1]].strftime("%d %B %Y")
print(f"Latest data: {last_updated}")

# Calculate Positive Rate

df["Tests XLS (MA)"] = df["Tests XLS"].rolling(7, 1, center=True).mean()
df["Pos XLS (MA)"] = df["Pos XLS"].rolling(7, 3, center=True).mean()
df["Positive Rate (7-day MA)"] = df["Pos XLS (MA)"] / df["Tests XLS (MA)"] * 100

# Capture latest numbers for chart titles

latest_new_confirmed = int(df['Cases'].dropna().iloc[-1])
latest_test_administered = int(df['Tested'].dropna().iloc[-1])
latest_hospitalized = int(df['Hospitalized'].dropna().iloc[-1])
latest_deaths = int(df['Deaths'].dropna().iloc[-1])

# Dataframe for serious cases plotting and hospitalized breakdown

serious_df = df[["Date", "Hospitalized Respirator", "Hospitalized Severe", "Deaths"]].melt('Date', var_name='cols', value_name='vals')
df['Hospitalized'] = np.where(df['Date'] >= hospitalized_breakdown_start, 0, df['Hospitalized']) #only for visualization
hospitalized_df = df[["Date", "Hospitalized", "Hospitalized Hospital", "Hospitalized Field"]].melt('Date', var_name='cols', value_name='vals')

# Forward fill some cumulative numbers that have gaps in data

col_to_ffill = ['Vac Group Medical Staff 1 Cum', 
                'Vac Group Other Frontline Staff 2 Cum',
                'Vac Group Over 60 1 Cum',
                'Vac Group Risk: Disease 1 Cum',
                'Vac Group Risk: Location 1 Cum',
                'Vac Given 1 Cum',
                'Vac Given 2 Cum']
df.loc[:,col_to_ffill] = df.loc[:,col_to_ffill].ffill()

# Calculate daily vaccination administered

df['Shot 1 Administered Daily'] = df['Vac Given 1 Cum'].diff()
df['Shot 2 Administered Daily'] = df['Vac Given 2 Cum'].diff()
vac_daily_df = df[["Date", "Shot 1 Administered Daily", "Shot 2 Administered Daily"]].melt('Date', var_name='cols', value_name='vals')

# Wide Plot

fig, axs = plt.subplots(3, 2, figsize=(12,12), sharex=True)
date_form = DateFormatter("%d-%b")
fmt_month = mdates.MonthLocator()

axs[0, 0].bar(df['Date'], df['Cases'], 1, label='cases')
axs[0, 0].set_title(f'Daily New Cases: {latest_new_confirmed:,d}')
axs[0, 0].set(ylabel="")
axs[0, 0].yaxis.set_major_formatter(ticker.EngFormatter())

axs[0, 1].bar(df['Date'], df['Tested'], 1, label='tested')
axs[0, 1].set_title(f'Daily Tests Administered: {latest_test_administered:,d}')
axs[0, 1].set(ylabel="")
axs[0, 1].yaxis.set_major_formatter(ticker.EngFormatter())

axs[1, 0].stackplot(df['Date'], df['Hospitalized'], 
                               df['Hospitalized Hospital'], 
                               df['Hospitalized Field'],
                               labels=['No Breakdown','Hospital', 'Field Hospital'])
axs[1, 0].yaxis.set_major_formatter(ticker.EngFormatter())
axs[1, 0].set_title(f'Number of People Hospitalized: {latest_hospitalized:,d}')
handles, labels = axs[1, 0].get_legend_handles_labels()  # get legend labels
axs[1, 0].legend(handles[::-1], labels[::-1], loc='upper left')  # reverse legend ordering to match chart

sns.lineplot(data=serious_df, x="Date", y="vals", hue="cols", legend="brief", ax=axs[1,1])
axs[1, 1].set_title('Number of Serious Cases and Deaths')
axs[1, 1].set(ylabel="")
axs[1, 1].yaxis.set_major_formatter(ticker.EngFormatter())
handles, labels = axs[1, 1].get_legend_handles_labels() # get labels
axs[1, 1].legend(handles=handles[0:], labels=labels[0:]) # show all label except the first one (which is the title)
axs[1, 1].text(0.95, 0.05, f"{latest_deaths}", transform=axs[1, 1].transAxes)

axs[2, 0].bar(df['Date'], df['Shot 1 Administered Daily'], 1, label='shot1')
axs[2, 0].set_title('1st shot vaccine administered (daily)')
axs[2, 0].set(ylabel="")
axs[2, 0].yaxis.set_major_formatter(ticker.EngFormatter())

axs[2, 1].stackplot(df['Date'], df['Vac Group Medical Staff 1 Cum'], 
                               df['Vac Group Other Frontline Staff 2 Cum'], 
                               df['Vac Group Over 60 1 Cum'], 
                               df['Vac Group Risk: Disease 1 Cum'], 
                               df['Vac Group Risk: Location 1 Cum'], 
                               labels=['Medical Staff','Other Frontline Staff', 'Over 60', 'Risk: Disease', 'Risk: Location'])
axs[2, 1].yaxis.set_major_formatter(ticker.EngFormatter())
axs[2, 1].set_title("1st shot vaccine administered (cumulative)")
handles, labels = axs[2, 1].get_legend_handles_labels()  # get legend labels
axs[2, 1].legend(handles[::-1], labels[::-1], loc='upper left')  # reverse legend ordering to match chart

axs[0, 0].xaxis.set_major_locator(fmt_month)
axs[0, 0].xaxis.set_major_formatter(date_form)

fig.suptitle('Thailand COVID data up to ' + last_updated + '\n Data from https://github.com/djay/covidthailand')

if plot_save == True:
  fig.savefig('full_figure.png')

if plot_upload == True:
  canvas = FigureCanvas(fig) # renders figure onto canvas
  imdata = io.BytesIO() # prepares in-memory binary stream buffer (think of this as a txt file but purely in memory)
  canvas.print_png(imdata) # writes canvas object as a png file to the buffer. You can also use print_jpg, alternatively
  s3.Object('covidviz','full_figure.png').put(Body=imdata.getvalue(), ContentType='image/png') 
  s3.ObjectAcl('covidviz','full_figure.png').put(ACL='public-read')

# narrow plot for mobile screen

fig, axs = plt.subplots(6, 1, figsize=(6, 9.5), sharex=True)

axs[0].bar(df['Date'], df['Cases'], 1, label='cases')
axs[0].set_title(f'Daily New Cases: {latest_new_confirmed:,d}')
axs[0].set(ylabel="")
axs[0].yaxis.set_major_formatter(ticker.EngFormatter())

axs[1].bar(df['Date'], df['Tested'], 1, label='tested')
axs[1].set_title(f'Daily Tests Administered: {latest_test_administered:,d}')
axs[1].set(ylabel="")
axs[1].yaxis.set_major_formatter(ticker.EngFormatter())

axs[2].stackplot(df['Date'], df['Hospitalized'], 
                               df['Hospitalized Hospital'], 
                               df['Hospitalized Field'],
                               labels=['No Breakdown','Hospital', 'Field Hospital'])
axs[2].yaxis.set_major_formatter(ticker.EngFormatter())
axs[2].set_title(f'Number of People Hospitalized: {latest_hospitalized:,d}')
handles, labels = axs[2].get_legend_handles_labels()  # get legend labels
axs[2].legend(handles[::-1], labels[::-1], loc='upper left')  # reverse legend ordering to match chart

sns.lineplot(data=serious_df, x="Date", y="vals", hue="cols", legend="brief", ax=axs[3])
axs[3].set_title('Number of Serious Cases and Deaths')
axs[3].set(ylabel="")
axs[3].yaxis.set_major_formatter(ticker.EngFormatter())
handles, labels = axs[3].get_legend_handles_labels() # get labels
axs[3].legend(handles=handles[0:], labels=labels[0:]) # show all label except the first one (which is the title)
axs[3].text(0.95, 0.05, f"{latest_deaths}", transform=axs[3].transAxes)

# sns.lineplot(data=df, x="Date", y="Tested", ax=axs[2,0])
# axs[2, 0].set_title('Daily Tests Administered')
# axs[2, 0].set(ylabel="")
# axs[2, 0].yaxis.set_major_formatter(ticker.EngFormatter())

axs[4].bar(df['Date'], df['Shot 1 Administered Daily'], 1, label='shot1')
axs[4].set_title('1st shot vaccine administered (daily)')
axs[4].set(ylabel="")
axs[4].yaxis.set_major_formatter(ticker.EngFormatter())

axs[5].stackplot(df['Date'], df['Vac Group Medical Staff 1 Cum'], 
                               df['Vac Group Other Frontline Staff 2 Cum'], 
                               df['Vac Group Over 60 1 Cum'], 
                               df['Vac Group Risk: Disease 1 Cum'], 
                               df['Vac Group Risk: Location 1 Cum'], 
                               labels=['Medical Staff','Other Frontline Staff', 'Over 60', 'Risk: Disease', 'Risk: Location'])
axs[5].yaxis.set_major_formatter(ticker.EngFormatter())
axs[5].set_title("1st shot vaccine administered (cumulative)")
handles, labels = axs[5].get_legend_handles_labels()  # get legend labels
axs[5].legend(handles[::-1], labels[::-1], labelspacing=0.1, loc='upper left')  # reverse legend ordering to match chart

axs[0].xaxis.set_major_locator(fmt_month)
axs[0].xaxis.set_major_formatter(date_form)

fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.suptitle('Thailand COVID data up to ' + last_updated + '\n Data from https://github.com/djay/covidthailand')

if plot_save == True:
  fig.savefig('mobile_figure.png')

if plot_upload == True:
  canvas = FigureCanvas(fig) # renders figure onto canvas
  imdata = io.BytesIO() # prepares in-memory binary stream buffer (think of this as a txt file but purely in memory)
  canvas.print_png(imdata) # writes canvas object as a png file to the buffer. You can also use print_jpg, alternatively
  s3.Object('covidviz','mobile_figure.png').put(Body=imdata.getvalue(), ContentType='image/png') 
  s3.ObjectAcl('covidviz','mobile_figure.png').put(ACL='public-read')


#
# Vaccination Summary
#

vac_df = pd.read_parquet('vaccination.parquet')
top10province = vac_df[vac_df['Date'] == '2021-05-13'].sort_values(by="Vac Given 1 Cum", ascending=False)['Province'][:10].values
vac_df_top10 = vac_df[vac_df['Province'].isin(top10province)]
vac_bkk_df = vac_df[vac_df['Province'] == "Bangkok"]

fig, axs = plt.subplots(3, 1, figsize=(6, 9.5), sharex=True)

sns.lineplot(data=vac_df_top10, x="Date", y="Vac Given 1 Cum", hue="Province", ax=axs[0])
axs[0].set_title(f'Top 10 Most Vaccination Administered (1st shot)')
axs[0].set(ylabel="")
handles, labels = axs[0].get_legend_handles_labels()  # get legend labels
axs[0].legend(handles[::1], labels[::1], loc='upper left')  # reverse legend ordering to match chart
axs[0].yaxis.set_major_formatter(ticker.EngFormatter())

sns.lineplot(data=vac_df_top10, x="Date", y="Vac Given 2 Cum", hue="Province", ax=axs[1])
axs[1].set_title(f'Top 10 Most Vaccination Administered (2nd shot)')
axs[1].set(ylabel="")
handles, labels = axs[1].get_legend_handles_labels()  # get legend labels
axs[1].legend(handles[::1], labels[::1], loc='upper left')  # reverse legend ordering to match chart
axs[1].yaxis.set_major_formatter(ticker.EngFormatter())

axs[2].stackplot(vac_bkk_df['Date'], vac_bkk_df['Vac Group Medical Staff 1 Cum'], 
                               vac_bkk_df['Vac Group Other Frontline Staff 2 Cum'], 
                               vac_bkk_df['Vac Group Over 60 1 Cum'], 
                               vac_bkk_df['Vac Group Risk: Disease 1 Cum'], 
                               vac_bkk_df['Vac Group Risk: Location 1 Cum'], 
                               labels=['Medical Staff','Other Frontline Staff', 'Over 60', 'Risk: Disease', 'Risk: Location'])
axs[2].yaxis.set_major_formatter(ticker.EngFormatter())
axs[2].set_title("Bangkok 1st shot vaccine administered (cumulative)")
handles, labels = axs[2].get_legend_handles_labels()  # get legend labels
axs[2].legend(handles[::-1], labels[::-1], labelspacing=0.1, loc='upper left')  # reverse legend ordering to match chart

axs[0].xaxis.set_major_locator(fmt_month)
axs[0].xaxis.set_major_formatter(date_form)

fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.suptitle('Thailand COVID Vaccination data up to ' + last_updated + '\n Data from https://github.com/djay/covidthailand')

if plot_save == True:
  fig.savefig('mobile_vaccination.png')

if plot_upload == True:
  canvas = FigureCanvas(fig) # renders figure onto canvas
  imdata = io.BytesIO() # prepares in-memory binary stream buffer (think of this as a txt file but purely in memory)
  canvas.print_png(imdata) # writes canvas object as a png file to the buffer. You can also use print_jpg, alternatively
  s3.Object('covidviz','mobile_vaccination.png').put(Body=imdata.getvalue(), ContentType='image/png') 
  s3.ObjectAcl('covidviz','mobile_vaccination.png').put(ACL='public-read')