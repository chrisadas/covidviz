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

# Filter data excluding dates prior to 2021
# Uncomment to exclude today's data
# yesterday = datetime.now() - timedelta(days=1)
# datemask = (df['Date'] >= begin_date) & (df['Date'] <= yesterday)
datemask = (df['Date'] >= begin_date)
df = df.loc[datemask]
df.set_index("Date", drop=False, inplace=True)

last_updated = df['Date'][df.index[-1]].strftime("%d %B %Y")
print(f"Latest data: {last_updated}")

# Calculate Positive Rate

df["Tests XLS (MA)"] = df["Tests XLS"].rolling(7, 1, center=True).mean()
df["Pos XLS (MA)"] = df["Pos XLS"].rolling(7, 3, center=True).mean()
df["Positive Rate (7-day MA)"] = df["Pos XLS (MA)"] / df["Tests XLS (MA)"] * 100

# Dataframe for serious cases plotting

serious_df = df[["Date", "Hospitalized Respirator", "Hospitalized Severe", "Deaths"]].melt('Date', var_name='cols', value_name='vals')

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

sns.lineplot(data=df, x="Date", y="Cases", ax=axs[0,0])
axs[0, 0].set_title('Daily New Cases')
axs[0, 0].set(ylabel="")
axs[0,0].yaxis.set_major_formatter(ticker.EngFormatter())

# sns.lineplot(data=df, x="Date", y="Positive Rate (7-day MA)", ax=axs[0,1])
# axs[0, 1].set_title('% Tested and Positive (7-day MA)')
# axs[0, 1].set(ylabel="")
# axs[0,1].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))

sns.lineplot(data=df, x="Date", y="Tested", ax=axs[0,1])
axs[0, 1].set_title('Daily Tests Administered')
axs[0, 1].set(ylabel="")
axs[0, 1].yaxis.set_major_formatter(ticker.EngFormatter())

sns.lineplot(data=df, x="Date", y="Hospitalized", ax=axs[1,0])
axs[1, 0].set_title('Number of People Hospitalized')
axs[1, 0].set(ylabel="")
axs[1, 0].yaxis.set_major_formatter(ticker.EngFormatter())

sns.lineplot(data=serious_df, x="Date", y="vals", hue="cols", legend="brief", ax=axs[1,1])
axs[1, 1].set_title('Number of Serious Cases and Deaths')
axs[1, 1].set(ylabel="")
axs[1, 1].yaxis.set_major_formatter(ticker.EngFormatter())
handles, labels = axs[1, 1].get_legend_handles_labels() # get labels
axs[1, 1].legend(handles=handles[0:], labels=labels[0:]) # show all label except the first one (which is the title)

# sns.lineplot(data=df, x="Date", y="Tested", ax=axs[2,0])
# axs[2, 0].set_title('Daily Tests Administered')
# axs[2, 0].set(ylabel="")
# axs[2, 0].yaxis.set_major_formatter(ticker.EngFormatter())

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
date_form = DateFormatter("%d-%b")
fmt_month = mdates.MonthLocator()

sns.lineplot(data=df, x="Date", y="Cases", ax=axs[0])
axs[0].set_title('Daily New Cases')
axs[0].set(ylabel="")
axs[0].yaxis.set_major_formatter(ticker.EngFormatter())

# sns.lineplot(data=df, x="Date", y="Positive Rate (7-day MA)", ax=axs[0,1])
# axs[0, 1].set_title('% Tested and Positive (7-day MA)')
# axs[0, 1].set(ylabel="")
# axs[0,1].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))

sns.lineplot(data=df, x="Date", y="Tested", ax=axs[1])
axs[1].set_title('Daily Tests Administered')
axs[1].set(ylabel="")
axs[1].yaxis.set_major_formatter(ticker.EngFormatter())

sns.lineplot(data=df, x="Date", y="Hospitalized", ax=axs[2])
axs[2].set_title('Number of People Hospitalized')
axs[2].set(ylabel="")
axs[2].yaxis.set_major_formatter(ticker.EngFormatter())

sns.lineplot(data=serious_df, x="Date", y="vals", hue="cols", legend="brief", ax=axs[3])
axs[3].set_title('Number of Serious Cases and Deaths')
axs[3].set(ylabel="")
axs[3].yaxis.set_major_formatter(ticker.EngFormatter())
handles, labels = axs[3].get_legend_handles_labels() # get labels
axs[3].legend(handles=handles[0:], labels=labels[0:]) # show all label except the first one (which is the title)

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