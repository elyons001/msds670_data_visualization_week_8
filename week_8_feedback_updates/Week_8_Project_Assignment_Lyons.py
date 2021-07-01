#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  20 19:16:48 2021

@author: earle

Title: Week 8 Project Assignment Lyons
Date: 01JUL2021
Author: Earle Lyons
Purpose: Create visuals for Week 8 Assignment Project
    - Line plot and small multiple plots
    - Small multiple plots
    - Seaborn heatmap
    - GeoPandas choropleth map
    - Line plots
Inputs:
Outputs:
Notes:
     MSDS670 Data Visualization (Regis University)
     21M8W1: 05/03/21-06/27/21

Data Sources:
    https://data.bls.gov/timeseries/LNU04000000?years_option=all_years&periods_option=specific_periods&periods=Annual+Data
    https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-poverty-people.html
    https://www.census.gov/data/datasets/2010/demo/saipe/2010-state-and-county.html
    https://www.census.gov/data/datasets/2019/demo/saipe/2019-state-and-county.html
    https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
    https://www.census.gov/library/publications/2020/demo/p60-270.html

Data Files: 
    https://data.bls.gov/timeseries/LNU04000000?years_option=all_years&periods_option=specific_periods&periods=Annual+Data
    https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov2.xlsx
    https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov21.xlsx
    https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov7.xlsx
    https://www2.census.gov/programs-surveys/demo/tables/p60/270/tableC1.xlsx
    https://www2.census.gov/programs-surveys/saipe/datasets/2010/2010-state-and-county/est10all.xls
    https://www2.census.gov/programs-surveys/saipe/datasets/2019/2019-state-and-county/est19all.xls
    https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip

References:
    https://towardsdatascience.com/how-to-split-shapefiles-e8a8ac494189
    https://stackoverflow.com/questions/48470251/move-tick-marks-at-the-top-of-the-seaborn-plot?noredirect=1&lq=1
    https://gis.stackexchange.com/questions/343881/using-vmin-and-vmax-does-not-change-anything-for-my-plots
"""

#%% Import libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
from geopandas import GeoDataFrame

#%% Set DPI
dpi = 300

#%% Reset Matplotlib rcParams to default
# https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rc_file_defaults
mpl.rc_file_defaults()

#%% Create DataFrames

#%% Create overall poverty rate DataFrame
pov_rate_overall = r'https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov2.xlsx'
headers = ['Year', 'Poverty %']
pr_oa_df = pd.read_excel(pov_rate_overall, names=headers, 
                         usecols='A,D',
                         skiprows=[0,1,2,3,4,5,6,11,16], 
                         nrows=12)
# Rename year to remove extra characters, cast 'Year' column as int, and 
# set index
pr_oa_df.at[2, 'Year']='2017'
pr_oa_df.at[6, 'Year']='2013'
pr_oa_df.at[9, 'Year']='2010'
pr_oa_df = pr_oa_df.copy()
pr_oa_df['Year'] = pr_oa_df['Year'].astype(int)
pr_oa_df.set_index('Year', inplace=True)
pr_oa_df.sort_index(inplace=True, ascending=True)

# Create family poverty rate DataFrame
pov_rate_fam = r'https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov2.xlsx'
headers = ['Year', 'All families %', 'Families with female householder %', 
           'Unrelated individuals %']
pr_fam_df = pd.read_excel(pov_rate_fam, names=headers, 
                         usecols='A,G,J,M',
                         skiprows=[0,1,2,3,4,5,6,11,16], 
                         nrows=12)
# Rename year to remove extra characters, cast 'Year' column as int, and 
# set index
pr_fam_df.at[2, 'Year']='2017'
pr_fam_df.at[6, 'Year']='2013'
pr_fam_df.at[9, 'Year']='2010'
pr_fam_df = pr_fam_df.copy()
pr_fam_df['Year'] = pr_fam_df['Year'].astype(int)
pr_fam_df.set_index('Year', inplace=True)
pr_fam_df.sort_index(inplace=True, ascending=True)

#%% Create sex poverty rate DataFrame
pov_rate_sex_age = r'https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov7.xlsx'
# Create columns for headers
headers = ['Year', 'Total', 'Male_Total', 'Male_Total_Below_Num', 
           'Male_Total_Below_%', 'Male_Under18_Total', 
           'Male_Under18_Below_Num', 'Male_Under18_Below_%',
           'Male_18-64_Total', 'Male_18-64_Below_Num', 
           'Male_18-64_Below_%', 'Male_65-Over_Total', 
           'Male_65-Over_Below_Num', 'Male_65-Over_Below_%',
           'Female_Total', 'Female_Total_Below_Num', 
           'Female_Total_Below_%', 'Female_Under18_Total', 
           'Female_Under18_Below_Num', 'Female_Under18_Below_%',
           'Female_18-64_Total', 'Female_18-64_Below_Num', 
           'Female_18-64_Below_%', 'Female_65-Over_Total', 
           'Female_65-Over_Below_Num', 'Female_65-Over_Below_%']
# Read Excel worksheet into DataFrame
pr_sa_df = pd.read_excel(pov_rate_sex_age, names=headers, 
                         skiprows=[0,1,2,3,4,5,10,15], nrows=12)
# Rename year to remove extra characters, cast 'Year' column as int, and 
# set index
pr_sa_df.at[2, 'Year']='2017'
pr_sa_df.at[6, 'Year']='2013'
pr_sa_df.at[9, 'Year']='2010'
pr_sa_df.copy = pr_sa_df.copy()
pr_sa_df['Year'] = pr_sa_df['Year'].astype(int)
pr_sa_df.set_index('Year', inplace=True)
pr_sa_df.sort_index(inplace=True, ascending=True)
# Create male by age DataFrame
male_pr_sa_df = pr_sa_df[['Male_Total_Below_%', 
                          'Male_Under18_Below_%',
                          'Male_18-64_Below_%', 
                          'Male_65-Over_Below_%']].copy()
# Create female by age DataFrame
female_pr_sa_df = pr_sa_df[['Female_Total_Below_%', 
                            'Female_Under18_Below_%',
                            'Female_18-64_Below_%', 
                            'Female_65-Over_Below_%']].copy()

#%% Create race poverty rate DataFrame
pov_rate_race = r'https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov2.xlsx'
headers = ['Year', 'Percent']
pr_race_whtalo = pd.read_excel(pov_rate_race, 
                               names=['Year','White alone'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 75), 
                               nrows=12)
pr_race_whtnhi = pd.read_excel(pov_rate_race, 
                               names=['Year','White alone, not hispanic'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 148), 
                               nrows=12)
pr_race_blkalo = pd.read_excel(pov_rate_race, 
                               names=['Year','Black alone'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 232), 
                               nrows=12)
pr_race_blkalc = pd.read_excel(pov_rate_race, 
                               names=['Year','Black alone or combination'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 207), 
                               nrows=12)
pr_race_asialc = pd.read_excel(pov_rate_race, 
                               names=['Year','Asian alone or combination'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 299), 
                               nrows=12)
pr_race_asialo = pd.read_excel(pov_rate_race, 
                               names=['Year','Asian alone'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 324), 
                               nrows=12)
pr_race_hisany = pd.read_excel(pov_rate_race, 
                               names=['Year','Hispanic (any race)'], 
                               index_col='Year', usecols='A,D',
                               skiprows=range(0, 369), 
                               nrows=12)
pr_race_frames = [pr_race_whtalo, pr_race_whtnhi, 
                  pr_race_blkalo, pr_race_blkalc, 
                  pr_race_asialc, pr_race_asialo, 
                  pr_race_hisany]
pr_race_df = pd.concat(pr_race_frames, axis=1)
pr_race_df = pr_race_df.copy()
pr_race_df = pr_race_df.drop(2017)
pr_race_df = pr_race_df.drop('2013 (18)')
pr_race_df = pr_race_df.rename(index={'2017 (21)': 2017})
pr_race_df = pr_race_df.rename(index={'2013 (19)': 2013})
pr_race_df = pr_race_df.rename(index={'2010 (17)': 2010})
pr_race_df.index = pd.to_numeric(pr_race_df.index)

#%% Create poverty rate by state DataFrame
pov_rate_state_url = r'https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov21.xlsx'
pr_st_percent_2019_df = pd.read_excel(pov_rate_state_url, header=4, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2019_df.rename(columns={'Percent': '2019'}, inplace=True)
pr_st_percent_2018_df = pd.read_excel(pov_rate_state_url, header=57, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2018_df.rename(columns={'Percent': '2018'}, inplace=True)
pr_st_percent_2017_df = pd.read_excel(pov_rate_state_url, header=110, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2017_df.rename(columns={'Percent': '2017'}, inplace=True)
pr_st_percent_2016_df = pd.read_excel(pov_rate_state_url, header=216, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2016_df.rename(columns={'Percent': '2016'}, inplace=True)
pr_st_percent_2015_df = pd.read_excel(pov_rate_state_url, header=269, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2015_df.rename(columns={'Percent': '2015'}, inplace=True)
pr_st_percent_2014_df = pd.read_excel(pov_rate_state_url, header=322, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2014_df.rename(columns={'Percent': '2014'}, inplace=True)
pr_st_percent_2013_df = pd.read_excel(pov_rate_state_url, header=375, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2013_df.rename(columns={'Percent': '2013'}, inplace=True)
pr_st_percent_2012_df = pd.read_excel(pov_rate_state_url, header=481, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2012_df.rename(columns={'Percent': '2012'}, inplace=True)
pr_st_percent_2011_df = pd.read_excel(pov_rate_state_url, header=534, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2011_df.rename(columns={'Percent': '2011'}, inplace=True)
pr_st_percent_2010_df = pd.read_excel(pov_rate_state_url, header=587, 
                                      index_col='STATE', 
                                      usecols='A,E', nrows=51)
pr_st_percent_2010_df.rename(columns={'Percent': '2010'}, inplace=True)
pr_st_percent_frames = [pr_st_percent_2010_df, pr_st_percent_2011_df, 
                        pr_st_percent_2012_df, pr_st_percent_2012_df, 
                        pr_st_percent_2013_df, pr_st_percent_2014_df, 
                        pr_st_percent_2015_df, pr_st_percent_2016_df, 
                        pr_st_percent_2017_df, pr_st_percent_2018_df, 
                        pr_st_percent_2019_df]
pr_st_percent_df = pd.concat(pr_st_percent_frames, axis=1)

#%% Create poverty rate by county DataFrame
# 2010
pov_rate_county_url = r'https://www2.census.gov/programs-surveys/saipe/datasets/2010/2010-state-and-county/est10all.xls'
pr_cnty_percent_2010_df = pd.read_excel(pov_rate_county_url, header=2, 
                                        dtype={'State FIPS':object}, 
                                        usecols='A:D,H', nrows=3195)
pr_cnty_percent_2010_df["County FIPS"] = pr_cnty_percent_2010_df["County FIPS"].map("{:03}".format)
# Drop United States and individual states
pr_cnty_percent_2010_df = pr_cnty_percent_2010_df[pr_cnty_percent_2010_df['County FIPS']!= '000']
# Create 'GEOID' column with 'State FIPS Code' and 'County FIPS Code' columns
pr_cnty_percent_2010_df['GEOID'] = pr_cnty_percent_2010_df['State FIPS'] + pr_cnty_percent_2010_df['County FIPS']
# Rename 'Poverty Percent, All Ages' column to '2010 Poverty %'
pr_cnty_percent_2010_df.rename(columns={'Poverty Percent All Ages': '2010 Poverty %'}, inplace=True)
# Display counties that are not present in 2010 data and have '.' as a value
#pr_cnty_percent_2010_df.loc[pr_cnty_percent_2010_df['GEOID'] == '02270']
#pr_cnty_percent_2010_df.loc[pr_cnty_percent_2010_df['GEOID'] == '46113']
#pr_cnty_percent_2010_df.loc[pr_cnty_percent_2010_df['GEOID'] == '51515']
#pr_cnty_percent_2010_df.loc[pr_cnty_percent_2010_df['GEOID'] == '15005']
# Drop counties that are not present in 2010 data and have '.' as a value
pr_cnty_percent_2010_df = pr_cnty_percent_2010_df[pr_cnty_percent_2010_df['GEOID']!= '02270']
pr_cnty_percent_2010_df = pr_cnty_percent_2010_df[pr_cnty_percent_2010_df['GEOID']!= '46113']
pr_cnty_percent_2010_df = pr_cnty_percent_2010_df[pr_cnty_percent_2010_df['GEOID']!= '51515']
pr_cnty_percent_2010_df = pr_cnty_percent_2010_df[pr_cnty_percent_2010_df['GEOID']!= '15005']
# Reset index and set index to 'GEOID' column
pr_cnty_percent_2010_df.reset_index(drop=True)
pr_cnty_percent_2010_df.set_index('GEOID', inplace=True)
# 2019
pov_rate_county_url = r'https://www2.census.gov/programs-surveys/saipe/datasets/2019/2019-state-and-county/est19all.xls'
pr_cnty_percent_2019_df = pd.read_excel(pov_rate_county_url, header=3, 
                                        dtype={'State FIPS Code':object, 
                                               'County FIPS Code':object},
                                        usecols='A:D,H', nrows=3194)
# Drop United States and individual states
pr_cnty_percent_2019_df = pr_cnty_percent_2019_df[pr_cnty_percent_2019_df['County FIPS Code']!= '000']
# Create 'GEOID' column with 'State FIPS Code' and 'County FIPS Code' columns
pr_cnty_percent_2019_df['GEOID'] = pr_cnty_percent_2019_df['State FIPS Code'] + pr_cnty_percent_2019_df['County FIPS Code']
# Rename 'Poverty Percent, All Ages' column to '2019 Poverty %'
pr_cnty_percent_2019_df.rename(columns={'Poverty Percent, All Ages': '2019 Poverty %'}, inplace=True)
# Display counties that are not present in 2019 data and have '.' as a value
#pr_cnty_percent_2019_df.loc[pr_cnty_percent_2019_df['GEOID'] == '02158']
#pr_cnty_percent_2019_df.loc[pr_cnty_percent_2019_df['GEOID'] == '46102']
#pr_cnty_percent_2019_df.loc[pr_cnty_percent_2019_df['GEOID'] == '15005']
# Drop counties that are not present in 2019 data and have '.' as a value
pr_cnty_percent_2019_df = pr_cnty_percent_2019_df[pr_cnty_percent_2019_df['GEOID']!= '02158']
pr_cnty_percent_2019_df = pr_cnty_percent_2019_df[pr_cnty_percent_2019_df['GEOID']!= '46102']
pr_cnty_percent_2019_df = pr_cnty_percent_2019_df[pr_cnty_percent_2019_df['GEOID']!= '15005']
# Reset index and set index to 'GEOID' column
pr_cnty_percent_2019_df.reset_index(drop=True)
pr_cnty_percent_2019_df.set_index('GEOID', inplace=True)

# Combine DataFrames and use only 
pr_cnty_percent_frames = [pr_cnty_percent_2010_df, pr_cnty_percent_2019_df]
pr_cnty_percent_df = pd.concat(pr_cnty_percent_frames, axis=1)
pr_cnty_percent_df = pr_cnty_percent_df[['2010 Poverty %',
                                         '2019 Poverty %']]

#%% Create county GeoPandas DataFrame from shape file
# Read county shape file from Census.gov
url = 'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip'
gdf_counties = geopandas.read_file(url)
gdf_counties.set_index('GEOID', inplace=True)
# Merge county shape file with DataFrames containing values
pr_cnty_geo_df = pr_cnty_percent_df.merge(gdf_counties, how='left', on='GEOID')
pr_cnty_geo_df = GeoDataFrame(pr_cnty_geo_df)
# Cast Poverty % columns to a float dtype
pr_cnty_geo_df = pr_cnty_geo_df.copy()
pr_cnty_geo_df['2010 Poverty %'] = pr_cnty_geo_df['2010 Poverty %'].astype(float)
pr_cnty_geo_df['2019 Poverty %'] = pr_cnty_geo_df['2019 Poverty %'].astype(float)
# Include only U.S. contiguous states
pr_cnty_geo_df_cont = pr_cnty_geo_df[~pr_cnty_geo_df['STUSPS'].isin(['AK', 'HI'])]
# Include AK
pr_cnty_geo_df_ak = pr_cnty_geo_df[pr_cnty_geo_df['STUSPS'].isin(['AK'])]
# Include HI
pr_cnty_geo_df_hi = pr_cnty_geo_df[pr_cnty_geo_df['STUSPS'].isin(['HI'])]

#%%  Create median income DataFrame
median_inc_url = r'https://www2.census.gov/programs-surveys/demo/tables/p60/270/tableC1.xlsx'
headers = ['Year', 'Median income']
median_inc_df = pd.read_excel(median_inc_url, names=headers, usecols='A,B',
                              skiprows=[0,1,2,3,4,9,14], 
                              nrows=12)
# Rename year to remove extra characters, cast 'Year' column as int, and 
# set index
median_inc_df.at[2, 'Year']='2017'
median_inc_df.at[6, 'Year']='2013'
median_inc_df.at[9, 'Year']='2010'
median_inc_df = median_inc_df.copy()
median_inc_df['Year'] = median_inc_df['Year'].astype(int)
median_inc_df.set_index('Year', inplace=True)
median_inc_df.sort_index(inplace=True, ascending=True)

#%% Create unemployment rate DataFrame
unemp_rt_url = r'https://data.bls.gov/timeseries/LNU04000000?years_option=all_years&periods_option=specific_periods&periods=Annual+Data'
unemp_rt_html = pd.read_html(unemp_rt_url, attrs={'id':'table0'})
unemp_rt_df = unemp_rt_html[0]
# Rename year to remove extra characters, cast 'Year' column as int, and 
# set index
unemp_rt_df = unemp_rt_df[(unemp_rt_df['Year'] >= 2010) 
                          & (unemp_rt_df['Year'] <= 2019)]
unemp_rt_df= unemp_rt_df.copy()
unemp_rt_df['Year'] = unemp_rt_df['Year'].astype(int)
unemp_rt_df.set_index('Year', inplace=True)
unemp_rt_df.sort_index(inplace=True, ascending=True)
unemp_rt_df.rename(columns={'Annual': 'Unemployment rate'}, inplace=True)

#%% Create labels for legend_kwds
#labels = [12, 14, 16, 18, 20, 22]
labels = ['0 to 12', '12 to 14', '14 to 16', '16 to 18', '18 to 20', 
          '20 to 22', '22 to 52']
# Create bins for classification_kwds
bins = dict(bins=[12, 14, 16, 18, 20, 22])

#%% Create visuals

#%% Create 'U.S. Poverty rate overall and by family' visual
# Create figure using GridSpec and customize layout of subplots
fig = plt.figure(figsize=(8, 4), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[-1, 0])
ax3 = fig.add_subplot(gs[-1, 1])
ax4 = fig.add_subplot(gs[-1, 2])
# Create suptitle
fig.suptitle('U.S. poverty rate overall and by family/individuals', 
             x=0.34, y=1.1, 
             fontsize=16)
gs.tight_layout(fig)
# Plot line and scatter
ax1.plot(pr_oa_df.index, pr_oa_df['Poverty %'], color='black', linewidth=3)
ax1.scatter(2015, 13.5, color='#D62828')
# Plot small multiple
ax2.plot(pr_fam_df.index, pr_fam_df['All families %'], color='#7B9E89', 
         linewidth=3)
ax3.plot(pr_fam_df.index, pr_fam_df['Families with female householder %'], 
         color='#7B9E89', linewidth=3)
ax4.plot(pr_fam_df.index, pr_fam_df['Unrelated individuals %'], 
         color='#7B9E89', linewidth=3)
# Add axvline
ax1.axvline(2015, ymin=0.1, ymax=0.44, color='#343A40', ls='dotted')
ax1.axvline(2015, ymin=0.58, ymax=1.0, color='#343A40', ls='dotted')
# Set titles
ax1.set_title('Overall', loc='left', weight='semibold')
ax2.set_title('Families and individuals', loc='left', y=1.2, 
              fontweight='semibold')
ax2.set_title('All families', loc='center')
ax3.set_title('Female householder', loc='center')
ax4.set_title('Unrelated individuals', loc='center')
# Create ax text
#2010	15.1
#2019	10.5
ax1.text(2010, 17.1, '15.1%', fontsize='medium')
ax1.text(2009.75, 1.5, '46.9M', color='#7A777E', 
         fontsize='medium')
ax1.text(2014, 16.8, '14.8%', color='#343A40', fontsize='medium', 
         ha='center',)
ax1.text(2015, 9.5, '13.5%', color='#D62828', fontsize='medium', 
         ha='center')
ax1.text(2015.1, 4.0, 'Median househould income', color='#343A40', 
         fontsize='medium', ha='left')
ax1.text(2015.1, 2.0, 'increased 5.2%', color='#343A40', 
         fontsize='medium', ha='left')
ax1.text(2019, 6.5, '10.5%', fontsize='medium', ha='right')
ax1.text(2019.25, 1.5, '34.0M', color='#7A777E', 
         fontsize='medium', ha='right')
#2010	13.2	34.3	22.9
#2019	8.5	24.3	18.8
ax2.text(2010, 15.2, '13.2%', fontsize='small')
ax2.text(2009.25, 1, '33.1M', color='#7A777E', fontsize='small')
ax2.text(2019, 4.5, '8.5%', fontsize='small', ha='right')
ax2.text(2019.75, 1, '22.4M', color='#7A777E', fontsize='small', ha='right')
ax3.text(2010, 36.3, '34.3%', fontsize='small')
ax3.text(2009.25, 1, '15.9M', color='#7A777E', fontsize='small')
ax3.text(2019, 20.3, '24.3%', fontsize='small', ha='right')
ax3.text(2019.75, 1, '11.3M', color='#7A777E', fontsize='small', ha='right')
ax4.text(2010, 24.9, '22.9%', fontsize='small')
ax4.text(2009.25, 1, '12.4M', color='#7A777E', fontsize='small')
ax4.text(2019, 14.8, '18.8%', fontsize='small', ha='right')
ax4.text(2019.75, 1, '11.3M', color='#7A777E', fontsize='small', ha='right')
# Set y limits
ax1.set_ylim(0, 20)
ax2.set_ylim(0, 40)
ax3.set_ylim(0, 40)
ax4.set_ylim(0, 40)
# Set x and y parameters
ax1.tick_params(colors='#7A777E')
ax2.tick_params(colors='#7A777E')
ax3.tick_params(colors='#7A777E')
ax4.tick_params(colors='#7A777E')
# Set x ticks
ax1.set_xticks([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
ax1.set_xticklabels(['2010', '2011', '2012', '2013', '2014', '2015', '2016',
                     '2017', '2018', '2019'])
ax2.set_xticks([2010, 2019])
ax2.set_xticklabels(['2010', '2019'])
ax3.set_xticks([2010, 2019])
ax3.set_xticklabels(['2010', '2019'])
ax4.set_xticks([2010, 2019])
ax4.set_xticklabels(['2010', '2019'])
# Set y ticks
ax1.set_yticks([])
ax1.set_yticklabels([])
# ax1.set_yticks([0, 10, 20])
# ax1.set_yticklabels(['0', '10', '20'])
ax2.set_yticks([0, 10, 20, 30])
ax2.set_yticklabels(['0', '10', '20', '30'])
ax3.set_yticks([])
ax3.set_yticklabels([])
ax4.set_yticks([])
ax4.set_yticklabels([])
# Set all spines invisible
ax1.spines.right.set_visible(False)
ax1.spines.left.set_visible(False)
ax1.spines.top.set_visible(False)
ax1.spines.bottom.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
ax2.spines.top.set_visible(True)
ax2.spines.bottom.set_visible(False)
ax3.spines.right.set_visible(False)
ax3.spines.left.set_visible(False)
ax3.spines.top.set_visible(True)
ax3.spines.bottom.set_visible(False)
ax4.spines.right.set_visible(False)
ax4.spines.left.set_visible(False)
ax4.spines.top.set_visible(True)
ax4.spines.bottom.set_visible(False)
plt.show()
plot1_filename = 'week_8_matplotlib_lyons_1_25JUN21.png'
fig.savefig(plot1_filename, dpi=dpi, bbox_inches='tight')

#%% Create 'U.S. poverty by sex, age, and race' visual
# Create figure using GridSpec and customize layout of subplots
fig = plt.figure(figsize=(8, 6), constrained_layout=True)
gs = fig.add_gridspec(nrows=3, ncols=4, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, 0])
ax6 = fig.add_subplot(gs[1, 1])
ax7 = fig.add_subplot(gs[1, 2])
ax8 = fig.add_subplot(gs[1, 3])
ax9 = fig.add_subplot(gs[2, 0])
ax10 = fig.add_subplot(gs[2, 1])
ax11 = fig.add_subplot(gs[2, 2])
ax12 = fig.add_subplot(gs[2, 3])
# Create suptitle
fig.suptitle('U.S. poverty rate by sex, age, and race', 
             x=0.24, y=1.1, fontsize=16)
# Plot small multiples
ax1.plot(pr_sa_df.index, male_pr_sa_df['Male_Total_Below_%'], 
         color='#415A77', linewidth=3)
ax2.plot(pr_sa_df.index, male_pr_sa_df['Male_Under18_Below_%'], 
         color='#415A77', linewidth=3)
ax3.plot(pr_sa_df.index, male_pr_sa_df['Male_18-64_Below_%'], 
         color='#415A77', linewidth=3)
ax4.plot(pr_sa_df.index, male_pr_sa_df['Male_65-Over_Below_%'], 
         color='#AC3D2C', linewidth=3)
ax5.plot(pr_sa_df.index, female_pr_sa_df['Female_Total_Below_%'], 
         color='#774C60', linewidth=3)
ax6.plot(pr_sa_df.index, female_pr_sa_df['Female_Under18_Below_%'], 
         color='#774C60', linewidth=3)
ax7.plot(pr_sa_df.index, female_pr_sa_df['Female_18-64_Below_%'], 
         color='#774C60', linewidth=3)
ax8.plot(pr_sa_df.index, female_pr_sa_df['Female_65-Over_Below_%'], 
         color='#F77F00', linewidth=3)
ax9.plot(pr_race_df.index, pr_race_df['White alone, not hispanic'], 
         color='#A49694', linewidth=3)
ax10.plot(pr_race_df.index, pr_race_df['Black alone'], 
          color='#A49694', linewidth=3)
ax11.plot(pr_race_df.index, pr_race_df['Asian alone'], 
          color='#A49694', linewidth=3)
ax12.plot(pr_race_df.index, pr_race_df['Hispanic (any race)'], 
          color='#A49694', linewidth=3)
# Set titles
ax1.set_title('Male - All ages', loc='center')
ax2.set_title('Male - Under 18', loc='center')
ax3.set_title('Male - 18 to 24', loc='center')
ax4.set_title('Male - 65 and over', loc='center')
ax5.set_title('Female - All ages', loc='center')
ax6.set_title('Female - Under 18', loc='center')
ax7.set_title('Female - 18 to 24', loc='center')
ax8.set_title('Female - 65 and over', loc='center')
ax9.set_title('White', loc='center')
ax10.set_title('Black', loc='center')
ax11.set_title('Asian', loc='center')
ax12.set_title('Hispanic', loc='center')
# Create ax text
#2010	14.0	22.2	12.0	6.7
#2019	9.4	14.4	8.1	7.2
ax1.text(2010, 16, '14.0%', fontsize='small')
ax1.text(2009.25, 1, '20.9M', color='#7A777E', fontsize='small')
ax1.text(2019, 5.4, '9.4%', fontsize='small', ha='right')
ax1.text(2019.75, 1, '15.0M', color='#7A777E', fontsize='small', ha='right')
ax2.text(2010, 24.2, '22.2%', fontsize='small')
ax2.text(2009.25, 1, '8.4M', color='#7A777E', fontsize='small')
ax2.text(2019, 10.4, '14.4%', fontsize='small', ha='right')
ax2.text(2019.75, 1, '5.3M', color='#7A777E', fontsize='small', ha='right')
ax3.text(2010, 14.0, '12.0%', fontsize='small')
ax3.text(2009.25, 1, '11.3M', color='#7A777E', fontsize='small')
ax3.text(2019, 4.1, '8.1%', fontsize='small', ha='right')
ax3.text(2019.75, 1, '7.9M', color='#7A777E', fontsize='small', ha='right')
ax4.text(2010, 2.7, '6.7%', fontsize='small')
ax4.text(2009.25, 1, '1.2M', color='#7A777E', fontsize='small')
ax4.text(2019, 9.2, '7.2%', fontsize='small', ha='right')
ax4.text(2019.75, 1, '1.8M', color='#7A777E', fontsize='small', ha='right')
#2010	16.3	21.9	15.5	10.7
#2019	11.5	14.5	10.8	10.3
ax5.text(2010, 18.3, '16.3%', fontsize='small')
ax5.text(2009.25, 1, '25.5M', color='#7A777E', fontsize='small')
ax5.text(2019, 8.5, '11.5%', fontsize='small', ha='right')
ax5.text(2019.75, 1, '19.0M', color='#7A777E', fontsize='small', ha='right')
ax6.text(2010, 23.9, '21.9%', fontsize='small')
ax6.text(2009.25, 1, '7.9M', color='#7A777E', fontsize='small')
ax6.text(2019, 11.5, '14.5%', fontsize='small', ha='right')
ax6.text(2019.75, 1, '5.1M', color='#7A777E', fontsize='small', ha='right')
ax7.text(2010, 17.5, '15.5%', fontsize='small')
ax7.text(2009.25, 1, '15.2M', color='#7A777E', fontsize='small')
ax7.text(2019, 7.8, '10.8%', fontsize='small', ha='right')
ax7.text(2019.75, 1, '10.8M', color='#7A777E', fontsize='small', ha='right')
ax8.text(2010, 12.7, '10.7%', fontsize='small')
ax8.text(2009.25, 1, '2.4M', color='#7A777E', fontsize='small')
ax8.text(2019, 7.3, '10.3%', fontsize='small', ha='right')
ax8.text(2019.75, 1, '3.1M', color='#7A777E', fontsize='small', ha='right')
#2010	9.9	27.4	12.0	26.5
#2019	7.3	18.8	7.1	15.7
ax9.text(2010, 11.9, '9.9%', fontsize='small')
ax9.text(2009.25, 1, '19.7M', color='#7A777E', fontsize='small')
ax9.text(2019, 4.3, '7.3%', fontsize='small', ha='right')
ax9.text(2019.75, 1, '14.3M', color='#7A777E', fontsize='small', ha='right')
ax10.text(2010, 29.4, '27.4%', fontsize='small')
ax10.text(2009.25, 1.5, '11.0M', color='#7A777E', fontsize='small')
ax10.text(2019, 15.8, '18.8%', fontsize='small', ha='right')
ax10.text(2019.75, 1.5, '8.2M', color='#7A777E', fontsize='small', ha='right')
ax11.text(2010, 14.0, '12.0%', fontsize='small')
ax11.text(2009.25, 1.5, '1.9M', color='#7A777E', fontsize='small')
ax11.text(2019, 4.1, '7.1%', fontsize='small', ha='right')
ax11.text(2019.75, 1.5, '1.5M', color='#7A777E', fontsize='small', ha='right')
ax12.text(2010, 28.5, '26.5%', fontsize='small')
ax12.text(2009.25, 1.5, '13.8M', color='#7A777E', fontsize='small')
ax12.text(2019, 12.7, '15.7%', fontsize='small', ha='right')
ax12.text(2019.75, 1.5, '9.6M', color='#7A777E', fontsize='small', ha='right')
# Set y limits
ax1.set_ylim(0, 35)
ax2.set_ylim(0, 35)
ax3.set_ylim(0, 35)
ax4.set_ylim(0, 35)
ax5.set_ylim(0, 35)
ax6.set_ylim(0, 35)
ax7.set_ylim(0, 35)
ax8.set_ylim(0, 35)
ax9.set_ylim(0, 35)
ax10.set_ylim(0, 35)
ax11.set_ylim(0, 35)
ax12.set_ylim(0, 35)
# Set x and y parameters
ax1.tick_params(colors='#7A777E')
ax2.tick_params(colors='#7A777E')
ax3.tick_params(colors='#7A777E')
ax4.tick_params(colors='#7A777E')
ax5.tick_params(colors='#7A777E')
ax6.tick_params(colors='#7A777E')
ax7.tick_params(colors='#7A777E')
ax8.tick_params(colors='#7A777E')
ax9.tick_params(colors='#7A777E')
ax10.tick_params(colors='#7A777E')
ax11.tick_params(colors='#7A777E')
ax12.tick_params(colors='#7A777E')
# Set x ticks
ax1.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticks([])
ax2.set_xticklabels([])
ax3.set_xticks([])
ax3.set_xticklabels([])
ax4.set_xticks([])
ax4.set_xticklabels([])
ax5.set_xticks([])
ax5.set_xticklabels([])
ax6.set_xticks([])
ax6.set_xticklabels([])
ax7.set_xticks([])
ax7.set_xticklabels([])
ax8.set_xticks([])
ax8.set_xticklabels([])
ax9.set_xticks([2010, 2019])
ax9.set_xticklabels(['2010', '2019'])
ax10.set_xticks([2010, 2019])
ax10.set_xticklabels(['2010', '2019'])
ax11.set_xticks([2010, 2019])
ax11.set_xticklabels(['2010', '2019'])
ax12.set_xticks([2010, 2019])
ax12.set_xticklabels(['2010', '2019'])
# Set y ticks
ax1.set_yticks([0, 10, 20, 30])
ax1.set_yticklabels(['0', '10', '20', '30'])
ax2.set_yticks([])
ax2.set_yticklabels([])
ax3.set_yticks([])
ax3.set_yticklabels([])
ax4.set_yticks([])
ax4.set_yticklabels([])
ax5.set_yticks([0, 10, 20, 30])
ax5.set_yticklabels(['0', '10', '20', '30'])
ax6.set_yticks([])
ax6.set_yticklabels([])
ax7.set_yticks([])
ax7.set_yticklabels([])
ax8.set_yticks([])
ax8.set_yticklabels([])
ax9.set_yticks([0, 10, 20, 30])
ax9.set_yticklabels(['0', '10', '20', '30'])
ax10.set_yticks([])
ax10.set_yticklabels([])
ax11.set_yticks([])
ax11.set_yticklabels([])
ax12.set_yticks([])
ax12.set_yticklabels([])
# Set all spines invisible
ax1.spines.right.set_visible(False)
ax1.spines.left.set_visible(False)
ax1.spines.top.set_visible(True)
ax1.spines.bottom.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
ax2.spines.top.set_visible(True)
ax2.spines.bottom.set_visible(False)
ax3.spines.right.set_visible(False)
ax3.spines.left.set_visible(False)
ax3.spines.top.set_visible(True)
ax3.spines.bottom.set_visible(False)
ax4.spines.right.set_visible(False)
ax4.spines.left.set_visible(False)
ax4.spines.top.set_visible(True)
ax4.spines.bottom.set_visible(False)
ax5.spines.right.set_visible(False)
ax5.spines.left.set_visible(False)
ax5.spines.top.set_visible(True)
ax5.spines.bottom.set_visible(False)
ax6.spines.right.set_visible(False)
ax6.spines.left.set_visible(False)
ax6.spines.top.set_visible(True)
ax6.spines.bottom.set_visible(False)
ax7.spines.right.set_visible(False)
ax7.spines.left.set_visible(False)
ax7.spines.top.set_visible(True)
ax7.spines.bottom.set_visible(False)
ax8.spines.right.set_visible(False)
ax8.spines.left.set_visible(False)
ax8.spines.top.set_visible(True)
ax8.spines.bottom.set_visible(False)
ax9.spines.right.set_visible(False)
ax9.spines.left.set_visible(False)
ax9.spines.top.set_visible(True)
ax9.spines.bottom.set_visible(False)
ax10.spines.right.set_visible(False)
ax10.spines.left.set_visible(False)
ax10.spines.top.set_visible(True)
ax10.spines.bottom.set_visible(False)
ax11.spines.right.set_visible(False)
ax11.spines.left.set_visible(False)
ax11.spines.top.set_visible(True)
ax11.spines.bottom.set_visible(False)
ax12.spines.right.set_visible(False)
ax12.spines.left.set_visible(False)
ax12.spines.top.set_visible(True)
ax12.spines.bottom.set_visible(False)
plt.show()
plot2_filename = 'week_8_matplotlib_lyons_2_25JUN21.png'
fig.savefig(plot2_filename, dpi=dpi, bbox_inches='tight')

#%% Create 'U.S. poverty rate by state' heatmap
fig, ax = plt.subplots(figsize = (8, 12))
# Plot seaborn heatmap
pr_hm = sns.heatmap(pr_st_percent_df, vmin=5, vmax=25, cmap='Reds', 
                    annot=True, fmt=".1f", annot_kws={"fontsize":8}, 
                    cbar=False, xticklabels=True, yticklabels=True)
# Set title
ax.set_title('U.S. poverty rate by state', 
             fontdict={'fontsize':16}, loc='left')
# Set x and y labels to ''
ax.set_xlabel('')
ax.set_ylabel('')
# Set x ticks to top position
# Reference:
# https://stackoverflow.com/questions/48470251/move-tick-marks-at-the-top-of-the-seaborn-plot?noredirect=1&lq=1
ax.xaxis.set_ticks_position('top')
# Set yticklabel colors
ax.get_yticklabels()[4].set_color('#386641')
ax.get_yticklabels()[4].set_fontweight('bold')
ax.get_yticklabels()[10].set_color('#386641')
ax.get_yticklabels()[10].set_fontweight('bold')
ax.get_yticklabels()[12].set_color('#386641')
ax.get_yticklabels()[12].set_fontweight('bold')
ax.get_yticklabels()[28].set_color('#386641')
ax.get_yticklabels()[28].set_fontweight('bold')
ax.get_yticklabels()[8].set_color('#BC4749')
ax.get_yticklabels()[8].set_fontweight('bold')
ax.get_yticklabels()[17].set_color('#BC4749')
ax.get_yticklabels()[17].set_fontweight('bold')
ax.get_yticklabels()[18].set_color('#BC4749')
ax.get_yticklabels()[18].set_fontweight('bold')
ax.get_yticklabels()[24].set_color('#BC4749')
ax.get_yticklabels()[24].set_fontweight('bold')
ax.get_yticklabels()[31].set_color('#BC4749')
ax.get_yticklabels()[31].set_fontweight('bold')
plt.show()
plot3_filename = 'week_8_matplotlib_lyons_3_25JUN21.png'
fig.savefig(plot3_filename, dpi=dpi, bbox_inches='tight')

#%% Create 'U.S. poverty rate by county' GeoPandas map
fig = plt.figure(figsize=(20, 24))
fig.suptitle('U.S. poverty rate by county', x=0.28, y=0.94, fontsize=36)
#[left, bottom, width, height]
ax1 = fig.add_axes([0.1, 0.5, 0.9, 0.4])
ax2 = fig.add_axes([0.12, 0.52, 0.12, 0.1])
ax3 = fig.add_axes([0.24, 0.52, 0.12, 0.1])
ax4 = fig.add_axes([0.1, 0.1, 0.9, 0.4])
ax5 = fig.add_axes([0.12, 0.12, 0.12, 0.1])
ax6 = fig.add_axes([0.24, 0.12, 0.12, 0.1])
cax = fig.add_axes([0.95, 0.52, 0.02, 0.1])
# Set vmin and vmax
vmin, vmax = 0, 51
# Plot the GeoPandas DataFrames
pr_cnty_geo_df_cont.plot(column='2010 Poverty %', 
                         cmap='Reds', ax=ax1,
                         vmin=vmin, vmax=vmax, 
                         legend=False, 
                         edgecolor='white', linewidth=1.0)
pr_cnty_geo_df_ak.plot(column='2010 Poverty %', 
                       cmap='Reds', ax=ax2,
                       vmin=vmin, vmax=vmax,             
                       edgecolor='white', linewidth=1.0)
pr_cnty_geo_df_hi.plot(column='2010 Poverty %', 
                       cmap='Reds', ax=ax3,
                       vmin=vmin, vmax=vmax, 
                       edgecolor='white', linewidth=1.0)
pr_cnty_geo_df_cont.plot(column='2019 Poverty %', 
                         cmap='Reds', ax=ax4,
                         vmin=vmin, vmax=vmax, 
                         legend=False, 
                         edgecolor='white', linewidth=1.0)
pr_cnty_geo_df_ak.plot(column='2019 Poverty %', 
                       cmap='Reds', ax=ax5, 
                       vmin=vmin, vmax=vmax, 
                       edgecolor='white', linewidth=1.0)
pr_cnty_geo_df_hi.plot(column='2019 Poverty %', 
                       cmap='Reds', ax=ax6, 
                       vmin=vmin, vmax=vmax, 
                       edgecolor='white', linewidth=1.0)
# Create colorbar
# Reference:
# https://gis.stackexchange.com/questions/343881/using-vmin-and-vmax-does-not-change-anything-for-my-plots
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), 
                           cmap='Reds')
sm.set_array([])
fig.colorbar(sm, cax=cax)
cax.tick_params(labelsize='x-large')
# Remove unnecessary space for AK
# Reference:
# https://towardsdatascience.com/how-to-split-shapefiles-e8a8ac494189
ax2.set_xlim(75, 50)
ax2.set_xlim(-175, -125)
ax5.set_xlim(75, 50)
ax5.set_xlim(-175, -125)
# Set titles
ax1.set_title('2010', fontsize='24', loc='center')
# ax2.set_title('Alaska', fontsize='20', loc='left')
# ax3.set_title('Hawaii', fontsize='20', loc='left')
ax4.set_title('2019', fontsize='24', loc='center')
# ax5.set_title('Alaska', fontsize='20', loc='left')
# ax6.set_title('Hawaii', fontsize='20', loc='left')
# Set axis off
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
ax5.set_axis_off()
ax6.set_axis_off()
plt.show()
plot4_filename = 'week_8_matplotlib_lyons_4_25JUN21.png'
fig.savefig(plot4_filename, dpi=dpi, bbox_inches='tight')

#%% Create 'Poverty, unemployment, and median income' visual
# Create figure and subplots
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pr_oa_df.index, pr_oa_df['Poverty %'], color='black', linewidth=3)
ax.scatter(2015, 13.5, color='#D62828')
ax1 = ax.twinx()
ax1.plot(unemp_rt_df.index, unemp_rt_df['Unemployment rate'], 
         color='#A26C02', linewidth=3)
ax2 = ax.twinx()
ax2.plot(median_inc_df.index, median_inc_df['Median income'], 
         color='#606C38', linewidth=3)
ax2.scatter(2015, 56516, color='#D62828')
# Add axvline
ax.axvline(2015, ymin=0.1, ymax=0.78, color='#343A40', ls='dotted')
ax.axvline(2015, ymin=0.95, ymax=1.0, color='#343A40', ls='dotted')
# Set titles
ax1.set_title('U.S. poverty rate, median income, and unemployment rate', 
              loc='left', 
              weight='semibold')
# Create ax text
#2010	15.1
#2019	10.5
ax.text(2009.75, 15.1, '15.1%', color='black', fontsize='large', ha='right')
ax.text(2019.25, 10.5, '10.5%, Poverty rate', color='black', fontsize='large')
ax.text(2015, 14.2, '13.5%, Poverty rate', color='#343A40', fontsize='medium', 
         ha='left')
ax.text(2015.1, 11.0, 'Median househould income', color='#343A40', 
         fontsize='medium', ha='left')
ax.text(2015.1, 10.0, 'increased 5.2%', color='#343A40', 
         fontsize='medium', ha='left')
#2010	9.6
#2019	3.7
ax1.text(2009.75, 9.6, '9.6%', color='#A26C02', 
         fontsize='large', ha='right')
ax1.text(2019.25, 3.7, '3.7%, Unemployment rate', color='#A26C02', 
         fontsize='large')
#2010	49276
#2019	68703
ax2.text(2009.75, 49276, '$49,276', color='#606C38', 
         fontsize='large', ha='right')
ax2.text(2019.25, 68703, '$68,703, Median income', color='#606C38', 
         fontsize='large')
# Set x and y parameters
ax.tick_params(colors='#7A777E')
ax1.tick_params(colors='#7A777E')
ax2.tick_params(colors='#7A777E')
# Set y limits
ax.set_ylim(0, 16)
ax1.set_ylim(0, 16)
ax2.set_ylim(0, 70000)
# Set x ticks
ax.set_xticks([2010 , 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
ax.set_xticklabels(['2010', '2011', '2012', '2013', '2014', '2015', '2016', 
                    '2017', '2018', '2019'])
# Set y ticks
ax.set_yticks([])
ax.set_yticklabels([])
ax1.set_yticks([])
ax1.set_yticklabels([])
ax2.set_yticks([])
ax2.set_yticklabels([])
# Set all spines invisible
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.bottom.set_visible(False)
ax1.spines.right.set_visible(False)
ax1.spines.left.set_visible(False)
ax1.spines.top.set_visible(False)
ax1.spines.bottom.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
# Show plot and save figure
plt.show()
plot5_filename = 'week_8_matplotlib_lyons_5_25JUN21.png'
fig.savefig(plot5_filename, dpi=dpi, bbox_inches='tight')

## THANK YOU!