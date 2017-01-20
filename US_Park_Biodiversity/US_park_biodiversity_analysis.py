#!/usr/bin/env python

"""
Using the US National Park Biodiversity database to run some analyses
database available at: https://www.kaggle.com/nationalparkservice/park-biodiversity
data downloaded on 20 January 2017
"""

import pandas as pd
import matplotlib.pyplot  as plt
from __future__ import division

# for prettier plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 10)


# Load the dataset
species = pd.read_csv('species.csv')
species['Conservation Status'].fillna('Not Threatened', inplace=True)	# change NaN to 'Not Threatened' in the Conservation Status column
parks = pd.read_csv('parks.csv')

# how many unique species there are
len(species['Scientific Name'].unique())
# 33273

# Count the occurrence of each conservation status across all parks
status = species['Conservation Status'].value_counts()
status

# Break down of Conservation Status of each Organism category
org = species[['Category', 'Conservation Status']]
org_count = org.groupby(['Category', 'Conservation Status']).size()
org_count.unstack(level=1).plot(kind='bar', stacked=True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()
#get rid of 'not threatened' column
org_count_not_safe= org_count.unstack(level=1)
org_count_not_safe = org_count_not_safe.ix[:, org_count_not_safe.columns != 'Not Threatened']
org_count_not_safe.plot.bar(stacked=True)
plt.xlabel('Organism Category')
plt.ylabel('Number of species')
plt.tight_layout()
#plt.show()
plt.savefig('Organism_Conservation_Status.png')


#######
# Find out more about each park individually
#######

# Calculate number of each conservation status category for each park
props = species[['Park Name', 'Conservation Status']].groupby(['Park Name', 'Conservation Status']).size()


"""
# Plot an example park (Hawaii Volcanoes National Park)
HVNP = props['Hawaii Volcanoes National Park']
HVNP.plot(kind='barh')
plt.tight_layout()
#plt.show()
plt.savefig('Hawaii_Volcanoes_NP_Cons_Status.png')

"""


# Find the park with the min/max for each Conservation category
props_df = props.to_frame().reset_index()
props_df.columns = ['Park Name', 'Conservation Status', 'Count']

def max_min_parks(df):
	""" Function that finds the min/max for each Conservation category"""
	for i in df['Conservation Status'].unique():
		print "Maximum number of" , i.lower() ,  ":" ,  "%s (%d)" % (df.loc[df['Count'].where(df['Conservation Status'] == i).idxmax()][0], df.loc[df['Count'].where(df['Conservation Status'] == i).idxmax()][2])
		print "Minimum number of" , i.lower() ,  ":" ,  "%s (%d)" % (df.loc[df['Count'].where(df['Conservation Status'] == i).idxmin()][0], df.loc[df['Count'].where(df['Conservation Status'] == i).idxmin()][2])

max_min_parks(props_df)



# Count of each conservation category per acre for each NP - i.e. per acre of land, which park harbors the most e.g. endangered species
park_dict = dict(zip(parks['Park Name'], parks['Acres']))

def divide_count(row):
	return row['Count']/(park_dict[row['Park Name']])

props_df['CountPerAcre'] = props_df.apply(divide_count, axis=1)

def per_acre_max_min_parks(df):
	""" Function that finds the min/max for each Conservation category per acre of park"""
	for i in df['Conservation Status'].unique():
		print "Per acre of park maximum number of" , i.lower() ,  ":" ,  "%s (%f)" % (df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmax()][0], df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmax()][3])
		print "Per acre of park minimum number of" , i.lower() ,  ":" ,  "%s (%f)" % (df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmin()][0], df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmin()][3])

per_acre_max_min_parks(props_df)


# Count the proportion of each conservation category for each park - i.e. find the park with the highest proportion of endangered species
#1. Sum up all conservation classes per park
park_sums = props_df.groupby(['Park Name']).agg('sum').reset_index()
# Create a dictionary with 'Park' : 'Total count'
park_sums_dict = dict(zip(park_sums['Park Name'], park_sums['Count']))
# Create function that divides each conservation category count by the total count
def divide_total(row):
	return row['Count']/(park_sums_dict[row['Park Name']])
# Create new column with proportional count of conservation category
props_df['ProportionalCount'] = props_df.apply(divide_total, axis=1)
# FInd out which parks have the maximum proportion of each conservation category
def proportional_min_max_parks(df):
	""" Function that finds the park with the highest / lowest proportion of each conservation category"""
	for i in df['Conservation Status'].unique():
		print "Park with maximum proportion of" , i.lower() ,  ":" ,  "%s (%f)" % (df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmax()][0], df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmax()][4])
		print "Park with minimum proportion of" , i.lower() ,  ":" ,  "%s (%f)" % (df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmin()][0], df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmin()][4])

proportional_min_max_parks(props_df)

###