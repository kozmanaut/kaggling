#!/usr/bin/env python

"""
Using the US National Park Biodiversity database to run some stats on the parks - which ones contain the most amount of imperiled species?
database available at: https://www.kaggle.com/nationalparkservice/park-biodiversity
data downloaded on 21 January 2017
"""

from __future__ import division
import pandas as pd
import matplotlib.pyplot  as plt


# for prettier plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 10)


# Load the dataset
species = pd.read_csv('species.csv', usecols=range(1,13))
species['Conservation Status'].fillna('Not Threatened', inplace=True)	# change NaN to 'Not Threatened' in the Conservation Status column
species = species[species['Conservation Status'].str.contains("Breeder|Resident|Migratory") == False]   # drop the rows where conservation category is either "Breeder", "Resident" or "Migratory"
parks = pd.read_csv('parks.csv')

# how many unique species there are
len(species['Scientific Name'].unique())
# 46022

# Count the occurrence of each conservation status across all parks
status = species['Conservation Status'].value_counts()
status

# Break down of Conservation Status of each Organism category
org = species[['Category', 'Conservation Status']]
org_count = org.groupby(['Category', 'Conservation Status']).size()
org_count.unstack(level=1).plot(kind='bar', stacked=True)
plt.legend(loc='upper left')
plt.xlabel('Organism class')
plt.ylabel('Count')
plt.title('Breakdown of each conservation category across all organism classes')
plt.tight_layout()
plt.show()
#get rid of 'not threatened' column
org_count_not_safe= org_count.unstack(level=1)
org_count_not_safe = org_count_not_safe.ix[:, org_count_not_safe.columns != 'Not Threatened']
org_count_not_safe.plot.bar(stacked=True)
plt.xlabel('Organism class')
plt.ylabel('Count')
plt.title('\'Not Threatened\' conservation category excluded')
plt.tight_layout()
plt.show()
#plt.savefig('Organism_Conservation_Status.png')


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
		print "Park with the maximum number of" , i.lower() ,  "species:" ,  "%s (%d)" % (df.loc[df['Count'].where(df['Conservation Status'] == i).idxmax()][0], df.loc[df['Count'].where(df['Conservation Status'] == i).idxmax()][2])
		print "Park with the minimum number of" , i.lower() ,  "species:" ,  "%s (%d)" % (df.loc[df['Count'].where(df['Conservation Status'] == i).idxmin()][0], df.loc[df['Count'].where(df['Conservation Status'] == i).idxmin()][2])
	print('')
max_min_parks(props_df)

# PARK with:
# Maximum number of endangered species: Hawaii Volcanoes National Park (44)
# Maximum number of in recovery species: Redwood National Park (7)
# Maximum number of species of concern : Death Valley National Park (177)
# Maximum number of threatened species: Death Valley National Park (16)
# Maximum number of proposed endangered species: Haleakala National Park (11)



# Count of each conservation category per acre for each NP - i.e. per acre of land, which park harbors the most e.g. endangered species
#1. Create a dictionary with 'Park' : 'Acres'
park_dict = dict(zip(parks['Park Name'], parks['Acres']))
#2. Create a function that divides each row's conservation category count by the park's area
def divide_count(row):
	return row['Count']/(park_dict[row['Park Name']])
#3. Create a new column with the count per acre measure
props_df['CountPerAcre'] = props_df.apply(divide_count, axis=1)
#4. Find which parks have the max/min of each conservation category per acre of land
def per_acre_max_min_parks(df):
	""" Function that finds the min/max for each Conservation category per acre of park"""
	for i in df['Conservation Status'].unique():
		print "Park with the per acre maximum number of" , i.lower() ,  "species:" ,  "%s (%f)" % (df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmax()][0], df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmax()][3])
		print "Park with the per acre minimum number of" , i.lower() ,  "species:" ,  "%s (%f)" % (df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmin()][0], df.loc[df['CountPerAcre'].where(df['Conservation Status'] == i).idxmin()][3])
	print('')
per_acre_max_min_parks(props_df)

# Park with the per acre maximum number of endangered species: Haleakala National Park (0.001375)
# Park with the per acre maximum number of in recovery species: Hot Springs National Park (0.000180)
# Park with the per acre maximum number of species of concern: Hot Springs National Park (0.010991)
# Park with the per acre maximum number of threatened species: Hot Springs National Park (0.000360)
# Park with the per acre maximum number of proposed endangered species: Haleakala National Park (0.000378)




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
		print "Park with maximum proportion of" , i.lower() ,  "species:" ,  "%s (%f)" % (df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmax()][0], df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmax()][4])
		print "Park with minimum proportion of" , i.lower() ,  "species:" ,  "%s (%f)" % (df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmin()][0], df.loc[df['ProportionalCount'].where(df['Conservation Status'] == i).idxmin()][4])
	print('')
proportional_min_max_parks(props_df)

# Park with maximum proportion of endangered species: Haleakala National Park (0.015504)
# Park with maximum proportion of in recovery species: Channel Islands National Park (0.002653)
# Park with maximum proportion of species of concern species: Petrified Forest National Park (0.086753)
# Park with maximum proportion of threatened species: Dry Tortugas National Park (0.007075)
# Park with maximum proportion of proposed endangered species: Haleakala National Park (0.004264)


###
# STILL TO DO:
