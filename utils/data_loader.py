import numpy as np 
import pandas as pd


def load_data_file(filepath):
	'''Function to load data required for Famiano calculations.
	Input: the relative file path.
	Output: A .csv formatted file'''

	df = pd.read_csv(filepath,sep='\s+',header=0)

	#print(df.head(5))

	return df


# hdp_rxn = load_data_file('../data/NACRE_2/2hdp3h.dat')

# print(hdp_rxn.head(10))