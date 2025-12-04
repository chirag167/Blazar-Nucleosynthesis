import numpy as np 
import pandas as pd

def read_nacre_1_rates(T9):

	df = pd.read_csv('../data/NACRE_1_coeff.csv',header=0,comment='#')

	a0,a1,a2,a3,a4,a5,a6 = df['a0'],df['a1'],df['a2'],df['a3'],df['a4'],df['a5'],df['a6']

	term_2 = a1/T9 + a2/T9**(1/3) + a3*T9**(1/3) + a4*T9 + a5*T9**(5/3)

	term_3 = a6*np.log(T9)

	rate = np.exp(a0 + term_2 + term_3)

	return rate

