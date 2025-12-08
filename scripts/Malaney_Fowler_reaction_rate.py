import numpy as np 


def calc_reaction_rate(Z_t,A_t,channel,T9):

	if Z_t == 3 and A_t == 8 and channel == 'ng':

		rate = 4.294*10**4 + 6.047*10**4 * T9**(-3/2) * np.exp(-2.866/T9)

	elif Z_t == 3 and A_t == 6 and channel == 'ng':

		rate = 5.10*10**3

	elif Z_t == 5 and A_t == 11 and channel == 'ng':

		rate == 7.29*10**2 + 2.40*10**3 * T9**(-3/2) *np.exp(-0.233/T9)

	return rate