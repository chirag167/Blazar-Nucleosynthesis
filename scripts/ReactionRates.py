import numpy as np 


def calc_reaction_rate(Z_t,A_t,channel,T9):

	# Malaney and Fowler, 1989 (https://ui.adsabs.harvard.edu/abs/1989ApJ...345L...5M/abstract)
	if channel == 'ng':

		if Z_t == 3 and A_t == 8 and channel == 'ng':

			rate = 4.294*10**4 + 6.047*10**4 * T9**(-3/2) * np.exp(-2.866/T9)

		elif Z_t == 3 and A_t == 6:

			rate = 5.10*10**3

		elif Z_t == 5 and A_t == 11 and channel == 'ng':

			rate == 7.29*10**2 + 2.40*10**3 * T9**(-3/2) *np.exp(-0.233/T9)

	elif channel == 'pg':

		# Thomas et al., 1998 (https://arxiv.org/pdf/astro-ph/9206002)
		if T9 <= 3:

			rate = 8.78e-13 * T9**(-2/3)* np.exp(-6.141/T9**(1/3))

		else:

			rate = 5.97e-15

	return rate