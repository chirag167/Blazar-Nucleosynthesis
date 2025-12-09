import numpy as np 


def calc_reaction_rate(Z_t,A_t,channel,T9):

	# Malaney and Fowler, 1989 (https://ui.adsabs.harvard.edu/abs/1989ApJ...345L...5M/abstract)
	if channel == 'ng':

		if Z_t == 3 and A_t == 8:

			rate = 4.294*10**4 + 6.047*10**4 * T9**(-3/2) * np.exp(-2.866/T9) # cm3/s/mol

		elif Z_t == 3 and A_t == 6:

			rate = 5.10*10**3 # cm3/s/mol

		elif Z_t == 5 and A_t == 11:

			rate == 7.29*10**2 + 2.40*10**3 * T9**(-3/2) *np.exp(-0.233/T9) # cm3/s/mol

	elif channel == 'dn':

		if Z_t == 3 and A_t == 6:

			rate = 1.48*10**12 * T9**(-2/3) * np.exp(-10.135/T9**(1/3)) # cm3/s/mol

		if Z_t == 6 and A_t == 14:

			rate = 4.27*10**13 * T9**(-2/3) * np.exp(-16.939/T9**(1/3))

	elif channel == 'dp':

		if Z_t == 3 and A_t == 6:

			rate = 1.48*10**12 * T9**(-2/3) * np.exp(-10.135/T9**(1/3)) # cm3/s/mol

		elif Z_t == 3 and A_t == 7:

			rate = 8.31*10**8 * T9**(-3/2) * np.exp(-6.998/T9) # cm3/s/mol

	elif channel == 'tn':

		if Z_t == 3 and A_t == 7:

			rate = 1.46*10**(11) * T9**(-2/3) *np.exp(-11.333/T9**(1/3))


	elif channel == 'pg':

		# Thomas et al., 1998 (https://arxiv.org/pdf/astro-ph/9206002)
		if Z_t == 2 and A_t == 3:

			if T9 <= 3:
				rate = 8.78e-13 * T9**(-2/3)* np.exp(-6.141/T9**(1/3))

			else:

				rate = 5.97e-15

		if Z_t == 6 and A_t == 14: #https://www.annualreviews.org/docserver/fulltext/astro/13/1/annurev.aa.13.090175.000441.pdf?expires=1765316320&id=id&accname=ar-366613&checksum=2193E735B399952E1E5CF7024E235C16

			rate = 6.80E+06/T9**(2/3)*np.exp(-13.741/T9**(1/3)-(T9/5.721)**2) * (1.+0.030*T9**(1/3)+0.503*T9**(2/3)+0.107*T9+0.213*T9**(4/3)+0.115*T9**(5/3)) 
			+ 5.36E+03/T9**(3/2)*np.exp(-3.811/T9) + 9.82E+4/T9**(1/3)*np.exp(-4.739/T9)

	elif channel == 'np':

		if Z_t == 7 and A_t == 14:

			rate = 2.40*10**5 * (1 + 0.361*T9**0.5 +0.502*T9) + 3.93*10**7 *np.exp(-4.079/T9)

	return rate