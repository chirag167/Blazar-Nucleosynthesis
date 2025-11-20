from scipy.interpolate import interp1d
import pandas as pd 

def interpolate_data(filepath,x_interp,i=0,j=1,sep='\s+',comments=None,header=None):

	'''Function to perform 1D interpolation between data points.

	INPUTS: filepath = the name of the file in which the data needs to be interpolated,
	x_interp = the x value at which the interpolation needs to happen.
	i,j = the column name or number at which the x and y data are stored. Default is 0,1
	sep = delimiter in the file.
	comments = if comments start with a symbol (like #) it will be ignored.
	header = if the file has a header, specify the row number.

	RETURNS: the interpolated value of the dataset in filepath at x_interp. '''

	df = pd.read_csv(filepath,sep=sep,comment=comments,header=header)
	print(df.head())

	x = df[i]

	y = df[j]

	y_interp = interp1d(x,y)

	interp_vals = y_interp(x_interp)

	return interp_vals

# testing the function

# hdp_interpolate = interpolate_data('../data/NACRE_2/2hdp3h.dat',0.33,comments='#')

# print(hdp_interpolate)
