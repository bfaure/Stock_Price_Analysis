
# for stock price data...
#from yahoo_finance import Share

# for plotting features...
import matplotlib.pyplot as plt

# for machine learning stuff...
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

# alternate stock price data source...
#from googlefinance import getQuotes

# for handling googlefinance returns
#import json

# another stock data source
import quandl

# for array manipulation (prep for keras)
import numpy as np

# for writing to terminal
import sys


#######################################################

# shouldn't really have this on a public repo but fuck it
def set_api_key(key="X6zvDM1FEEkWU7H1w___"):
	# register the key (free acct)
	quandl.ApiConfig.api_key = key

# fetches a data table from quandl (set to Visa, inc. by default)
def get_data(quandl_code="EOD/V"):
	# fetch some sample data (Visa, Inc.)
	data = quandl.get("EOD/V")
	return data


# all colums of the data tables returned from quandl, not including
# the first column which is the date (string formatted such as 
# '2008-03-19')
valid_table_columns = [	"High","Low","Open","Close","Volume",
						"Dividend","Split","Adj_Open","Adj_High",
						"Adj_Low","Adj_Close","Adj_Volume"]

# returns a single column of the provided data table
def get_table_column(data,column="High"):
	if column not in valid_table_columns: print "You are retarded"
	col = data[column]
	return col 

# returns the named dates of provided data in list format
def get_table_dates(data):
	axes = data.axes 
	dates = axes[0]
	return dates

def normalize(column):
	n_list = []
	col_size = len(column)
	last_item = None
	for i in range(col_size):
		current_item = column[i]
		if last_item!=None:
			perc = (current_item-last_item) / last_item
			perc *= 100.0
			n_list.append(perc)
		last_item = current_item
	return n_list 


# plots the input data
def plot(xaxis,ylabel="Data"):
	plt.plot(xaxis)
	plt.ylabel = ylabel
	plt.show()

# takes in a single column (for example the price), e.g. [0,1,2,3]
# and returns a matrix. for example, with a look_back set to 1 we would
# return: [0,1], [1,2], [2,3]
# the (look_back+1) is the number of elements in each row of the returned matrix
def prepare_data(data,look_back):

	print "preparing data of length "+str(len(data))

	matrix = []
	index=0

	while True:

		if index+look_back>=len(data): break

		row=[]
		for i in range(index,index+look_back+1):
			row.append(data[i])

		index+=look_back
		matrix.append(row)

	print "created matrix with "+str(len(matrix))+" rows"
	print "each containing "+str(look_back+1)+" items"
	return matrix

# splits the data (in matrix form, as output from prepare_data()) into 
# the specified train/test split, with 0.7 as a default. Also converts to
# numpy arrays (required by keras) and splits into X and y components. 
# the final output is train_X, train_y, test_X, test_y.
def train_test_split(data,look_back,split=0.7):

	print "performing train/test split, data length is "+str(len(data))

	print "sample data row:"
	print data[0]

	y = [] # outputs
	X = [] # inputs

	train_length = int(float(len(data))*split)
	test_length = len(data)-train_length

	print "train length: "+str(train_length)+", test: "+str(test_length)

	for d in data:
		y.append(d[-1])
		X.append(d[:-1])
	
	print "sample, X[0]:"
	print X[0]
	print "sample, y[0]:"
	print y[0]

	train_X = X[:train_length]
	train_y = y[:train_length]

	test_X = X[train_length:]
	test_y = y[train_length:]

	print "finished train/test split"
	print "converting to numpy arrays..."

	train_X = np.array(train_X)
	train_y = np.array(train_y)
	test_X = np.array(test_X)
	test_y = np.array(test_y)

	'''
	print "shapes..."
	print train_X.shape
	print train_y.shape
	print test_X.shape
	print test_y.shape
	'''

	print "converted successfully!"
	print "reshaping"

	# inputs should be of the form [samples, timesteps, features]
	# aka [samples, look_back, after look_back]
	train_X = np.reshape(train_X,(train_X.shape[0],1,train_X.shape[1]))
	test_X = np.reshape(test_X,(test_X.shape[0],1,test_X.shape[1]))

	return train_X,train_y,test_X,test_y

# builds a new model and fits it to the provided data
def fit_model(train_X,train_y,test_X,test_y,look_back):
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(train_X, train_y, epochs=30, batch_size=1, verbose=1)
	return model

# saves the model passed as a parameter, to the name fname
def save_model(model,fname="model.h5"):
	model.save(fname)

# loads the model from specified file name
def load_our_model(fname="model.h5"):
	sys.stdout.write("Loading model... ")
	model = load_model(fname)
	sys.stdout.write("success\n")
	return model 

# predicts future values using the provided model and last element of the dataset.
# data should be a single vector (single column), look_back is used to index into
# the data column, predictions are used as new input to generate new predictions.
# n is the number of data points to predict
def predict_future(model,data,look_back,n=50):

	i=0
	while i<n:

		# we only need the last look_back number of data points for the first prediction
		inputs = data[len(data)-look_back:]

		# putting into another list to emulate training inputs
		inputs = [inputs]

		# convert to numpy array
		inputs = np.array(inputs)

		# reshape the numpy array to fit model input
		inputs = np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1]))

		# get a prediction
		prediction = model.predict(inputs)

		# add the prediction to the data list
		data.append(prediction)

		# increment prediction count
		i+=1

	return data

# saves the data to a file
def save_data(data,fname="stock_data.tsv"):
	f = open(fname,"w")

	for v in valid_table_columns:
		f.write(v+"\t")
	f.write("\n\n")

	for i in range(len(data)):
		for v in valid_table_columns:
			f.write(str(data[v][i])+"\t")
		f.write("\n")

# loads the data from file assuming save_data() formatting
def load_data(fname="stock_data.tsv"):
	f = open(fname,"r")
	data = []
	col_names = []

	line_index=0
	for line in f:
		line_index+=1

		if line_index==1:
			line = line.strip().split("\t")
			for item in line:
				if len(item)!=0: col_names.append(item)

		if line_index>=3:
			line = line.strip().split("\t")
			row = []
			for item in line:
				if len(item)!=0: row.append(float(item))
			data.append(row)
	return col_names,data 

# loads in a single column from the specified file
def load_spec_data(column="Adj_High",fname="stock_data.tsv"):
	col_names,data = load_data(fname)
	spec_data=[]
	col_idx = col_names.index(column)
	for row in data:
		spec_data.append(row[col_idx])
	return spec_data

def main():
	set_api_key()

	"""
	#data = get_data()
	#highs = get_table_column(data,"Adj_High")
	#highs_normalized = normalize(highs)
	#plot(highs_normalized)
	#data = get_data()
	#save_data(data)
	"""

	# whether or not to retrain the model, if not, loading from disk
	retrain = False

	# number of inputs (prices) used to predict output (price)
	look_back = 2

	# load in a single column (adjusted high)
	data = load_spec_data()

	# prepare the data for splitting
	prepped = prepare_data(data,look_back=look_back)
	
	# split the data into training and testing
	train_X,train_y,test_X,test_y = train_test_split(prepped,look_back=look_back)
	
	if retrain:

		# build the model and fit to the data
		model = fit_model(train_X,train_y,test_X,test_y,look_back=look_back)

		# save the model to a file (so we can load later instead of retraining)
		save_model(model)

	else:

		# load the model from disk
		model = load_our_model()

	# predict 50 prices in the future
	predictions = predict_future(model,data,look_back,n=1000)

	plot(predictions)

if __name__ == '__main__':
	main()