
# for stock price data...
#from yahoo_finance import Share

# for plotting features...
import matplotlib.pyplot as plt

'''
# for machine learning stuff...
from keras.layer.core import Dense, Avctivation, Dropout
from keras.layer.recurrent import LSTM
from keras.models import Sequential
'''

# alternate stock price data source...
#from googlefinance import getQuotes

# for handling googlefinance returns
#import json

# another stock data source
import quandl

# for array manipulation (prep for keras)
import numpy as np

#######################################################

'''
def make_model():

	model = Sequential()
	model.add(LSTM(
		input_dim=1,
		output_dim=50,
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(
		100,
		return_sequences=False
		))
	model.add(Dropout(0.2))

	model.add(Dense(
		output_dim=1))
	model.(Activation('linear'))

	
	model.compile(loss='mse',optimizer='rmsprop')

	return model 


def train_model(model,X_train,y_train):
	model.fit(
		X_train,
		y_train,
		batch_size=512,
		nb_epoch=1,
		validation_split=0.05)
'''


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
def prepare_data(data,look_back=1):

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
def train_test_split(data,split=0.7):

	print "performing train/test split, data length is "+str(len(data))

	y = [] # outputs
	X = [] # inputs

	train_length = int(float(len(data))*split)
	test_length = len(data)-train_length

	print "train length: "+str(train_length)+", test: "+str(test_length)

	for d in data:
		y.append(d[-1])
		X.append(d[:len(d)-2])
	
	train_X = X[:train_length]
	train_y = y[:train_length]

	test_X = X[train_length:]
	test_y = y[train_length:]

	print "finished train/test split"

	return train_X,train_y,test_X,test_y


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

	#data = get_data()

	#highs = get_table_column(data,"Adj_High")

	#highs_normalized = normalize(highs)

	#plot(highs_normalized)

	#data = get_data()
	#save_data(data)

	data = load_spec_data()
	
	prepped = prepare_data(data)

	print prepped



if __name__ == '__main__':
	main()