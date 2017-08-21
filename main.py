
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
valid_table_columes = [	"High","Low","Open","Close","Volume",
						"Dividend","Split","Adj_Open","Adj_High",
						"Adj_Low","Adj_Close","Adj_Volume"]

# returns a single column of the provided data table
def get_table_column(data,column="High"):
	if column not in valid_table_columes: print "You are retarded"
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


def main():
	set_api_key()

	data = get_data()

	highs = get_table_column(data,"Adj_High")

	highs_normalized = normalize(highs)

	#$plot(highs)
	plot(highs_normalized)


if __name__ == '__main__':
	main()