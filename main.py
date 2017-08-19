
# for stock price data...
#from yahoo_finance import Share

# for plotting features...
import matplotlib.pyplot as plt

# for machine learning stuff...
#import keras

# alternate stock price data source...
#from googlefinance import getQuotes

# for handling googlefinance returns
#import json

# another stock data source
import quandl

#######################################################

'''
# plots the symbol price over the specified date range
# Yahoo finance api seems to be discontinued, this function
# no longer works, we should check out other stock price apis/libraries
# google I know has one
def plot_hist_stock_price(symbol='TSLA',start='2014-04-25',end='2014-04-29'):

	ticker = Share(symbol) # initialize share

	data = ticker.get_historical(start,end)

	opens 	= [] # open prices, not used
	closes 	= [] # close prices, not used
	highs 	= [] # high prices
	lows 	= [] # low prices, not used
	volumes = [] # volumes, not used

	for item in data:
		highs.append(item['High'])

	print highs
'''

def main():

	# free account key (quandl.com)
	api_key = "X6zvDM1FEEkWU7H1w___"

	# register the key
	quandl.ApiConfig.api_key = api_key 

	# fetch some sample data (Visa, Inc.)
	data = quandl.get("EOD/V")

	# the data object is a 2373 row, 12 column table

	# columns are:
	# Date, Open, High, Low, Close, Volume, Dividend, Split, \
	# Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume

	# each row is for a certain date, dates are structured like:
	# '2008-03-19'

	# putting the data into separate variables...

	dates 		= []
	opens 		= data['High']
	highs 		= data['Low']
	lows 		= data['Low']
	closes 		= data['Close']
	volumes 	= data['Volume']
	dividends 	= data['Dividend']
	splits 		= data['Split']
	adj_opens 	= data['Adj_Open']
	adj_highs 	= data['Adj_High']
	adj_lows 	= data['Adj_Low']
	adj_closes 	= data['Adj_Close']
	adj_volumes = data['Adj_Volume']

	# now each of the above variables (opens...adj_volumes) can be
	# addresses as: opens[0] or opens['2008-03-19'], fill in your own 
	# index or date

	# putting the opens data into a single list so we can plot...

	plot_opens = []
	for i in range(len(opens)):
		plot_opens.append(opens[i])

	plt.plot(plot_opens)
	plt.ylabel("Visa Open Values")
	plt.show()



if __name__ == '__main__':
	main()