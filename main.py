
# for stock price data...
from yahoo_finance import Share

# for plotting features...
import matplotlib.pyplot as plt

# for machine learning stuff...
#import keras

#######################################################


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

def main():

	plot_hist_stock_price()



if __name__ == '__main__':
	main()