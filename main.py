
from yahoo_finance import Share

def main():

	tesla = Share('TSLA')

	print tesla.get_open() # open price today



if __name__ == '__main__':
	main()