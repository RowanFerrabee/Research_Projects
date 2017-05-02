from yahoo_finance import Share
import numpy

################################################
#Helper Functions Designed to Abstract
#Interface with Yahoo Finance Application
#
#Author: Mitchell Catoen
################################################

#Returns financial data as series
def process_stock_data(date_1, date_2, stock_name):

	nasdaq = Share('^IXIC')
	yahoo = Share(stock_name)
	nasdaq.refresh()
	yahoo.refresh()
	index_data = nasdaq.get_historical(date_1, date_2)
	data = yahoo.get_historical(date_1, date_2)

	# 0 Volume
	# 1	Symbol
	# 2 Adj Close
	# 3 High
	# 4 Low
	# 5 Date
	# 6 Close
	# 7 Open

	close_prices = []

	#Set up as two dimensional data
	for day in data:
		close_price = []
		close_price.append(float(day.values()[7]))
		close_prices.insert(0, close_price)

	return close_prices

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back * 2, 0])
		print a, " and ", dataset[i + look_back * 2]
	return numpy.array(dataX), numpy.array(dataY)


