

################################################
#	Main Application File
#
#	Author: Mitchell Catoen
################################################

from helpers import *
import matplotlib.pyplot as plt
import svmpy
import numpy as np
import dill
import pickle
from yahoo_finance import Share


movo_days = 25
pred_days = 25 #Note, an extra day is added
stock_names = ["YHOO", "BBRY", "GOOG", "AAPL", "MSFT"]



################################################
#
#	Set New Dates and reprocess data
#
################################################

date_1 = '2014-01-17'
date_2 = '2017-03-17'

for i in range (0, len(stock_names)):
	predictor = pickle.load(open("stock_models/" + stock_names[i] + ".bin", "rb"))

	profit = 0

	vo_stock, mo_stock, me_stock, vo_index, mo_index, me_index = process_stock_data(date_1, date_2, movo_days, stock_names[i])
	inc_dec = make_dataset(me_stock, pred_days)


	################################################
	#
	#	Plot Graphs
	#
	################################################

	plot_figure_triple(me_stock, mo_stock, vo_stock, stock_names[i] + "_" + date_1 + "_" + date_2, i)
	#plot_figure_double(me_stock, inc_dec, "Yahoo Stock Over Time", 3)

	#run_test(pred_days, predictor, inc_dec, mo_stock, vo_stock, mo_index, vo_index, stock_names[i])
	buys = 0
	hit = 0
	spent_money = 0
	for j in range (0, len(inc_dec) - 1):
		prediction = predictor.predict([mo_stock[j], vo_stock[j], mo_index[j], vo_index[j]])
		if prediction == 1:
			#print "Before: ", me_stock[j], "After: ", me_stock[j + pred_days], inc_dec[j]
			profit += me_stock[j + pred_days] - me_stock[j]
			buys += 1
			spent_money += me_stock[j]
			if me_stock[j + pred_days] - me_stock[j] > 0:
				hit += 1
		else:
			if me_stock[j + pred_days] - me_stock[j] < 0:
				hit += 1

	print stock_names[i], "Profit: ", profit, "Buys: ", buys, " Ongoing_Investment: ", float(spent_money)/buys*pred_days, " Hit Percentage: ", float(hit)/len(mo_stock)*100.0, "%"


plot_figure_triple(me_index, mo_index, vo_index, 'NASDAQ_' + date_1 + "_" + date_2, 10)	





