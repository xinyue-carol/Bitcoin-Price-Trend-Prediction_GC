# Bitcoint-Price-Trend-Prediction_GC
 [Introduction](#introduction)
 [Dataset Selections](#dataset-selection)


# Introduction
Cryptocurrency becoming more and more popular, many big companies also choose investing in Bitcoin as part of their investement profolio. And crypptocurrency trading platform, like Coinbase, make the crypotocuurancy trading to the public. With this heat of the Cryptocuurency, I am curious wether we can predicate the change of the cryptocurrency price. In this project me and my partner applied different Machine Learning Methods to do the predict if Bitcoin price will go up or down future 30 business days,of and we eventually pick out 3 methods that perform the best: SVM, Decision Tree and Neural Network. We picked totally 17 features to do the predication, and we used PCA to do a dimension deduction and used 5 components, which can represent 91% of the variables. Final step, we built a voting system using the results from those 3 methods and able to achieve a 93% accuracy in the testing set. 

# Dataset Selection
Data we Choose including: 
Open price, high price, low price, close price, and trading volume of Bitcoin, and also, to including some information of the ccyptocurrency market, we includ ETH (another popular cryptocurrency)daily close price. We also include some market index data-S&P 500, Russell 2000, and Crude Oil, Gold price, Silver price, EUR/USD, USD/JPY, and US 10-Yr-Bond rate. Totally 17 features. Time period for the dataset is all work days from 2016-06-13 to 2021-12-02. Notice that though the crypotocurrency in trading on the daily based, however the traditional security market does not, so we delect all the weekend data. 
