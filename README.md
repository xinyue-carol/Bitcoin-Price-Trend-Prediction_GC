# Bitcoint-Price-Trend-Prediction_GC
 [Introduction](#introduction)\
 [Dataset Selection](#dataset-selection)
 [Dataset Dimension Reduction]

# Introduction
Cryptocurrency becoming more and more popular, many big companies also choose investing in Bitcoin as part of their investement profolio. And crypptocurrency trading platform, like Coinbase, make the crypotocuurancy trading to the public. With this heat of the Cryptocuurency, I am curious wether we can predicate the change of the cryptocurrency price. In this project me and my partner applied different Machine Learning Methods to do the predict if Bitcoin price will go up or down future 30 business days,of and we eventually pick out 3 methods that perform the best: SVM, Decision Tree and Neural Network. We picked totally 17 features to do the predication, and we used PCA to do a dimension deduction and used 5 components, which can represent 91% of the variables. Final step, we built a voting system using the results from those 3 methods and able to achieve a 93% accuracy in the testing set. \
The Program language used in this project is Python, and main package including Pandas, Numpy, Scikit-lean.

# Dataset Selection
We choose the data to build the dataset base one 3 part: 
First, the Bitcoin itself, which includes open price, high price, low price, close price, and trading volume of Bitcoin, and also, to including some information of the ccyptocurrency market, we includ ETH (another popular cryptocurrency)daily close price. Second, traditional market index data and information-S&P 500, Russell 2000, and Crude Oil, Gold price, Silver price, EUR/USD, USD/JPY, and US 10-Yr-Bond rate. Last but not least, for sentement data, we choose the Wikipedia pageviews on key words ‘Bitcoin’, ‘Coinbase’ and ‘Cryptocurrency’.Time period for the dataset is all work days from 2016-06-13 to 2021-12-02, sample size 1374. Notice that though the crypotocurrency in trading on the daily based, however the traditional security market does not, so we delect all the weekend data. \
Here is a quick review of our data.
<img width="924" alt="Picture1" src="https://user-images.githubusercontent.com/71731146/150919766-c4862020-843c-4b3e-b424-c6cd6eca0b5f.png">\
Since bitcion price flucuation is abnormal, as we can see from 2020 March to 2021 April, there was a 1200% increase. With this kinda of flucuation, it is very hard to predict a accurate price, but we can still try to predict the trend. \
<img width="960" alt="price_change" src="https://user-images.githubusercontent.com/71731146/150920361-3ab1500c-980d-4ea7-bfd5-f310ab44037a.png">\
So we calcualted bitcoin's price 30 day's return, and if the return is positive, we assgin the tuple with class 1, otherwise 0. 
<img width="735" alt="Screen Shot 2022-01-24 at 10 16 43 PM" src="https://user-images.githubusercontent.com/71731146/150921652-8bffc235-d544-470c-945c-3c7c69c29182.png">\
Here is the code for first step of data cleaning: 
```
import pandas as pd
import numpy as np
#Read Data 
df=pd.read_csv('/content/data.csv')

#Calculate 30-day return
df['return']=df['close'].pct_change(30).shift(-30)

#Assign Class
df_new['class'] = np.where(df_new['return']>1, 1,0)

```
# Dimention Reducation
Now we have two questions:
1. Are all those 17 features affect the change of the BTC price? By how much?
2. Can we use less features or dimensions of the data to represent all 17 features?\
The Answer is: OF COURSE ! PCA(Principal Component Analysis) is a great tool to perform dimensionality reduction while preserving as much of the variance in the high dimensional space as possible. 






Plot 3- Component
```
#plot data
# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
 
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')
 
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,1],x[:,2], c=y,cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
```
<img width="593" alt="Screen Shot 2022-01-24 at 10 31 45 PM" src="https://user-images.githubusercontent.com/71731146/150923371-d6b8fe95-8f9f-4e67-ab48-70a5aa721375.png">

