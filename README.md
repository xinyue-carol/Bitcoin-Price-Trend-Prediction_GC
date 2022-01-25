# Bitcoint-Price-Trend-Prediction_GC
 [Introduction](#introduction)\
 [Dataset Selection](#dataset-selection)\
 [Dataset Dimension Reduction](#dimention-reducation)\
 [Method 1_SVM](#method-1_SVM)

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
So we calcualted bitcoin's price 30 day's return, and if the return is positive, we assgin the tuple with class 1, otherwise 0. And we will treat this as an classification problem, we tried to treat this as an regression problem, but the restults is not as good.
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
X=df_new[['high', 'low', 'open', 'close', 'volumeto', 'ETH', 'SP500',
       'Russell_2000', 'Crude_oil', 'Gold', 'Silver', 'EUR/USD',
       'Treasury Yield 10 Years', 'USD/JPY', 'wiki_Bitcoin',
       'wiki_Cryptocurrency', 'wiki_Coinbase']]
y=df_new[['class']]
```
# Dimention Reducation
Now we have two questions:
1. Are all those 17 features affect the change of the BTC price? By how much?
2. Can we use less features or dimensions of the data to represent all 17 features?\
The Answer is: OF COURSE ! PCA(Principal Component Analysis) is a great tool to perform dimensionality reduction while preserving as much of the variance in the high dimensional space as possible. 
BTW! Remember to scales the data first!
```
#Scale the data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

#Import PCA from sklean
from sklearn.decomposition import PCA

# Set the n_components=7
principal=PCA(n_components=5)
principal.fit(X_scaled)
x_pca=principal.transform(X_scaled)

print(principal.explained_variance_ratio_)
print(principal.explained_variance_ratio_.cumsum())
```
Here is the results: 
PC_1: contains 55.24% of the variations
PC_2: contains 18.89% of the variations
PC_3: contains 9.99% of the variations
PC_4: contains 5.14% of the variations
PC_5: contains 2.61% of the variations
Total first 5 component are enough to explain 91.94% of all variations, for following analysis, we are using this result as the input data. Here is about the data looks like:

<img width="413" alt="Picture_PCA" src="https://user-images.githubusercontent.com/71731146/150928075-70593387-fd73-4d4b-80cc-219604c81290.png">\

And I plot out first 3 component: 
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

# Method 1_SVM
First Machine leaning we used here is Support Vector Machine (SVM), it is a great supervised learning methods for classification, and it also works great with high dimension data sets. The kernel selection plays a inportant rule when using SVM, it used to meansure the similarity of data points. As we can tell from the graph of 3 dimension data, two class are tangling together, intuitivly we can tell that the linear kernel won't able to seperate our data efectively, so we picked Polynomial(POLY) and Gaussian Radial Basis Function (RBF)
## Polynomial Kenel
Formula: <img width="210" alt="Picture_poly" src="https://user-images.githubusercontent.com/71731146/150930363-2bd98bf9-9c79-4e91-bee6-9f8d6e0b13ee.png">
Results: R^2=0.7546

## Gaussian Radial Basis Function (RBF)
Formula:<img width="280" alt="Picture1" src="https://user-images.githubusercontent.com/71731146/150930686-f2a1558b-c53a-46e5-bed8-e58073561f74.png">
Results: R^2=0.80660\
Here is the code to apply SVM model to our data set
```
from sklearn.model_selection import train_test_split
# splitting the data
# x_train, x_test, y_train, y_test = train_test_split(X, y_tl, test_size = 0.2, random_state = 0)
x_train, x_test, y_train, y_test = train_test_split(x_lda_data, y_lda_class, test_size = 0.2, shuffle=True,random_state=22)

# Try different kernels
from sklearn import svm
kernels = [ 'rbf', 'poly']
for kernel in kernels:
  svc = svm.SVC(kernel=kernel).fit(x_train, y_train)
  score = svc.score(x_test, y_test)
  print('kernels for', kernel, 'r2 socre is',score)
```
There are two parameter we need to adjust in order have a better results.
1. C: Inverse of the strength of regularization.
As the value of ‘c’ increases the model gets overfits.
As the value of ‘c’ decreases the model underfits.

2.   γ : Gamma (used only for RBF kernel)
As the value of ‘ γ’ increases the model gets overfits.
As the value of ‘ γ’ decreases the model underfits.
After we tested on different number, we have when c is 100, and gamma is 1, using RBF kernel, the SVM model can achieve 91.4% accuracy on testing data set.

