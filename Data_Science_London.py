import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train_x=pd.read_csv(r'C:\Users\huang\Downloads\train (1)_scilearn.csv',header=None)
train_y=pd.read_csv(r'C:\Users\huang\Downloads\trainLabels_scilearn.csv',header=None)
test_x=pd.read_csv(r'C:\Users\huang\Downloads\test (1)_scilearn.csv',header=None)

logreg=LogisticRegression()

logreg.fit(train_x,train_y)
predicted=logreg.predict(test_x)

predicted=pd.DataFrame(predicted,index=np.arange(1,len(predicted)+1),columns=['Solution'])
predicted['Id']=predicted.index.name()
predicted.index=predicted['Id']

predicted.to_csv(r'C:\Users\huang\Desktop\london.csv')

