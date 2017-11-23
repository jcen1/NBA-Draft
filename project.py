# Feature Extraction with RFE
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "Test-Dataset-Revised-Copy.csv"
names = ['G','MP','MP','FG','FGA','2P','2PA','3P','3PA','FT','FTA',	'ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','WinShares_48','WinShares_Career','school','position']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,1:]
Y = array[:,0]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
