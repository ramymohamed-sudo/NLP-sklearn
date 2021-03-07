
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess_kgptalkie as kgp

from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
from sklearn.svm import SVR

# ! git clone https://github.com/laxmimerit/twitter-disaster-prediction-dataset.git
url = 'https://raw.githubusercontent.com/laxmimerit/twitter-disaster-prediction-dataset/master/train.csv'
tweeter = pd.read_csv(url)
tweet = kgp.get_basic_features(tweeter)


""" Classification with Support Vector Machine and TFIDF"""
text = tweeter['text']
y = tweeter['target']
tfidf = TfidfVectorizer()
print(tweeter.shape)
X = tfidf.fit_transform(text)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify=y) 

clf = LinearSVC()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print('Classification Report')
print(classification_report(y_test,y_pred))


""" Classification with Support Vector Machine and Word2vec"""













