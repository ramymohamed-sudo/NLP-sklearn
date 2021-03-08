

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
# Example of TfidfVectorizer
tfidf = TfidfVectorizer(
  norm=l1 #(default =l2)
  ngram_range=(1,2)		#(default=1,1)
  analyzer='word'		# tokenization done word by word (‘char’)
  max_features = 5000, #(limit the dictionary to 5000)
  stop_words=stopwords.words('english'),
  max_df=0.5,
  min_df=0.1,
  lowercase=True
)
feature_matrix = tfidf.fit_transform(body_all_articles)
feature_matrix.toarray()
tfidf.get_feature_names() 
tfidf.labels_
tfidf.vocabulary_
fmArray = feature_matrix.toarray() 

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


""" Classification with Support Vector Machine and (Word2vec)"""
import spacy 
nlp = spacy.load('en_core_web_lg')        # nlp = spacy.load('en',disable=['parser', 'tagger','ner'])
tweet['vec'] = tweet['text'].apply(lambda x: nlp('x').vector)
X = tweet['vec'].to_numpy()
X = X.reshape(-1,1)
X = np.concatenate(np.concatenate(X,axis=0),axis=0).reshape(-1,300)           # X.shape = (7613,300)
# accuract is lower than TFIDF as Spacy is trained for Standard English not Twitter data
# 







""" Pipeline """
pass
>> from sklearn.pipeline import Pipeline
>> from sklearn.model_selection import GridSearchCV
>> pipe = Pipeline([(“classifier”, RandomForestClassifier())])
>> search_space = 			# list of dictionaries 

>> clf = clf = GridSearchCV(pipe,search_space,cv=5,njobs=-1)
>> clf.fit(X_train,y_train)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])




For the pipeline, i.e., TFIDF and Classifier 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

""" K-mean Clustering """
num = 5
kmeans = KMeans(n_clusters = num, init = 'k-means++', max_iter = 500, n_init = 1)
kmeans.fit(feature_matrix)




""" Other Models """
pass














