

# ! git clone https://github.com/laxmimerit/twitter-disaster-prediction-dataset.git
url = 'https://raw.githubusercontent.com/laxmimerit/twitter-disaster-prediction-dataset/master/train.csv'
tweeter = pd.read_csv(url). # of size 7613 rows 
tweet = kgp.get_basic_features(tweeter)


""" ........... 1- NLTK ........... """
! pip install -U nltk		# (nltk = python modules + datasets)

# To download a particular dataset/models, use the nltk.download() function, e.g. if you are looking to download the punkt sentence tokenizer, use:
import nltk
nltk.download('punkt')  # Sometimes this is needed to use nltk.corpus.NAME(e.g. gutenberg).sents(‘book_name.txt’)

# If you're unsure of which data/model you need, you can start out with the basic list of data + models with:
nltk.download('popular')    # It will download a list of "popular" resources, these includes:
<collection id="popular" name="Popular packages">
      <item ref="cmudict" />
      <item ref="gazetteers" />
      <item ref="genesis" />
      <item ref="gutenberg" />
      <item ref="inaugural" />
      <item ref="movie_reviews" />
      <item ref="names" />
      <item ref="shakespeare" />
      <item ref="stopwords" />
      <item ref="treebank" />
      <item ref="twitter_samples" />
      <item ref="omw" />
      <item ref="wordnet" />
      <item ref="wordnet_ic" />
      <item ref="words" />
      <item ref="maxent_ne_chunker" />
      <item ref="punkt" />
      <item ref="snowball_data" />
      <item ref="averaged_perceptron_tagger" />
    </collection>

# stopwords    
from nltk.corpus import stopwords, twitter_samples
stopwords.words('english')[0:500:25]

# Part of Speech tagging 
from nltk import pos_tag
def _get_pos(text):
        pos=nltk.pos_tag(word_tokenize(text))
        pos=list(map(list,zip(*pos)))[1]
        return pos
      
# Name Entity Recognition NER via NLTK:
nltk.download('words')
nltk.download('maxent_ne_chunker')
from nltk import ne_chunk, pos_tag
chunked = ne_chunk(pos_tag(clean_tokens_list))
chunked.draw()

# The Brown Corpus was the first million-word electronic corpus of English, created in 1961 at Brown University
nltk.download('brown')
from nltk.corpus import brown
print(brown.sents())

from gensim.models import Word2Vec
w2v_model = Word2Vec(brown.sents(), size=128, window=5, min_count=3, workers=4)     # word2vec = Word2Vec(all_words, min_count=2)
ger_vec = w2v_model.wv['Germany']
w2v_model.wv.most_similar('Vienna')
w2v_model.wv.most_similar(positive=['woman',  'king'], negative=['man'],topn=5)
better_w2v_model = Word2Vec(Text8Corpus('data_text8_corpus.txt'), size=100, window=5, min_count=150, workers=4)
words_voc = []
for word in better_w2v_model.wv.vocab:
      words_voc.append(better_w2v_model.wv[word])

# Lemmatizer and Stemmer  
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('going', wordnet.VERB)

# 
S_stemmer = SnowballStemmer(language=”english”)
stemmer = PorterStemmer()
tokens = [stemmer.stem(t) for t in tokens]	where tokens = words 

# Tokenizers 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Sentiment Analysis with NLTK
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer     # Give a sentiment intensity score to sentences.
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(text)
ss.pop('compound')
key = ss.get


# Wordnet is an large, freely and publicly available lexical database for the English language aiming to establish structured semantic relationships between words
# It offers lemmatization capabilities as well and is one of the earliest and most commonly used lemmatizers
import nltk
nltk.download('wordnet')
nltk.download(‘vader_lexicon’)  # VADER sentiment analysis
nltk.download(“punkt”)	# download pre-trained punkt tokenizer for English

## 2. Wordnet Lemmatizer and 3. Wordnet Lemmatizer with appropriate POS tag
from nltk.stem.wordnet import WordNetLemmatizer
pass 






""" ....................................................... 2- TextBlob ....................................................... """ 
# # used for spelling corrections 
from textblob import TextBlob	# used for spelling corrections 

>> x = ‘sentence wiht topys’
>> x = TextBlob(x).correct() 


""" ...................................................... 3- Spacy ............................................................ """
# SpaCy offers the fastest syntactic parser available on the market today. Moreover, since the toolkit is written in Cython, it’s also really speedy and efficient
! python3 -m spacy download en_core_web_lg    #python3 -m spacy download en_core_web_sm
import spacy 
nlp = spacy.load('en_core_web_lg')        # nlp = spacy.load('en',disable=['parser', 'tagger','ner'])
nlp.max_length = 1198623
# doc = nlp('x')    # x = 'cat dog'
# vec = doc.vector        # now, we need to get this vector in the form of numpy array 
tweet['vec'] = tweet['text'].apply(lambda x: nlp('x').vector)     # This is word2vec word embedding from Spacy 
X = tweet['vec'].to_numpy()
X = X.reshape(-1,1)
X = np.concatenate(np.concatenate(X,axis=0),axis=0).reshape(-1,300)           # X.shape = (7613,300)


doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')    # u for unicode
for token in doc:
  print(token.text, token.pos_, token.dep_,token.lemma_,token.tag_,token.dep_,token.shape_,token.is_alpha,token.is_stop) # Token.dep_ for dependencies, e.g., 'nsubj' for nominal subject
  spacy.explain('PROPN')
  spacy.explain('nsubj')

# Tokenization with Spacy
[token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


nlp.pipeline
nlp.pipe_names		# ['tagger', 'parser', 'ner']
for sentence in doc4.sents:
    print(sentence)
doc4[6].is_sent_start		True
len(doc.vocab)
NER:  >> [(x.text,x.label_) for x in doc2.ents]

for ent in doc8.ents:		doc8.ents for name entities 
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))

doc9=nlp(‘sentence’)
doc9.noun_chunks		e.g., Autonomous cars	(nouns)

from spacy import displacy
doc = nlp(u'Apple is going to build a U.K. factory for $6 million.')
displacy.render(doc, style='dep', jupyter=True, options={'distance': 110}) 110 is the distance between tokens 
displacy.render(doc, style='ent', jupyter=True)
displacy.serve(doc, style=dep)

from spacy.tokens import span
for sent in doc.sents:		here doc.sents: is a generator, i.e., there is no doc.sents[0] in the memory. Instead, list(doc.sents)[0] (and the type is a span not actually a list).

    
Phrase Matching with Spacy:
>> from spacy.matcher import Matcher, PhraseMatcher
>> from spacy.tokens import span 
>> from spacy.lang.en.stop_words import STOP_WORDS as stopwords

>> len(stopwords) 		# 26 

pattern = [{'LOWER':'hello'},{'IS_PUNCT':True},{'LOWER':'world'}]
matcher = Matcher(nlp.vocab)
matcher.add('hw', None, pattern)
matches = matcher(doc)
    
    
    
