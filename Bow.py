import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph ="Al, machine learning and deep learning are common terms in3 enterpriseIT and sometimes used interchangeably, especially by companies in their marketing materials. But there are distinctions. The term Al, coined in the 1950s, refers to the simulation of human intelligence by machines. It covers an ever-changing set of capabilities as new technologies are developed. Technologies that come under the umbrella of Al include machine learning and deep learning. Machine learning enables software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values. This approach became vastly more effective with the rise of large data sets to train on. Deep learning, a subset of machine learning, is based on our understanding of how the brain is structured. Deep learning's use of artificial neural networks structure is the underpinning of recent advances in Al, including self-driving cars and ChatGPT."
#cleaning the texts
import re #re library will will for rwgular expession
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentence = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentence)):
    review = re.sub("[^a-zA-Z]", " ", sentence[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set (stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
#create the bow model
from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer()
x1 = cv1.fit_transform(corpus).toarray()

#creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()