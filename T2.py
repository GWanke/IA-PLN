#modulo para o regex
import re
#modulo para o html parser
from bs4 import BeautifulSoup
#modulo de acentos
import unidecode
import pandas as pd
import time
#modulo do Stemmer. Utiliza o Snowball Stemmer, com um wrap em C para melhorar o desempenho.
import Stemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
#import para a lista de stopwords do spacy
import spacy
#lista global de stopwords -- MEXER DEPOIS NELA, PARA ALTERAR OS STOPWORDS SENTIMENTAIS.
sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words

def readInput():
	#Le a entrada, converte a coluna de sentimentos para binario, adotando 1 como positivo.
	entrada = pd.read_csv('IMDB Dataset.csv',converters={'sentiment': lambda x: int(x == 'positive')})
	#passa a entrada para uma serie, para o preprocessamento.
	entrada['review'] = entrada['review'].apply(preProcessamento)
	return entrada


def preProcessamento(linha):
	#tira o HTML.
	soup = BeautifulSoup(linha,'html.parser')
	linha = soup.get_text()
	#tira os acentos
	linha = unidecode.unidecode(linha)
	#tira os pontos, transforma em minuscula
	linha = re.sub('[^a-z\s]', '', linha.lower())
	#stemming + tokenizer + Retira os stopWords.
	ps = Stemmer.Stemmer('english')
	linha = [ps.stemWord(palavra) for palavra in linha.split() if palavra not in stopwords]
	#adiciona no dataframe.
	return ' '.join(linha)

def Vectorizer(treino,teste):
	#Vectorizers. Da para mudar o ngram_range, mas a complexidade aumenta. 
	vectBow = CountVectorizer(binary=False,min_df=2,ngram_range=(1,1))
	vectTfidf = TfidfVectorizer(use_idf=True,min_df=2,ngram_range=(1,1))
	#Aplica o vector no dataframe, resultando em um Bag of Words.
	bowX = vectBow.fit_transform(treino)
	bowY = vectBow.transform(teste)
	#Aplica o vector no dataframe, resultando no TFIDF. 
	tfidfX = vectTfidf.fit_transform(treino)
	tfidfY = vectTfidf.transform(teste)
	#da pra saber as palavras no vocabulario assim:
	#print(vectBow.get_feature_names())
	return bowX,bowY,tfidfX,tfidfY



def TreinoModel(Xtreino,Xteste,Ytreino,Yteste):
	MNB = MultinomialNB()
	#treinando o model
	modelo = MNB.fit(Xtreino, Ytreino)
	resTreino = modelo.predict(Xtreino)
	resTeste = modelo.predict(Xteste)

	print('Train Accuracy:', accuracy_score(Ytreino, resTreino))
	print('Test Accuracy:', accuracy_score(Yteste, resTeste))


def main():	
	start = time.time()
	entrada=readInput()
	#Equivale as features para o MNB.
	x = entrada['review']
	#Equivale as variaveis para o MNB.
	y = entrada['sentiment']
	#split das entradas. Conjunto de treino = 0.8%.
	x_treino, x_teste , y_treino, y_teste = train_test_split(x,y,test_size = 0.2)
	#vetorizando - preparando para aplicar o modelo MNB
	bowX,bowY,tfidfX,tfidfY = Vectorizer(x_treino,x_teste)
	#da pra variar os parametros do metodo para variar a forma de vetorizacao(bow ou tfidf)
	TreinoModel(tfidfX,tfidfY,y_treino,y_teste)
	end = time.time()
	print(end-start)
if __name__ == '__main__':
    main()