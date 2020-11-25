#modulo para o regex
import re
#modulo para o html parser
from bs4 import BeautifulSoup
#modulo de acentos
import unidecode
import pandas as pd
import time
#import para a lista de stopords do spicy
import spacy
#lista global de stopwords -- MEXER DEPOIS NELA, PARA ALTERAR OS STOPWORDS SENTIMENTAIS.
sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words

def readInput():
	#panda dataframe
	entrada = pd.read_csv('IMDB Dataset.csv')
	#passa a entrada para uma serie, para o preprocessamento.
	entrada=entrada['review'].apply(preProcessamento)
	print(entrada.head())

def preProcessamento(linha):
	#tira o HTML.
	soup = BeautifulSoup(linha,'html.parser')
	linha = soup.get_text()
	#tira os acentos
	linha = unidecode.unidecode(linha)
	#tira os pontos, transforma em minuscula
	linha = re.sub('[^a-z\s]', '', linha.lower())
	#separa em tokens. E possivel utilizar split pois foi usado um regex para tirar os pontos anteriormente. Alem disto, retira os stopWords.
	linha = [palavra for palavra in linha.split() if palavra not in set(stopwords)]
	#adiciona no dataframe.
	return ','.join(linha)  

def main():
	readInput()
if __name__ == '__main__':
    main()