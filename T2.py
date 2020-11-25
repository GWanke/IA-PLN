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
#import para a lista de stopwords do spacy
import spacy
#lista global de stopwords -- MEXER DEPOIS NELA, PARA ALTERAR OS STOPWORDS SENTIMENTAIS.
sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words

def readInput():
	#panda dataframe
	entrada = pd.read_csv('IMDB Dataset.csv')
	#passa a entrada para uma serie, para o preprocessamento.
	entrada['review'] = entrada['review'].apply(preProcessamento)
	print(entrada.head(10))

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
	return ','.join(linha)  

def main():
	if 'like' in stopwords:
		print('ye')
	start = time.time()
	readInput()
	end = time.time()
	print(end-start)
if __name__ == '__main__':
    main()