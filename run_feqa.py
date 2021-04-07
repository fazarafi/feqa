import benepar
import nltk
from feqa import FEQA

benepar.download('benepar_en2')
nltk.download('stopwords')

#!python -m spacy download en_core_web_sm
scorer = FEQA(use_gpu=True)

documents = [
             "The world's oldest person has died a \
             few weeks after celebrating her 117th birthday.  \
             Born on March 5, 1898, the greatgrandmother had lived through two world \
             wars, the invention of the television and the \
             first successful powered aeroplane.", 
            "The world's oldest person has died a \
             few weeks after celebrating her 117th birthday.  \
             Born on March 5, 1898, the greatgrandmother had lived through two world \
             wars, the invention of the television and the \
             first successful powered aeroplane."]
summaries = [
             "The world's oldest person died in 1898",
             "The world's oldest person died after her 117th birthday"]
scorer.compute_score(documents, summaries, aggregate=False)
