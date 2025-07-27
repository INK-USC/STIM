import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def find_all_substring_indices(s, sub):
    results = []
    start = 0
    while True:
        start = s.find(sub, start)
        if start == -1:
            break
        end = start + len(sub)  # End index is exclusive
        results.append((start, end))
        start = end
    return results

def is_stop_words(word: str) -> bool:
    """
    Check whether a word is a stop word
    """
    stop_words = set(stopwords.words('english'))
    if word.lower() in stop_words:
        return True
    elif (word in string.punctuation and word not in "+-*/=%") or word == "":
        return True
    else:
        return False