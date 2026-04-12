import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r'http[s]?://\S+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
WHITESPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHARS = re.compile(r'[^\w\s]')


def strip_html(text):
    if not text:
        return ''
    try:
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ')
    except Exception:
        return re.sub(r'<[^>]+>', ' ', text)


def normalize_text(text):
    if not text:
        return ''

    text = strip_html(text)
    text = text.lower()
    text = URL_PATTERN.sub(' url_token ', text)
    text = EMAIL_PATTERN.sub(' email_token ', text)
    text = SPECIAL_CHARS.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()

    return text


def tokenize_and_clean(text, remove_stopwords=True, lemmatize=True):
    if not text:
        return []

    normalized = normalize_text(text)

    try:
        tokens = word_tokenize(normalized)
    except Exception:
        tokens = normalized.split()

    cleaned = []
    for token in tokens:
        if len(token) < 2 or token.isdigit():
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        if lemmatize:
            token = LEMMATIZER.lemmatize(token)
        cleaned.append(token)

    return cleaned


def preprocess_for_model(subject='', body=''):
    clean_subject = normalize_text(subject)
    clean_body = normalize_text(body)

    # Repeat subject 3x to give it more weight
    return f'{clean_subject} {clean_subject} {clean_subject} {clean_body}'.strip()