import re
import numpy as np
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {'.xyz', '.tk', '.ml', '.ga', '.cf', '.top', '.click'}
FREE_EMAIL_PROVIDERS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'}
URL_SHORTENERS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly'}
IP_IN_URL = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
OBFUSCATED_URL = re.compile(r'%[0-9a-fA-F]{2}')
TYPOSQUATTING_BRANDS = ['paypa1', 'g00gle', 'amaz0n', 'app1e', 'micros0ft']
URL_PATTERN = re.compile(r'http[s]?://\S+')


def extract_urls_from_text(text):
    return URL_PATTERN.findall(text) if text else []


def extract_header_features(sender_email='', reply_to='', sender_domain=''):
    features = []

    domain = sender_domain.lower().strip()
    if not domain and sender_email and '@' in sender_email:
        domain = sender_email.split('@')[-1].lower().strip()

    # Is sender a free email provider?
    features.append(1.0 if domain in FREE_EMAIL_PROVIDERS else 0.0)

    # Suspicious TLD?
    features.append(1.0 if any(domain.endswith(t) for t in SUSPICIOUS_TLDS) else 0.0)

    # Reply-to differs from sender?
    reply_domain = ''
    if reply_to and '@' in reply_to:
        reply_domain = reply_to.split('@')[-1].lower().strip()
    features.append(1.0 if (reply_domain and reply_domain != domain) else 0.0)

    # No domain at all?
    features.append(1.0 if not domain else 0.0)

    # Digits in domain? (paypa1.com)
    features.append(1.0 if re.search(r'\d', domain) else 0.0)

    # Lots of hyphens in domain?
    features.append(min(domain.count('-') / 5.0, 1.0))

    return np.array(features, dtype=np.float32)


def analyze_url(url):
    result = {
        'uses_ip': False,
        'is_shortened': False,
        'has_obfuscation': False,
        'suspicious_tld': False,
        'is_typosquatting': False,
        'url_length': len(url),
    }
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()

        result['uses_ip'] = bool(IP_IN_URL.match(url))
        result['is_shortened'] = netloc in URL_SHORTENERS
        result['has_obfuscation'] = bool(OBFUSCATED_URL.search(url))
        result['suspicious_tld'] = any(netloc.endswith(t) for t in SUSPICIOUS_TLDS)
        result['is_typosquatting'] = any(b in netloc for b in TYPOSQUATTING_BRANDS)
    except Exception:
        pass
    return result


def extract_url_features(urls):
    if not urls:
        return np.zeros(7, dtype=np.float32)

    analyses = [analyze_url(u) for u in urls]

    features = [
        max(a['uses_ip'] for a in analyses),
        max(a['is_shortened'] for a in analyses),
        max(a['has_obfuscation'] for a in analyses),
        max(a['suspicious_tld'] for a in analyses),
        max(a['is_typosquatting'] for a in analyses),
        min(len(urls) / 20.0, 1.0),
        min(np.mean([a['url_length'] for a in analyses]) / 200.0, 1.0),
    ]

    return np.array(features, dtype=np.float32)