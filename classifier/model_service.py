import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np

from classifier.preprocessing import preprocess_for_model
from classifier.features import extract_urls_from_text, extract_header_features, extract_url_features

logger = logging.getLogger(__name__)
_lock = threading.Lock()


class ModelNotLoadedError(Exception):
    pass


class ModelService:
    _instance = None

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self._model = None
        self._vectorizer = None
        self._label_encoder = None
        self._metadata = {}
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from django.conf import settings
            cls._instance = cls(settings.MODELS_DIR)
            cls._instance.load()
        return cls._instance

    def load(self):
        with _lock:
            if self._loaded:
                return True
            try:
                self._model = joblib.load(self.models_dir / 'best_model.joblib')
                self._vectorizer = joblib.load(self.models_dir / 'tfidf_vectorizer.joblib')
                self._label_encoder = joblib.load(self.models_dir / 'label_encoder.joblib')

                meta_path = self.models_dir / 'model_metadata.json'
                if meta_path.exists():
                    with open(meta_path) as f:
                        self._metadata = json.load(f)

                self._loaded = True
                logger.info(f'Model loaded: {self._metadata.get("model_type")} v{self._metadata.get("version")}')
                return True

            except Exception as e:
                logger.error(f'Failed to load model: {e}')
                return False

    def reload(self):
        with _lock:
            self._loaded = False
        return self.load()

    def is_loaded(self):
        return self._loaded

    def predict(self, subject='', body='', sender_email='', reply_to='', sender_domain=''):
        if not self._loaded:
            if not self.load():
                raise ModelNotLoadedError(
                    'No trained model loaded. Run: from classifier.training import train; train()'
                )

        # Preprocess text
        processed_text = preprocess_for_model(subject=subject, body=body)
        urls = extract_urls_from_text(f'{subject} {body}')

        # TF-IDF features
        tfidf_features = self._vectorizer.transform([processed_text])

        # For now we use TF-IDF only (we'll add header/URL features in the next step)
        X = tfidf_features

        # Predict
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(X)[0]
            predicted_idx = int(np.argmax(proba))
            confidence = float(proba[predicted_idx])
        else:
            predicted_idx = int(self._model.predict(X)[0])
            confidence = 1.0
            proba = np.zeros(len(self._label_encoder.classes_))
            proba[predicted_idx] = 1.0

        label = self._label_encoder.inverse_transform([predicted_idx])[0]
        probabilities = {
            cls: float(p)
            for cls, p in zip(self._label_encoder.classes_, proba)
        }

        return {
            'label': label,
            'confidence': round(confidence, 4),
            'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
            'model_version': self._metadata.get('version', 'unknown'),
        }

    @property
    def metadata(self):
        return self._metadata.copy()