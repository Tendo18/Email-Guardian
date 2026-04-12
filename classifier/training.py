import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

from classifier.preprocessing import preprocess_for_model

logger = logging.getLogger(__name__)


def create_synthetic_dataset():
    """
    Small built-in dataset for testing the pipeline.
    Replace this with real data for production use.
    """
    legitimate = [
        "Hi, following up on our meeting yesterday. Can we sync next week?",
        "Please find attached the Q3 report. Let me know if you have questions.",
        "Your package has shipped and will arrive by Thursday.",
        "Team lunch is at noon on Friday. We have a reservation.",
        "Could you review the pull request I opened? It is ready for feedback.",
        "The project kickoff is scheduled for Monday at 10am.",
        "Reminder: all-hands meeting tomorrow at 3pm in the main conference room.",
        "Your subscription renewal is coming up. No action needed to continue.",
    ] * 40

    spam = [
        "CONGRATULATIONS!!! You have WON a $1,000,000 prize!!! CLICK HERE NOW!!!",
        "Make MILLIONS working from home!!! No experience needed!! Limited time!!!",
        "Lose 30 pounds in 30 days!!! GUARANTEED results!!! Order NOW!!!",
        "FREE VIAGRA!!! Best prices!!! No prescription needed!!! Click here!!!",
        "You have been selected for an exclusive offer. Claim your FREE gift today!",
        "Hot singles in your area are waiting!!! Meet them now!!!",
        "Buy now pay later!!! No credit check!!! Apply today for instant approval!!!",
        "URGENT: Your account needs attention. Click to verify immediately.",
    ] * 40

    phishing = [
        "Dear Customer your PayPal account has been suspended. Verify here: http://paypa1.xyz/login",
        "Your Bank of America account will be closed. Log in now: http://192.168.1.1/boa",
        "Your Apple ID has been locked. Verify your identity at: http://apple-support.tk",
        "Security Alert: Unusual activity on your account. Confirm: http://amaz0n.ml",
        "Your Netflix subscription expired. Update payment: http://netflix-billing.xyz",
        "IRS Notice: You owe back taxes. Pay now to avoid arrest: http://irs-pay.top",
        "Your Microsoft account needs verification: http://micros0ft-verify.click",
        "Your Google account was accessed from new device. Secure: http://g00gle.ga",
    ] * 40

    data = (
        [{'text': t, 'label': 'legitimate'} for t in legitimate]
        + [{'text': t, 'label': 'spam'} for t in spam]
        + [{'text': t, 'label': 'phishing'} for t in phishing]
    )

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def get_models():
    return {
        'complement_nb': ComplementNB(alpha=0.1),
        'logistic_regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'svm': CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3),
    }


def train(data_path=None, output_dir='models', phishing_path=None, extra_path=None):
    logger.info('Starting training pipeline')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    frames = []

    # 1. Load main spam/ham dataset
    if data_path and Path(data_path).exists():
        logger.info(f'Loading main dataset from {data_path}')
        df = pd.read_csv(data_path)
        df.columns = [c.lower().strip() for c in df.columns]
        label_map = {
            'ham': 'legitimate',
            'spam': 'spam',
            'legitimate': 'legitimate',
            'phishing': 'phishing',
        }
        df['label'] = df['label'].str.lower().str.strip().map(label_map)
        df = df.dropna(subset=['label', 'text'])
        df = df[['text', 'label']]
        frames.append(df)
        logger.info(f'Main dataset: {len(df)} samples')
    else:
        logger.warning('No data path given — using synthetic dataset')
        frames.append(create_synthetic_dataset())

    # 2. Load phishing dataset
    if phishing_path and Path(phishing_path).exists():
        logger.info(f'Loading phishing dataset from {phishing_path}')
        phish_df = pd.read_csv(phishing_path)
        phish_label_map = {
            'safe email': 'legitimate',
            'phishing email': 'phishing',
        }
        phish_df['label'] = phish_df['Email Type'].str.lower().str.strip().map(phish_label_map)
        phish_df = phish_df.rename(columns={'Email Text': 'text'})
        phish_df = phish_df[['text', 'label']].dropna()
        phish_df = phish_df[phish_df['label'] == 'phishing']
        frames.append(phish_df)
        logger.info(f'Phishing dataset: {len(phish_df)} phishing samples')

    # 3. Load extra dataset (CEAS_08 style — sender, subject, body, label)
    if extra_path and Path(extra_path).exists():
        logger.info(f'Loading extra dataset from {extra_path}')
        extra_df = pd.read_csv(extra_path)
        extra_df.columns = [c.lower().strip() for c in extra_df.columns]

        # Combine subject and body into one text field
        extra_df['subject'] = extra_df['subject'].fillna('')
        extra_df['body'] = extra_df['body'].fillna('')
        extra_df['text'] = extra_df['subject'] + ' ' + extra_df['body']

        # Binary labels: 1 = spam, 0 = legitimate
        extra_df['label'] = extra_df['label'].map({1: 'spam', 0: 'legitimate'})
        extra_df = extra_df[['text', 'label']].dropna()
        frames.append(extra_df)
        logger.info(f'Extra dataset: {len(extra_df)} samples')

    # 4. Combine all datasets
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['label', 'text'])

    print(f'Dataset size: {len(df)} samples')
    print(f'Label distribution:\n{df["label"].value_counts()}')

    # 5. Preprocess
    print('\nPreprocessing text...')
    df['processed'] = df['text'].apply(lambda t: preprocess_for_model(body=str(t)))
    df = df[df['processed'].str.len() > 10]

    # 6. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    # 7. Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['processed'], y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. TF-IDF vectorization
    print('\nFitting TF-IDF vectorizer...')
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 9. Train and evaluate all models
    label_names = label_encoder.classes_.tolist()
    results = {}
    trained_models = {}

    for name, model in get_models().items():
        print(f'\nTraining {name}...')
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            }

            results[name] = metrics
            trained_models[name] = model

            print(f'  Accuracy : {metrics["accuracy"]:.3f}')
            print(f'  F1 Score : {metrics["f1_score"]:.3f}')
            print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

        except Exception as e:
            print(f'  Failed: {e}')

    # 10. Pick best model
    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]
    print(f'\nBest model: {best_name} (F1={best_metrics["f1_score"]:.3f})')

    # 11. Save artifacts
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(best_model, output_path / 'best_model.joblib')
    joblib.dump(vectorizer, output_path / 'tfidf_vectorizer.joblib')
    joblib.dump(label_encoder, output_path / 'label_encoder.joblib')

    metadata = {
        'version': version,
        'model_type': best_name,
        'labels': label_names,
        'accuracy': best_metrics['accuracy'],
        'f1_score': best_metrics['f1_score'],
        'confusion_matrix': best_metrics['confusion_matrix'],
        'training_samples': len(df),
        'trained_at': version,
        'all_results': results,
    }

    with open(output_path / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nSaved model artifacts to {output_path}/')
    return metadata