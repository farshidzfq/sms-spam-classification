import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# دانلود منابع NLTK فقط یکبار
nltk.download('stopwords')
nltk.download('punkt')

# تعریف متغیرهای سراسری برای پیش‌پردازش
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # حذف نویز و کاراکترهای غیرحروف
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens if len(word) > 2]  # حذف کلمات خیلی کوتاه
    return " ".join(tokens)

# دریافت و بارگذاری دیتاست SMS Spam Collection از اینترنت
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

print(f"تعداد کل نمونه‌ها: {len(df)}")
print(df['label'].value_counts())
print("\nنمونه داده‌ها:")
print(df.head())

# پیش‌پردازش متن
df['processed_text'] = df['text'].apply(preprocess_text)

# تبدیل برچسب به عدد
df['label_numeric'] = df['label'].map({'ham': 0, 'spam': 1})

# برداردهی با TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label_numeric']

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nتعداد نمونه‌های آموزشی: {X_train.shape[0]}")
print(f"تعداد نمونه‌های تست: {X_test.shape[0]}")

# مدل‌ها
model_nb = MultinomialNB()
model_lr = LogisticRegression(solver='liblinear', random_state=42)

# آموزش مدل‌ها
model_nb.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

# ارزیابی
def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    print(f"\n--- نتایج مدل {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")

    # ماتریس درهم ریختگی
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # نمودار ROC فقط اگر قابلیت داشت
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    return accuracy, precision, recall, f1, roc_auc

results_nb = evaluate(model_nb, X_test, y_test, "Naive Bayes")
results_lr = evaluate(model_lr, X_test, y_test, "Logistic Regression")

# تابع پیش‌بینی روی متن جدید
def predict_spam(text, vectorizer, model):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred_numeric = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "اسپم" if pred_numeric == 1 else "غیر اسپم"
    return label, prob

# تست روی نمونه جدید
new_samples = [
    "Congratulations! You won a free ticket to Bahamas!",
    "Can we have a meeting tomorrow?",
    "Lowest price for your meds here, buy now!",
    "Here's the report you asked for."
]

print("\n--- پیش‌بینی روی نمونه‌های جدید ---")
for text in new_samples:
    label, prob = predict_spam(text, vectorizer, model_lr)
    print(f"متن: {text}")
    print(f"پیش‌بینی: {label}, احتمال اسپم: {prob[1]:.2%}")
    print("-" * 40)
