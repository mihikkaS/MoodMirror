import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from sklearn.utils import resample

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Load the dataset from a TXT file with ; delimiter
df = pd.read_csv('data/train.txt', sep=';', names=['text', 'emotion'])

print(df['emotion'].value_counts())
max_size = df['emotion'].value_counts().max()

balanced_df = pd.DataFrame()

for emotion in df['emotion'].unique():
    emotion_df = df[df['emotion'] == emotion]
    if len(emotion_df) < max_size:
        emotion_df_upsampled = resample(emotion_df, replace=True, n_samples=max_size, random_state=42)
        balanced_df = pd.concat([balanced_df, emotion_df_upsampled])
    else:
        balanced_df = pd.concat([balanced_df, emotion_df])

balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)  # Shuffle

df = balanced_df 

# Step 2: Clean the text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Features and labels
X = df['clean_text']
y = df['emotion']

# Step 4: Vectorization
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Step 5: Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Step 6: Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Step 8: Save the model
joblib.dump(model, 'model/emotion_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
