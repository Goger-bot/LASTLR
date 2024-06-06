import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
import ssl

# Отключение проверки SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = text.lower().split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)

# Шаг 1: Чтение и предобработка данных
file_path = 'tweets.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
data.columns = ['target', 'id', 'date', 'query', 'user', 'text']

# Применение предобработки к текстам
data['text'] = data['text'].apply(preprocess_text)

# Отбор только необходимых столбцов и преобразование целевой переменной
data = data[['target', 'text']]
data['target'] = data['target'].map({0: 0, 4: 1})

# Токенизация и заполнение последовательностей
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
data_padded = pad_sequences(sequences, maxlen=100)

# Подготовка меток и разделение данных
labels = data['target'].values
X_train, X_val, y_train, y_val = train_test_split(data_padded, labels, test_size=0.2, random_state=42)

# Шаг 3: Построение нейронной сети
input_layer = Input(shape=(100,))
embedding_layer = Embedding(len(tokenizer.word_index) + 1, 300, input_length=100, trainable=True)(input_layer)
x = LSTM(20)(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(x)
nn_model = Model(inputs=input_layer, outputs=output_layer)
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Компиляция и обучение модели с ограничением количества шагов в эпохе
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
history = nn_model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint], steps_per_epoch=200)

# Визуализация истории обучения
fig, ax1 = plt.subplots()
color1 = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color1)
ax1.plot(history.history['loss'], 'r-', label='Training Loss')
ax1.plot(history.history['val_loss'], 'r--', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color2)
ax2.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], 'b--', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color2)
fig.tight_layout()
plt.title('Training Progress')
fig.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()

# Шаг 5: Оценка и визуализация
# Выборка и классификация случайных предложений
random_indices = np.random.choice(data.index, size=100, replace=False)
random_samples = data.loc[random_indices, 'text']
random_samples_sequences = tokenizer.texts_to_sequences(random_samples)
random_samples_padded = pad_sequences(random_samples_sequences, maxlen=100)
actual_classes = data.loc[random_indices, 'target'].values
predicted_classes = nn_model.predict(random_samples_padded).round().astype(int)
report = classification_report(actual_classes, predicted_classes, target_names=['Negative', 'Positive'])
print("Classification Report:\n", report)

# Получение весов слоя встраивания
embedding_weights = nn_model.layers[1].get_weights()[0]

# Подготовка слов и их вложений для TSNE
word_counts = {}
for sample in random_samples:
    for word in sample.split():
        if word in tokenizer.word_index:
            word_counts[word] = word_counts.get(word, 0) + 1
sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)[:10]
word_embeddings = np.array([embedding_weights[tokenizer.word_index[word]] for word in sorted_words if word in tokenizer.word_index])

# Выполнение TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(word_embeddings)-1))
transformed_embeddings = tsne.fit_transform(word_embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1])
for i, word in enumerate(sorted_words):
    plt.annotate(word, (transformed_embeddings[i, 0], transformed_embeddings[i, 1]))
plt.title('t-SNE visualization of top 10 words')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()
