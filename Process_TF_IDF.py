import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC, NuSVC

folder_path = 'data/X_data.pkl'
X_data = pickle.load(open(folder_path, 'rb'))
folder_path = 'data/y_data.pkl'
y_data = pickle.load(open(folder_path, 'rb'))
folder_path = 'data/X_test.pkl'
X_test = pickle.load(open(folder_path, 'rb'))
folder_path = 'data/y_test.pkl'
y_test = pickle.load(open(folder_path, 'rb'))

# biến đổi nhãn về dạng số

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)

# Biến đỗi các doc về dạng if-idf

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf_vect = TfidfVectorizer(analyzer='word', max_features=250000)
tfidf_vect.fit(X_data)
X_data_tfidf = tfidf_vect.transform(X_data)
X_test_tfidf = tfidf_vect.transform(X_test)
# Train-model

from sklearn.model_selection import train_test_split
from sklearn import metrics


def train_model(classifier, X_data, y_data, X_test, y_test):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=None)
    classifier.fit(X_train, y_train)
    print(len(classifier.coef_))
    train_predictions = classifier.predict(X_train)
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)

    print('Train accuracy: ', metrics.accuracy_score(train_predictions, y_train))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))
    print(classification_report(y_test, test_predictions))

# Mô hình Naive-Bayes
from sklearn.naive_bayes import MultinomialNB

train_model(LinearSVC(multi_class='ovr',C=1), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n)