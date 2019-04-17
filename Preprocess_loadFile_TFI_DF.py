from pyvi import ViTokenizer, ViPosTagger
import gensim
import os
import pickle

from settings import DATA_TRAIN_PATH, DATA_TEST_PATH


def getData(folder_path,everyClass=10000000):

    X = []
    y = []
    # trong bộ dữ liệu thì tên thư mục là tên loại văn bản 
    categories = os.listdir(folder_path)  # danh sach ten thu muc
    for category in categories:
        cate_path = os.path.join(folder_path, category)
        documents = os.listdir(cate_path)  # danh sach ten van ban
        i = 0
        for document in documents:
            i=i+1
            doc_path = os.path.join(cate_path, document)
            document = open(doc_path, 'r', encoding="utf-8")
            contentDoc = document.read()
            contentDoc = ViTokenizer.tokenize(contentDoc)  # tach tu
            contentDoc = gensim.utils.simple_preprocess(contentDoc)  # xoa cac ki tu dac biet
            X.append(contentDoc)
            y.append(category)

            if i == everyClass :
                break
        # print(category," ",i)
    return X, y


def list_to_doc(X_data):
    x = []
    for doc in X_data:
        x.append(' '.join(doc))

    return x


# File Train ---------------------------------------
train_path = DATA_TRAIN_PATH
X_data, y_data = getData(train_path,300)

X_data = list_to_doc(X_data)
pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))

# File Test ----------------------------------------
test_path = DATA_TEST_PATH
X_test, y_test = getData(test_path,300)
X_test = list_to_doc(X_test)
pickle.dump(X_test, open('data/X_test.pkl', 'wb'))
pickle.dump(y_test, open('data/y_test.pkl', 'wb'))