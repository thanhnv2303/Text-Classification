from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR, NuSVC, OneClassSVM
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import  MLPClassifier
import settings
from FeatureExtraction import FeatureExtraction
from LoadFile import FileStore, FileReader
from sklearn import metrics

class Classifier(object):
    def __init__(self, features_train = None, labels_train = None, features_test = None, labels_test = None,  estimator = SVC( C=2)):
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        print(self.estimator.coef_)
        self.__training_result()

    def save_model(self, filePath): 
        FileStore(filePath=filePath).save_pickle(obj=est)
    #trả về kết quả phân lớp
    def __training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        #self.estimator.predict(self.features_test) trả về tập nhãn của features_test được dự đoán
        print("Test accuracy: ", metrics.accuracy_score(y_true, y_pred))
        print("Test Precision-Recall : ")
        print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    train_loader = FileReader(filePath=settings.DATA_TRAIN_JSON)
    test_loader = FileReader(filePath=settings.DATA_TEST_JSON)
    data_train = train_loader.read_json()
    data_test = test_loader.read_json()
    features_train, labels_train = FeatureExtraction(data=data_train).get_data_and_label()
    # X_train,y_train,X_val,y_val= train_test_split(features_train,labels_train,test_size=0.1, random_state=42)
    features_test, labels_test = FeatureExtraction(data=data_test).get_data_and_label()

    # est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train,
    #                  labels_test=labels_test)
    est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train,
                     labels_test=labels_test)
    est.training()

    # est.save_model(filePath='trained_model/linear_svc_3c_9k.pk')