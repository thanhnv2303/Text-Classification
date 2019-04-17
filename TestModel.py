import pickle
import Classifier
import settings
from FeatureExtraction import FeatureExtraction
from LoadFile import FileReader

if __name__ == '__main__':
    train_loader = FileReader(filePath=settings.DATA_TRAIN_JSON)
    test_loader = FileReader(filePath=settings.DATA_TEST_JSON)
    data_train = train_loader.read_json()
    data_test = test_loader.read_json()
    features_train, labels_train = FeatureExtraction(data=data_train).get_data_and_label()
    features_test, labels_test = FeatureExtraction(data=data_test).get_data_and_label()

    est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train,
                     labels_test=labels_test)
    est = pickle.load(open('trained_model/linear_svc_model.pk', 'rb'))