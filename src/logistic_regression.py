from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, log_loss
from scikitplot.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ReadImage as RI


def train_lr(x, y, C=1, penalty='none', solver='lbfgs'):
    lr_model = LogisticRegression(penalty=penalty, solver=solver, max_iter=3000, C=C).fit(x, y)
    return lr_model


def cross_validation(model, x, y, cv):
    scores = cross_val_score(model, x, y, cv=cv, scoring='f1')
    model.fit(x, y)
    mean_score = np.array(scores).mean()
    std_dev = np.array(scores).std()
    return (mean_score, std_dev)


def convert_to_vector(imgs):
    ret = []
    for img in imgs:
        ret.append(img.reshape(img.size))
    return np.array(ret)


def get_pca():
    Children_train = "Image/Children_train"
    Adults_train = "Image/Adults_train"
    x_children_train = RI.read_img_batch(Children_train)
    x_adults_train = RI.read_img_batch(Adults_train)
    x_train = np.array(x_children_train + x_adults_train)
    pca = PCA(917)
    pca.fit(convert_to_vector(x_train))
    return pca

def find_dimension(imgs):
    vectorised_x_train = convert_to_vector(imgs)
    n_components = 1
    while True:
        print(n_components)
        pca = PCA(n_components)
        pca.fit_transform(vectorised_x_train)
        flag = False
        if pca.explained_variance_ratio_.sum() > 0.99:
            n_components -= 100
            while True:
                print(n_components)
                pca = PCA(n_components)
                pca.fit_transform(vectorised_x_train)
                if pca.explained_variance_ratio_.sum() > 0.99:
                    flag = True
                    break
                else:
                    n_components += 1
            if flag:
                break
        else:
            n_components += 100
    return n_components


def get_plot_data():
    x_train, y_train, x_test, y_test = RI.make_dataset()

    pca = PCA(917)
    pca.fit(convert_to_vector(x_train))

    reduced_x_train = pca.fit_transform(convert_to_vector(x_train))
    model = train_lr(reduced_x_train, y_train.ravel())
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(pca.transform(convert_to_vector(x_test))))

    return (fpr, tpr)


if __name__ == '__main__':
    ts = datetime.datetime.now()
    x_train, y_train, x_test, y_test = RI.make_dataset()

    # find #component
    n_components = find_dimension(x_train)
    pca = PCA(n_components)
    reduced_x_train = pca.fit_transform(convert_to_vector(x_train))
    model = train_lr(reduced_x_train, y_train.ravel())
    score = f1_score(y_test.ravel(), model.predict(pca.transform(convert_to_vector(x_test))))
    # evaluation on training set
    pred = model.predict(reduced_x_train)
    cm_lr = confusion_matrix(y_train.ravel(), model.predict(pca.transform(convert_to_vector(x_train))))
    tn, fp, fn, tp = cm_lr.ravel()
    print("(on training set)")
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print("accuracy: {}".format((tn + tp) / (tn + tp + fn + fp)))
    print("f1 score: {}".format(2 * tp / (2 * tp +fn + fp)))
    print("cross-entropy loss: {}".format(log_loss(y_train, model.predict(pca.transform(convert_to_vector(x_train))))))
    # evaluation on test set
    pred = model.predict(pca.transform(convert_to_vector(x_test)))
    cm_lr = confusion_matrix(y_test.ravel(), model.predict(pca.transform(convert_to_vector(x_test))))
    tn, fp, fn, tp = cm_lr.ravel()
    print("(on test set)")
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print("accuracy: {}".format((tn + tp) / (tn + tp + fn + fp)))
    print("f1 score: {}".format(2 * tp / (2 * tp +fn + fp)))
    y_cat_0 = y_test
    y_cat_1 = np.abs(y_test - 1)
    y_test_one_hot = np.column_stack((y_cat_0, y_cat_1))
    print("cross-entropy loss: {}".format(log_loss(y_test_one_hot, model.predict_proba(pca.transform(convert_to_vector(x_test))))))
    plot_confusion_matrix(y_test.ravel(), pred.ravel())
    # ROC curve on test set
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(pca.transform(convert_to_vector(x_test))))
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    te = datetime.datetime.now()
    print('time elapsed: {}s'.format((te - ts).total_seconds()))

    plt.show()