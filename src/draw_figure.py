from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import scikitplot

from CNN_Classifier import history,epochs,x_test,x_train,y_test,y_train,model
import logistic_regression as LR

# obtain data from the CNN model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plot accuracy on training set and validation set
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.yticks(np.arange(0.5, 1.0, step=0.025))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


# plot loss of training set and validation set
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# visualise the confusion matrix for the CNN model
yhat_valid = model.predict_classes(x_test)
yhat_valid = (model.predict(x_test) > 0.5).astype("int32")
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_test, axis=1), yhat_valid)
plt.title('Confusion Matrix for prediction using CNN')
plt.show()

# plot ROC curves for the CNN model and the Logistic Regression model
score = model.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(np.argmax(y_test, axis=1), score)
plt.plot(fpr, tpr, label='CNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Roc for prediction using CNN')
plt.plot([0, 1], [0, 1], color='green',linestyle='--')
# obtain fpr and tpr of the logistic regression model
fpr, tpr = LR.get_plot_data()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.legend()
plt.show()

# generate report
preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))
