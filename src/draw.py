from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import CNN_Classifier
from CNN_Classifier import history,epochs,x_test,x_train,y_test,y_train,model
import numpy as np
import scikitplot
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.yticks(np.arange(0.5, 1.0, step=0.025))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

yhat_valid = model.predict_classes(x_test)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_test, axis=1), yhat_valid)
plt.title('Confusion Matrix for prediction using CNN')
plt.show()

scrore = model.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(np.argmax(y_test, axis=1), scrore)
plt.plot(fpr, tpr)
# curve = plot_roc_curve(model, x_test, y_test)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Roc for prediction using CNN')
plt.plot([0, 1], [0, 1], color='green',linestyle='--')
plt.show()

preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))