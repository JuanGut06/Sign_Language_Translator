from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

width_shape = 200
height_shape = 200
batch_size = 32

names = ['ABRIL', 'ADULTO', 'AGOSTO', 'BIEN', 'BIENVENIDO', 'CAMA', 'CINCO', 'COMO_ESTA', 'CON_MUCHO_GUSTO',
               'CUATRO', 'DICIEMBRE', 'DOMINGO', 'DOS', 'ENERO', 'FEBRERO', 'GRACIAS', 'HABITACION', 'HOLA', 'HOTEL',
               'JUEVES', 'JULIO', 'JUNIO', 'LUNES', 'MAL', 'MARTES', 'MARZO', 'MAYO', 'MIERCOLES', 'NINO', 'NO',
               'NOVIEMBRE', 'OCTUBRE', 'POR_FAVOR', 'SABADO', 'SEPTIEMBRE', 'SI', 'TRES', 'UNO', 'VIERNES']

test_data_dir = 'D:/User/Project/Code/Sign_Language_Translator/test'
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(width_shape, height_shape),
    batch_size = batch_size,
    class_mode='categorical',
    shuffle=False)

custom_Model = load_model("model.h5")
predictions = custom_Model.predict_generator(generator=test_generator)
y_pred = np.argmax(predictions, axis=1)
y_real = test_generator.classes
matc = confusion_matrix(y_real, y_pred)
print(metrics.classification_report(y_real, y_pred, digits=4))
df_cm = pd.DataFrame(matc, index=names, columns=names)
chart = sns.heatmap(df_cm,cmap='Accent', annot=True)
chart.set(xlabel='Verdaderos', ylabel='Predicciones')
plt.show()
