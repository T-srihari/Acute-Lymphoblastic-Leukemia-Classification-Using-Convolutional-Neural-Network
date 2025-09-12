import re
import os
from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import pandas as pd
import csv
from django.conf import settings


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, 'Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def training(request):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    import os
    import glob as gb
    import tensorflow as tf
    import keras
    import cv2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.utils import to_categorical

    code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}

    def getcode(n):
        for x, y in code.items():
            if n == y:
                return x

    s = 224
    import cv2
    from tqdm import tqdm
    import os

    X_train = []
    y_train = []
    for img in tqdm(os.listdir(r'media\Original\Benign')):
        image = cv2.imread(os.path.join(r'media\Original\Benign', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_train.append(list(image_array))
        y_train.append(code['Benign'])

    for img in tqdm(os.listdir(r'media\Original\Early')):
        image = cv2.imread(os.path.join(r'media\Original\Early', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_train.append(list(image_array))
        y_train.append(code['Early'])

    for img in tqdm(os.listdir(r'media\Original\Pre')):
        image = cv2.imread(os.path.join(r'media\Original\Pre', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_train.append(list(image_array))
        y_train.append(code['Pre'])

    for img in tqdm(os.listdir(r'media\Original\Pro')):
        image = cv2.imread(os.path.join(r'media\Original\Pro', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_train.append(list(image_array))
        y_train.append(code['Pro'])

    plt.figure(figsize=(20, 20))
    for n, i in enumerate(list(np.random.randint(0, len(X_train), 36))):
        plt.subplot(6, 6, n + 1)
        plt.imshow(X_train[i])
        plt.axis('off')
        plt.title(getcode(y_train[i]))

    len(X_train)

    X_test = []
    y_test = []
    for img in tqdm(os.listdir(r'media\Original\Benign')):
        image = cv2.imread(os.path.join(r'media\Original\Benign', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_test.append(list(image_array))
        y_test.append(code['Benign'])

    for img in tqdm(os.listdir(r'media\Original\Early')):
        image = cv2.imread(os.path.join(r'media\Original\Early', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_test.append(list(image_array))
        y_test.append(code['Early'])

    for img in tqdm(os.listdir(r'media\Original\Pre')):
        image = cv2.imread(os.path.join(r'media\Original\Pre', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_test.append(list(image_array))
        y_test.append(code['Pre'])

    for img in tqdm(os.listdir(r'media\Original\Pro')):
        image = cv2.imread(os.path.join(r'media\Original\Pro', img), 1)
        image_array = cv2.resize(image, (s, s))
        X_test.append(list(image_array))
        y_test.append(code['Pro'])

    plt.figure(figsize=(20, 20))
    for n, i in enumerate(list(np.random.randint(0, len(X_test), 36))):
        plt.subplot(6, 6, n + 1)
        plt.imshow(X_test[i])
        plt.axis('off')
        plt.title(getcode(y_test[i]))

    da = []
    for i, j in zip(X_train, y_train):
        da.append([i, j])

    import random
    random.shuffle(da)

    len(da)

    X = []
    y = []
    for img, label in da:
        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True)
    print(xtest.shape)
    print(xtrain.shape)

    KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)),
        keras.layers.Conv2D(150, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D(4, 4),
        keras.layers.Conv2D(120, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(50, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D(4, 4),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(4, activation='softmax'),
    ])

    m_model = tf.keras.applications.vgg19.VGG19()
    model1 = tf.keras.models.Sequential()
    for layer in m_model.layers[:-1]:
        model1.add(layer)
    for layer in model1.layers:
        layer.trainable = False
    model1.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Model Details are : ')
    print(model1.summary())

    history = model1.fit(xtrain, ytrain, batch_size=16,
                         verbose=1,
                         validation_data=(xtest, ytest), epochs=10)

    # model1.save('New_Acute_Lymphoblastic_Lukemia_Model.h5')

    y_pred = model1.predict(xtest)
    print('Prediction Shape is {}'.format(y_pred.shape))

    import matplotlib.pyplot as plt
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'test'], loc='lower right')
    plt.tight_layout()

    return render(request, 'users/training.html', {'accuracy': accuracy, 'val_accuracy': val_accuracy})


def prediction(request):
    import numpy as np
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model
    import io
    from PIL import Image
    path = os.path.join(settings.MEDIA_ROOT, 'Acute_Lymphoblastic_Lukemia_Model.h5')
    # Load the trained model
    model = load_model(path)

    # Define the label mappings
    code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}
    reverse_code = {v: k for k, v in code.items()}

    def get_class_label(prediction):
        return reverse_code[np.argmax(prediction)]

    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image
        img = request.FILES['image']

        # Convert the image to a PIL Image
        img = Image.open(io.BytesIO(img.read()))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = get_class_label(predictions[0])

        return render(request, 'users/prediction.html', {'predicted_class': predicted_class})

    return render(request, 'users/prediction.html')
