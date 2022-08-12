#----------------***************************!중요중요중요!****************----------------------------------------------
# 파이썬 파일(Waste Classification.py)과 '재활용분류' 데이터셋을 같은 경로에 위치시키고
# 상대 참조이기 때문에 set console working directory 설정
#----------------***************************!중요중요중요!****************----------------------------------------------
# Python 3.9.7 version
# tensorflow 2.5.0 version
# keras 2.5.0 version

import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm  # 진행상황 파악 가능 #pip install tqdm
import cv2 #pip install opencv-python
import pandas as pd
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from glob import glob # glob 모듈의glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다
import warnings
warnings.filterwarnings('ignore')

# 파이썬 파일과 '재활용분류' 데이터셋을 같은 경로에 위치시키고
# 상대 참조이기 때문에set console working directory 설정
for dirname, _, _ in os.walk('./재활용분류'):
        print(dirname)



# train/val/test 나누기
# ratio 파라미터에 원하는 (train, validation, test) 비율을 입력, 6:2:2 
import splitfolders #pip install split-folders
splitfolders.ratio("./재활용분류", output="output", seed=1333, ratio=(.6, .2, .2)) # 'output'이름으로 폴더 생김
# seed값을 바꿔가며 재 스플릿: 1337, 1336, 1335, 1334, 1333

train_path = "./output/train" # train데이터 경로지정
test_path = "./output/test" # test데이터 경로지정
val_path = "./output/val" # validation데이터 경로지정


x_data = [] # 이미지 넣을 리스트
y_data = []  # 라벨 넣을 리스트

for category in glob(train_path+'/*'): 
    for file in tqdm(glob(category+'/*')):
        img_array=cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array) 
        y_data.append(category.split("/")[-1])
        
data=pd.DataFrame({'image': x_data,'label': y_data})

print(data.shape) # train 데이터 개수 -> 15,045개 

from collections import Counter
Counter(y_data) # 라벨이 'O', 'R' 두개인지 확인

colors = ['#a0d157','#c48bb8']
plt.pie(data.label.value_counts(),startangle=90,explode=[0.05,0.05],autopct='%0.2f%%',
labels=['Organic', 'Recyclable'], colors= colors,radius=2)
plt.show() # 'O'클래스가 56%, 'R'클래스가 44% 있음

# train 데이터 15,045개 중 9개 랜덤으로 시각화 
# 이미지 크기가 서로 다름.
# 대부분 이미지 가운데에 위치
# O -> Organic
# R -> Recyclable
plt.figure(figsize=(20,15))
for i in range(9):
    plt.subplot(4,3,(i%12)+1)   # nrows=4, ncols=3 , index=(i%12)+1
    index=np.random.randint(15045)
    plt.title('This image is of {0}'.format(data.label[index][-1]),fontdict={'size':20,'weight':'bold'})
    plt.imshow(data.image[index])
    plt.tight_layout()
plt.show()

# 클래스 이름이 'O', 'R'외에 있는지 확인(2개가 맞는지 확인)    
className = glob(train_path + '/*' )
numberOfClass = len(className)
print("Number Of Class: ",numberOfClass)

model = Sequential()
# 필터 16개, 필터사이즈(3,3), input사이즈 224, 활성화함수 'ReLU', padding값을 'same'으로 하면 입력과 출력의 크기가 같아짐
model.add(Conv2D(filters=16,kernel_size=(3,3),input_shape = (224,224,3),padding='same')) # Conv1
model.add(BatchNormalization()) # 배치 정규화를 하면 학습 속도가 빨라짐
model.add(Activation('relu')) # 배치 정규화를 하고 활성화 함수를 써야함
model.add(MaxPooling2D(pool_size=(2,2))) # 최대풀링레이어

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))  # Conv2
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))  # Conv3
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))  # Conv4
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # 3차원인 이미지 1차원으로 축소

model.add(Dense(256)) # Fully connected 1
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) # 배치정규화를 하고 드롭아웃까지 하면 성능이 더 좋아진다는 글을 봤습니다./ 성능이 약간 좋아짐.
# https://gaussian37.github.io/dl-concept-order_of_regularization_term/

model.add(Dense(64)) # Fully connected 2
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) 
model.add(Dense(2 ,activation=('sigmoid'))) # output

# 이진 분류이기때문에 'binary_crossentropy'
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

batch_size = 40  # gpu 메모리 제한때문에 큰 값 설정 안됨.(큰 값과 비교해보고 싶었는데 하지 못했습니다...)
epoch = 10 # 10초과로 설정하면 train정확도만 올라가는 과적합이 생김

# Rescaling
train_datagen = ImageDataGenerator(rescale= 1./255) # 값을 0과 1 사이로 변경
valid_datagen = ImageDataGenerator(rescale = 1./255) # 값을 0과 1 사이로 변경
test_datagen = ImageDataGenerator(rescale= 1./255) # 값을 0과 1 사이로 변경

train_generator = train_datagen.flow_from_directory(
        train_path, 
        target_size= (224,224), # 이미지크기를 (224,224)로 변환
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

valid_generator = valid_datagen.flow_from_directory(
        val_path,
        target_size = (224,224), # 이미지크기를 (224,224)로 변환
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size= (224,224), # 이미지크기를 (224,224)로 변환
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")
                            
hist = model.fit_generator( # 모델 fitting
        generator = train_generator,
        epochs=epoch, 
        validation_data = valid_generator,
        shuffle=True)

# 정확도
scores = model.evaluate_generator(train_generator,len(train_generator))
print('Training Accuracy: %.2f%%\n' %(scores[1]*100)) # train 정확도

scores = model.evaluate_generator(test_generator,len(test_generator))
print('Testing Accuracy: %.2f%%\n' %(scores[1]*100)) # test 정확도

# 모델 구조 확인
model.summary()

# 정확도 진행 시각화
plt.figure(figsize=[10,6])
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# loss 진행 시각화
plt.figure(figsize=(10,6))
plt.plot(hist.history['loss'], label = "Train loss")
plt.plot(hist.history['val_loss'], label = "Validation loss")
plt.legend()
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 결과(y와 y햇의 결과 확인)
test_x, test_y = test_generator.__getitem__(1)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

preds = model.predict(test_x) # fitting한 모델 test셋에 적용

plt.figure(figsize=(16, 16))
for i in range(16): # 예측한 데이터 16개 추출
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
