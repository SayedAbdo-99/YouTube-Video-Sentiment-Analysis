# import the necessary packages
from skimage import feature
import glob,os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder as BCF
import itertools
import pickle

def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def read_get_features(path,lable,IMG_SIZE,featureType):
    data = []
    lables=[]
    # loop over the image paths in the training set
    print(path)
    for imagePath in glob.glob(path):   #,recursive=True
    	# load the image, convert it to grayscale, and detect edges
        image = cv2.imread(imagePath)
        img = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    	#_____________________________ features extraction
        if featureType =='hog':
            # Histogram of Oriented Gradients from the logo
            image_feat = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),
        		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        elif featureType == 'gabor':
            image_feat=gabor_filter(img)
        else :
            print('Feature Extraction Type error')
            
        # update the data and labels
        data.append(image_feat)
        lables.append(lable)

    return np.array(data, 'float64'),np.array(lables)

def features(words):
	words = word_tokenize(words)

	scoreF = BigramAssocMeasures.chi_sq

	#bigram count
	n = 150

	bigrams = BCF.from_words(words).nbest(scoreF, n)

	return dict([word,True] for word in itertools.chain(words, bigrams))


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import pickle

def load_train_test_data():
    #____________________________________   Load Data as features in matrix _______________
    print('_______________________Load Data as features ... ')
    pos_sen =open("dataset/positive.txt", 'r', encoding = 'latin-1').read()
    neg_sen =open("dataset/negative.txt", 'r', encoding = 'latin-1').read()
    class1_feat = np.array([features(words) for words in pos_sen.split('\n')])
    class2_feat = np.array([features(words) for words in neg_sen.split('\n')])
    print(class1_feat)
    print(len(class1_feat))
    lab1=[0]*len(class1_feat);print('class 1 count : ',len(lab1));
    lab2=[1]*len(class2_feat);print('class 2 count : ',len(lab2));
    
    #concatenate features in one matrix and lables in matrix
    all_feat=np.concatenate((np.array(class1_feat),np.array(class2_feat)),axis=0) 
    all_lables=np.concatenate((np.array(lab1),np.array(lab2)),axis=0) 
    
    #____________________________________Splitting Data_______________
    # Split dataset into training set and test set
    Train,Test, Label_train, Label_test = train_test_split(all_feat,all_lables, test_size=0.3, random_state=1) # 70% training and 30% test
    print (Train)
    return Train,Test, Label_train, Label_test
    
def classification(svm_Kernal,adaboost,Bagging,knn_N_neighbors,modelVersion):
    Train,Test, Label_train, Label_test=load_train_test_data()
    #____________________________________Generating Model using SVM ______________
    print('SVM calssification ...')
    #Create a svm Classifier 
    if svm_Kernal=='linear':
        svmModel = LinearSVC(random_state=42)
    elif svm_Kernal=='rbf':
        svmModel = svm.SVC(kernel='rbf',gamma='scale')
    elif svm_Kernal=='poly':
        svmModel = svm.SVC(kernel='poly',gamma='auto')
    else :
        print('SVM Kernal error')
    
    #apply Bagging on svm
    if adaboost :
        svmModel = BaggingClassifier(base_estimator=svmModel, n_estimators=31, random_state=34)
    if Bagging :
        svmModel = AdaBoostClassifier(svmModel,n_estimators=50, learning_rate=1.0, algorithm='SAMME')
        
    
    #Train the model using the training sets
    svmModel.fit(Train,Label_train)
    #Predict the response for test dataset
    predictSvmModel = svmModel.predict(Test)
    # save the model to disk
    # SAVE IN FILE TO AVOID TRAIINING THE DATA AGAIN
    save_doc = open('models/svmModel'+modelVersion +'.pickle', 'wb')
    pickle.dump(svmModel, save_doc)
    save_doc.close();

    #____________________________________Generating Model using KNN ______________
    print('KNN calssification ...')
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=knn_N_neighbors)
    #Train the model using the training sets
    knn.fit(Train,Label_train)
    #Predict the response for test dataset
    predictknn = knn.predict(Test)
    # save the model to disk
    # SAVE IN FILE TO AVOID TRAIINING THE DATA AGAIN
    save_doc = open('models/KnnModel'+modelVersion +'.pickle', 'wb')
    pickle.dump(knn, save_doc)
    save_doc.close();
    
    #____________________________________Evaluating the Model_________
    print('_________________Evaluating ...')
    #Import scikit-learn metrics module for accuracy calculation
    # Model Accuracy: how often is the classifier correct?
    accsvm=metrics.accuracy_score(Label_test, predictSvmModel)
    print("Accuracy of svm :",accsvm)# Evaluate SVM
    # Model Accuracy, how often is the classifier correct?
    accknn=metrics.accuracy_score(Label_test, predictknn)
    print("Accuracy of Knn:",accknn)# Evaluate knn
    get_current_time()

classification('linear',False,False,5,'1.0')

'''
print('______________________________________Experiment 1  ______________________________________')
print('________________________Dataset (Autti):[ class1: cars vs. class2: non_cars]______________')
print('____________________________________Experiment vs-1.0 : _____________________________________')
classification('dataset/autti/cars/**/*.png','dataset/autti/non_cars/**/*.png',20,'hog','rbf',False,False,5,'1.0')

print('____________________________________Experiment vs-1.2 : _____________________________________')
classification('dataset/autti/cars/**/*.png','dataset/autti/non_cars/**/*.png',75,'hog','linear',True,True,11,'1.1')

print('____________________________________Experiment vs-1.1 : _____________________________________')
classification('dataset/autti/cars/**/*.png','dataset/autti/non_cars/**/*.png',50,'hog','poly',False,False,7,'1.2')

print('____________________________________Experiment vs-1.3 : _____________________________________')
classification('dataset/autti/cars/**/*.png','dataset/autti/non_cars/**/*.png',20,'gabor','poly',False,True,9,'1.3')

print('____________________________________Experiment vs-1.4 : _____________________________________')
classification('dataset/autti/cars/**/*.png','dataset/autti/non_cars/**/*.png',70,'gabor','rbf',False,False,5,'1.4')

print('____________________________________Experiment vs-1.5 : _____________________________________')
classification('dataset/autti/cars/**/*.png','dataset/autti/non_cars/**/*.png',75,'gabor','linear',True,True,11,'1.5')
'''

'''
print('===========================================================================================')
print('______________________________________Experiment 2  ______________________________________')
print('___Dataset (vidtimit):[ class1: Person1 vs. class2: Person2 vs. class3: Person3 vs. class3: Person4]____')
print('____________________________________Experiment vs-2.0 : _____________________________________')
classification('dataset/vidtimit/person1/video/**/*','dataset/vidtimit/person2/video/**/*','dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person4/video/**/*',70,'hog','rbf',False,False,11,'1.0')

print('____________________________________Experiment vs-2.1 : _____________________________________')
classification('dataset/vidtimit/person2/video/**/*','dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person4/video/**/*','dataset/vidtimit/person5/video/**/*',25,'hog','poly',True,True,5,'1.1')

print('____________________________________Experiment vs-2.2 : _____________________________________')
classification('dataset/vidtimit/person1/video/**/*','dataset/vidtimit/person2/video/**/*','dataset/vidtimit/person6/video/**/*','dataset/vidtimit/person7/video/**/*',50,'hog','linear',False,True,7,'1.2')

print('____________________________________Experiment vs-2.3 : _____________________________________')
classification('dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person4/video/**/*','dataset/vidtimit/person6/video/**/*','dataset/vidtimit/person7/video/**/*',65,'gabor','poly',True,True,5,'1.3')

print('____________________________________Experiment vs-2.4 : _____________________________________')
classification('dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person4/video/**/*','dataset/vidtimit/person7/video/**/*','dataset/vidtimit/person8/video/**/*',35,'gabor','rbf',True,False,9,'1.4')

print('____________________________________Experiment vs-2.5 : _____________________________________')
classification('dataset/vidtimit/person1/video/**/*','dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person5/video/**/*','dataset/vidtimit/person7/video/**/*',50,'gabor','linear',True,True,11,'1.5')

print('____________________________________Experiment vs-2.6 : _____________________________________')
classification('dataset/vidtimit/person5/video/**/*','dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person6/video/**/*','dataset/vidtimit/person8/video/**/*',32,'gabor','poly',False,False,11,'1.6')

print('____________________________________Experiment vs-2.7 : _____________________________________')
classification('dataset/vidtimit/person2/video/**/*','dataset/vidtimit/person3/video/**/*','dataset/vidtimit/person6/video/**/*','dataset/vidtimit/person8/video/**/*',75,'gabor','rbf',False,False,11,'1.7')
'''
'''
# load the model from disk
Train,Test, Label_train, Label_test=load_train_test_data('dataset/autti/cars/GTI_Left/*.png','dataset/autti/non_cars/Extras/*.png',100,'hog')
loaded_model = pickle.load(open('models/cars_vs_noncars/KnnModel1.7.sav', 'rb'))
result = loaded_model.predict(Test)
accsvm=metrics.accuracy_score(Label_test, result)
print("Accuracy of svm :",accsvm)
print(result)

vs = cv2.VideoCapture('challenge_video.mp4')
writer = None
(W, H) = (None, None)
 
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
 
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
 
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
        # clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean
    	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
 
	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
    	# draw the activity on the output frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)
 
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
 
	# write the output frame to disk
	writer.write(output)
 
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
'''