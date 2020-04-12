Phase 1:

TO DO:

1)You told some corrections in the dataset(like ground truth,frames(no of frames eyes closed))

2)Since the accuracy is not fine You said to increase the accuracy,by increasing dataset.

3)To do n-fold(10-fold),to get precise accuracy.

4)Send the video samples through dropbox link.

--------------------------------------------------------------------------------------------------------------------------------

COMPLETED:

1)We increase the dataset.(84 samples are there)

2)We have rectified the corrections in the dataset.

3)As before we use linear model for calculating accuracy,the accuracy varies. Since it is a supervised learning SVM works pretty well.
 while we use SVM for calculating accuracy strongly we get accuracy ab0ve 80 percent.

4)10-fold is done as well.The result of n-fold is given below.(Accuracy 88-90)

------------------------------------------------------------------------------------------------------------------------------

MODULES NEEDED:

1)face_utils
2)imutils
3)dlib
4)opencv2
5)numpy
6)pandas


HOW TO RUN:

1)train_phase1.py

	The train_phase1.py will analyse the video samples and create a dataset(ie.dataset_phase1.csv)

OUTPUT:
	Dataset Created Successfully!!!

	dataset_phase1.csv

	I have included the dataset.csv file also in the github link.

```````````````````````````````````````````````````````````````````````````````````

2)SVM_accuracy_phase1.py

	It will give the 10-fold result of accuracy.
	Each fold accuracy are seperately displayed and finally the average accuracy is also calculated.

OUTPUT:


Total number of rows in the dataset :  83

Accuracy for Test[ 0 - 7 ] and Train[ 8 - 82 ]			: 0.75
Accuracy for Test[  8 - 15 ] and Train[ 0 - 7 ]&[ 16 - 82 ]	: 0.875
Accuracy for Test[  16 - 23 ] and Train[ 0 - 15 ]&[ 24 - 82 ]	: 0.875
Accuracy for Test[  24 - 31 ] and Train[ 0 - 23 ]&[ 32 - 82 ]	: 1.0
Accuracy for Test[  32 - 39 ] and Train[ 0 - 31 ]&[ 40 - 82 ]	: 0.875
Accuracy for Test[  40 - 47 ] and Train[ 0 - 39 ]&[ 48 - 82 ]	: 0.875
Accuracy for Test[  48 - 55 ] and Train[ 0 - 47 ]&[ 56 - 82 ]	: 0.75
Accuracy for Test[  56 - 63 ] and Train[ 0 - 55 ]&[ 64 - 82 ]	: 1.0
Accuracy for Test[  64 - 71 ] and Train[ 0 - 63 ]&[ 72 - 82 ]	: 0.875
Accuracy for Test[ 75 - 82 ] and Train[ 0 - 74 ]		: 1.0


All Accuracies:
 [0.75, 0.875, 0.875, 1.0, 0.875, 0.875, 0.75, 1.0, 0.875, 1.0]


Average accuracy: 0.8875

````````````````````````````````````````````````````````````````````````````
Now the accuracy falls between 87-90
`````````````````````````````````````````````````````````````````````````````


3)sleepdetection.py

	it is a realtime application.It will alert the user when the eyes closed for more than 20 frames.


````````````````````````````````````````````````````````````````````````````````````````````````````````````````

