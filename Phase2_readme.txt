PHASE 2:

THINGS YOU TOLD TO DO:

1)Increase the attributes or features of the dataset like mouth, jaw, eye redness.

2)With  that attribute predict accuracy.

3)Implement it in the realtime application.

--------------------------------------------------------------------------------------------------------------

COMPLETED:

1)Maps the mouth and jaw to analyse it in the realtime application.

2)Now the complete realtime application works as:
	
	a)if subject close eyes for more the 3 sec(initial timing for warning) it alerts.
	
	b)if subject yawns it make count of the yawn.
	
	c)After 2 yawns ,it reduces the initial timing for warming(ie:3 sec) by 2 frames for each yawn.

	d)If the subject yawns more than 5 times, it displays a safety message and warn the subject to go to sleep.

	e)if the head is not straight for 3sec(initial timing for warning) it alerts.It checks for head position in all possible directions.

3)Implement this new features we created an alternated dataset(dataset_phase2.csv) with some additional features(columns).

4) Another 10-fold SVM accuracy program for the new dataset to compute accuracy.

---------------------------------------------------------------------------------------------------------------------------

HOW TO RUN:

1)train_phase2.py

	The train_phase2.py will analyse the video samples and create a new dataset(ie.dataset_phase2.csv)

OUTPUT:
	Dataset Created Successfully!!!

	dataset_phase2.csv

	I have included the dataset.csv file also in the github link.

```````````````````````````````````````````````````````````````````````````````````

2)SVM_accuracy_phase2.py

	It will give the 10-fold result of accuracy.
	Each fold accuracy are seperately displayed and finally the average accuracy is also calculated.

OUTPUT:

Total number of rows in the dataset :  83

Accuracy for Test[ 0 - 7 ] and Train[ 8 - 82 ]			: 1.0
Accuracy for Test[  8 - 15 ] and Train[ 0 - 7 ]&[ 16 - 82 ]	: 0.875
Accuracy for Test[  16 - 23 ] and Train[ 0 - 15 ]&[ 24 - 82 ]	: 1.0
Accuracy for Test[  24 - 31 ] and Train[ 0 - 23 ]&[ 32 - 82 ]	: 0.875
Accuracy for Test[  32 - 39 ] and Train[ 0 - 31 ]&[ 40 - 82 ]	: 0.875
Accuracy for Test[  40 - 47 ] and Train[ 0 - 39 ]&[ 48 - 82 ]	: 1.0
Accuracy for Test[  48 - 55 ] and Train[ 0 - 47 ]&[ 56 - 82 ]	: 1.0
Accuracy for Test[  56 - 63 ] and Train[ 0 - 55 ]&[ 64 - 82 ]	: 0.875
Accuracy for Test[  64 - 71 ] and Train[ 0 - 63 ]&[ 72 - 82 ]	: 0.75
Accuracy for Test[ 75 - 82 ] and Train[ 0 - 74 ]		: 0.875


All Accuracies:
 [1.0, 0.875, 1.0, 0.875, 0.875, 1.0, 1.0, 0.875, 0.75, 0.875]


Average accuracy: 0.9125

````````````````````````````````````````````````````````````````````````````
Now the accuracy falls between 88-92
`````````````````````````````````````````````````````````````````````````````

3)sleepdetection.py

	it is a complete realtime application with all the features .It will alert the user when the conditions satsfied given above.


````````````````````````````````````````````````````````````````````````````````````````````````````````````````

