ReadME.txt instructions
********************************************************************
Name: Transforming How We Diagnose Heart Disease
Author: Henrik von Kleist
********************************************************************
     
DESCRIPTION:

This README file explains how to reproduce the results from our 
approach for the second annual data science Bowl.

********************************************************************
First of all, the files have to be downloaded from the Kaggle 
website at: 
https://www.kaggle.com/c/second-annual-data-science-bowl/data

Secondly, the data has to be preprocessed. The preprocessing step
transforms the DICOM files to numpy vectors of the pictures.
We also produce a vector containing the ids of the patients, the 
training labels and the pixel_spacing information.
This is done in localisation.py. An older version without the
extraction of the heart location can be found in data.py.

Afterwards, we perform the training with train.py. The model of the 
Neural Network is build in model.py.

Lastly, we evaluate the performence with Submission.py which uses 
the neural network weights produced by train.py to predict the labels
for the test set and evaluates the CRPS error of the predictions. We
compute the CRPS score once for every SAX slice individually and once
we average over all Sax slices of the patient. 

We do not produce and output file, since we can't upload solutions, 
because the challenge has ended.





********************************************************************


