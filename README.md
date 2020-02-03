# Deep Learning approach to measure cardiac volumes
Author: Henrik von Kleist
Implementation of a Deep Learning algorithm to find the cardiac volumes 
and derive the ejection fraction from cardiac magnetic resonance images (MRI)
as part of the Kaggle competition: [the second annual data science bowl](https://www.kaggle.com/c/second-annual-data-science-bowl/overview/description)
For details of the project, also have a look at the [project report.

     
### HOW TO SET UP:
1. Download the files from the [Kaggle website](https://www.kaggle.com/c/second-annual-data-science-bowl/data)
2. Preprocess the data
* transform the DICOM files to numpy vectors of the pictures
* produce a vector containing the ids of the patients, the training labels and the pixel_spacing information.
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


