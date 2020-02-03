# Deep Learning approach to measure cardiac volumes
Author: Henrik von Kleist, Jacob Johnson <br>
Implementation of a Deep Learning algorithm to find the cardiac volumes 
and derive the ejection fraction from cardiac magnetic resonance images (MRI)
as part of the Kaggle competition: [the second annual data science bowl](https://www.kaggle.com/c/second-annual-data-science-bowl/overview/description)
For details of the project, also have a look at the [project report.

     
### HOW TO SET UP:
1. Download the files from the [Kaggle website](https://www.kaggle.com/c/second-annual-data-science-bowl/data)
* adapt data path in localisation.py
2. Preprocess the data (localisation.py)
* transform the DICOM files to numpy vectors of the pictures
* produce a vector containing the ids of the patients, the training labels and the pixel_spacing information.
* older version without the extraction of the heart location can be found in data.py.
3. perform the training (train.py)
* model is built in model.py
4. evaluate the performance (submission.py)
* uses the neural network weights produced by train.py to predict the labels
for the test set
* evaluates the CRPS error of the prediction
* compute CRPS score once for every SAX slice individually
5. Visualize and analyse results with Data project visualization.ipynb

Use test-notebook.ipynb to run everything



