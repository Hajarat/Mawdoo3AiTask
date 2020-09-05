# Mawdoo3 AI Task

### Data used: (dev-clean, train-clean-100) folders from the OpenSLR dataset: http://www.openslr.org/12/

### Work done on: "train_openSLR.ipynb" & "train_openSLR_2.ipynb" using pytorch

The data extracted consists of two separate folders (training, validation/testing), within these folders data is stored in the following manner:

![Directory Tree]()

The first subtree/directory represents the user Id, the second subtree represents the session belonging to the parent user Id, the files (.flac) within the session subtree represent short recordings (<15s), all of which record the user actively reading in english.

The training section of the extracted data consists of about 100 hours of such recordings evenly split between male/female readers (50.20 hours/ 50.38 hours respectively).

There exists a "speaker.txt file" packaged with the zip files containing the aforementioned extracted directories which contains the following information:
- Id of each user
- The subset of the dataset this row belongs to (we use dev-clean and train-clean-100 only)
- The number of minutes this user speaks (within the provided subset)
- The sex of that user

Note: with the help of oscarknagg (github.com/oscarknagg) I fixed some issues within the speaker.txt file where some users where misclassified in their gender.

### Pre-processing
First Model:
with the help of oscarknagg, preprocessing of the directory into workable torch tensors was already available, as will be indicated in the workbook. The directories were accessed sequentially using the os.walk() method in python which allowed the files to be indexed by Id to filepath and Id to sex, which in turn allowed the creation of the torch dataset that is then loaded onto the torch dataloader for training.

Second Model:
Features were extracted as indicated by the "Extracted Features" section below. These features then populated the torch dataset which the dataloader loads in preparation for training.

### Visualisation
Simple initial visualisations were made on ...

### Extracted Features
For the second model, a mfcc tranformation was done on each audio file, creating a 4x10 matrix of float values. These features were saved and later on extracted to perform 2D convolutions on a CNN.

### Model
First Model:

Second Model:

### Results
First Model:

Second Model:

Comparison:
