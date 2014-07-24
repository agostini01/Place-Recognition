Project:
Place Recognition

Created By:
Nicolas Bohm Agostini
n_b_agostini@gmail.com

In:
28 May 2014

Will have to install / use:
C++11
Boost
OPENCV-2.4.8 or above 
Gnuplot

__________________________________________________________________________

About this project:
It is meant to perform place recognition through different classification algorithms.
__________________________________________________________________________

The training procedure happens in 3 stages:

1) SIFT or SURF features are extracted

2) Using Bag of Visual Words the features are clustered. This process provide descriptors of smaller size which are going to improve speed during the classifier training and classification process.

3) The classifier is trained and can be used for classification
__________________________________________________________________________

The classification happens in 2 stages:

1) Descriptors are extracted using the same process the training stage was used (training and test descriptors have to be consistent)

2) Using the already trained classifier with the test image descriptors classification is performed.
__________________________________________________________________________

How to run:
1) Have a proper folder tree created. The project does not have the power to create folders when it tries to write files.  So the folders have to be created as follow. (NOTE: the project was initially meant to run in linux, so all the paths use linux folder/path syntax eg.: ‘/’ - to run in windows changes in code have to be made.)
- ../ProjectFolder/code (to store the code)
- ../ProjectFolder/build (to store the exe file)
- ../ProjectFolder/data (folder for training or test images)
- ../ProjectFolder/data/train (Create folders and put images inside. Each folder created here will correspond to a new class)
- ../ProjectFolder/data/test (to store test images - each folder correspond to a new test set)
- ../ProjectFolder/output (to store loadable files. Eg.: Descriptors matrices, Trained Classifiers, Results/

ATTENTION: If any of this folders has not been created the files are not going to be read/write leading for the project malfunction.

2) Have the right libs installed and linked to the project:
- Make use of C++11
- Make use of Boost
- Make use of OPENCV-2.4.8 or above (it has not been tested with previous versions)
- Install Gnuplot - this feature can be disable removing the gnuplot part from the code
-------------------------------------------------------------------------
ex.:
USING QT-CREATOR the above is performed with those lines in the .pro file.

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += /usr/local/include/opencv \
    /usr/include/boost
LIBS += -L/usr/local/lib \
    -lopencv_core \
    -lopencv_video \
    -lopencv_highgui \
    -lopencv_features2d \
    -lopencv_imgproc \
    -lopencv_ml \
    -lopencv_calib3d \
    -lopencv_objdetect \
    -lopencv_contrib \
    -lopencv_legacy \
    -lopencv_flann \
    -lopencv_nonfree \
    -lboost_system \
    -lboost_iostreams \
    -lboost_filesystem
-------------------------------------------------------------------------
USING CMAKE on  Linux
Everything is done in the CMakeLists.txt (the script has to be probably adapted a little bit to run on Windoes/Mac), the user just have to run the following commands:
in the PROJECT ROOT DIRECTORY:
cd build-PlaceRecognition-Desktop_CMAKE
cmake ../code -DOpenCV_LIBS=/usr/local/lib -DBoost_LIBRARIES=/usr/local/lib
make

to run:
./PlaceRecognitionCmake

__________________________________________________________________________

Output Files:
cvVocabulary.xml - holds the vocabulary after the Bag of visual Words extraction. Necessary to acquire new descriptors.
trainingSet.xml - holds a Mat file with the descriptors of the training set
trainingSetLabels.xml - holds a Mat File with the labels of the training set
svm_crosvalidation_x.xml - holds the svm classifier parameters for crossvalidade fold X
svm_filename.xml - holds the svm classifier parameters




