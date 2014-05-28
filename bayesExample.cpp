#include "engine.h"
#include "ml.h"
#include <iostream>
using namespace std;
using namespace cv;

int teste(int argc, char **argv)
{
//initModule_nonfree();

Ptr<FeatureDetector> features = FeatureDetector::create("SIFT");
Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("SIFT");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

//defining terms for bowkmeans trainer
TermCriteria tc(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10, 0.001);
int dictionarySize = 100;
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);

BOWImgDescriptorExtractor bowDE(descriptor, matcher);

//**creating dictionary**//

Mat features1, features2;
Mat img = imread("c:\\1.jpg", 0);
Mat img2 = imread("c:\\2.jpg", 0);
vector<KeyPoint> keypoints, keypoints2;
features->detect(img, keypoints);
features->detect(img2,keypoints2);
descriptor->compute(img, keypoints, features1);
descriptor->compute(img2, keypoints2, features2);
bowTrainer.add(features1);
bowTrainer.add(features2);

Mat dictionary = bowTrainer.cluster();
bowDE.setVocabulary(dictionary);

//**dictionary made**//

//**now training the classifier**//

Mat trainme(0, dictionarySize, CV_32FC1);
Mat labels(0, 1, CV_32FC1); //1d matrix with 32fc1 is requirement of normalbayesclassifier class

Mat bowDescriptor, bowDescriptor2;
bowDE.compute(img, keypoints, bowDescriptor);
trainme.push_back(bowDescriptor);
float label = 1.0;
labels.push_back(label);
bowDE.compute(img2, keypoints2, bowDescriptor2);
trainme.push_back(bowDescriptor2);
labels.push_back(label);

NormalBayesClassifier classifier;
classifier.train(trainme, labels);

//**classifier trained**//

//**now trying to predict using the same trained classifier, it should return 1.0**//

Mat tryme(0, dictionarySize, CV_32FC1);
Mat tryDescriptor;
Mat img3 = imread("2.jpg", 0);
vector<KeyPoint> keypoints3;
features->detect(img3, keypoints3);
bowDE.compute(img3, keypoints3, tryDescriptor);
tryme.push_back(tryDescriptor);

cout<<classifier.predict(tryme);
waitKey(0);



return 0;
}
