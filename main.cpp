/*

 * main.cpp
 *
 *  Created on: 06/03/2014
 *      Author: agostini
 *
 http://stackoverflow.com/questions/11602577/bag-of-words-training-and-testing-opencv-matlab
 To describe a proper bag of visual words the following steps will
 be follow

 1 - Extract the SIFT local feature vectors from your set of images;
 2 - Put all this local feature vectors into a single set. At this
 point you don't even need to store from which image each local
 feature vector was extracted;
 3 - Apply a clustering algorithm (e.g. k-means) over the set of
 local feature vectors in order to find centroid coordinates and
 assign an id to each centroid. This set of centroids will be your
 vocabulary;
 4 - The global feature vector will be a histogram that counts how
 many times each centroid occurred in each image. To compute the
 histogram find the nearest centroid for each local feature vector.
 */

#include "engine.h"
#include "examples.h"
#include "DataParser.h"
#include "datahandler.h"
#include <map>
#include <list>
#include <iterator>


#include <iostream>

using namespace std;
using namespace cv;

string makePathToImage(const std::string& pathToCategory, const std::string& imageName)
{
    std::string pathToImage;
    pathToImage.clear();
    pathToImage.append(pathToCategory);
    pathToImage.append("/");
    pathToImage.append(imageName);

    return pathToImage;
}

int main(int argc, char **argv)
{

    cout<<"Starting Application"<<endl;
//    examples example;
//    example.gnuplotHistExample();
//    example.gnuplotExample();
//    example.historiogramCalculationExample(argv[2]);
//    example.bowExtractorExample(argv[1], argv[2]);

    dataHandler handler1;
    std::map<std::string, std::list<std::string> > trainClassAndFiles=handler1.buildDataFiles("../data/train/");
    std::map<std::string, std::list<std::string> > testClassAndFiles=handler1.buildDataFiles("../data/test/");

    if (true)
    {
        Mat img1 = imread(argv[1]);
        Mat img2 = imread(argv[2]);
        Mat imgToAdd;
        cv::Mat* imgPointer;
        dataParser::DataParser parser;
        //Creates Detectors Extractors Matchers
        cv::Ptr<cv::FeatureDetector> featureDetector =
                cv::FeatureDetector::create("SURF");
        cv::Ptr<cv::DescriptorExtractor> descExtractor =
                cv::DescriptorExtractor::create("SURF");
        cv::Ptr<cv::DescriptorMatcher> descMatcher =
                cv::DescriptorMatcher::create("BruteForce");
        std::vector<cv::KeyPoint> imgKeypoints;

        //Creates the dictionary trainer and its vocabulary
        cv::TermCriteria tc(1000, 10, 0.1);
        int dictionarySize = 20;
        int retries = 1;
        int flags = cv::KMEANS_PP_CENTERS;
        cv::BOWKMeansTrainer bowTrain(dictionarySize, tc, retries, flags);
        cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor;
        bowExtractor = new cv::BOWImgDescriptorExtractor(descExtractor,
                descMatcher);
        Mat vocabulary;

        if(false)
        {
            //Start to handle images
            for (auto it = trainClassAndFiles.begin(); it != trainClassAndFiles.end(); ++it)
            {
                std::cout<<std::endl<<"Creating BOW for category: "<<it->first<<std::endl;
                std::cout<<"Processig images..."<<std::endl;
                std::string pathToCategory;
                pathToCategory.clear();
                pathToCategory.append("../data/train/");
                pathToCategory.append(it->first);
                for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    imgToAdd = imread(makePathToImage(pathToCategory, *it2));
                    imgPointer = &imgToAdd;
                    //Adds image descriptors to the trainer
                    cv::Mat imgDescriptors;
                    featureDetector->detect(*imgPointer, imgKeypoints); //Find keypoints
                    descExtractor->compute(*imgPointer, imgKeypoints, imgDescriptors); // Create Descriptors
                    bowTrain.add(imgDescriptors); //Add descriptors to trainer
                }
                std::cout<<std::endl;
            }
            //Set vocabulary with trainer
            vocabulary = bowTrain.cluster();
            parser.writeVocabulary(vocabulary,"../output/cvVocabulary.xml");
        }
        else
        {
            std::cout<<"Reading vocabulary from file"<<std::endl;
            vocabulary=parser.readVocabulary("../output/cvVocabulary.xml");
        }
        bowExtractor->setVocabulary(vocabulary);

        //  EXECUTING feature extraction
        Mat descriptors;
        Mat newDescriptors;
        vector<vector<int> > pointIdxsOfClusters;
        Mat trainingDescriptors(1500,dictionarySize,CV_32FC1);
        Mat trainingLabels = Mat::zeros(1500,1,CV_32FC1);

        for (int i = 0; i < 15; ++i)
        {
            for(int j=i*100; j<i*100+100; j++)
            {
                trainingLabels.at<float>(0,j)=i;
            }
        }

        cout<<"trainingLabels.size()"<<trainingLabels.size()<<endl;

        if(false)
        {
            int i = 0;
            for (auto it = trainClassAndFiles.begin(); it != trainClassAndFiles.end(); ++it)
            {
                std::cout<<std::endl<<"Adding category: "<<it->first<<" train matrix"<<std::endl;
                std::cout<<"Processig images..."<<std::endl;
                std::string pathToCategory;
                pathToCategory.clear();
                pathToCategory.append("../data/train/");
                pathToCategory.append(it->first);
                for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    imgToAdd = imread(makePathToImage(pathToCategory, *it2));
                    featureDetector->detect(imgToAdd,imgKeypoints);
                    bowExtractor->compute(imgToAdd, imgKeypoints, newDescriptors,
                            &pointIdxsOfClusters, &descriptors);

                    cout<<"-> "<<trainingLabels.at<float>(i,0)<<"\t";
                    for (int j = 0; j < trainingDescriptors.cols; ++j)
                    {
                        trainingDescriptors.at<float>(i,j)=newDescriptors.at<float>(j);
                        cout<<trainingDescriptors.at<float>(i,j)<<"\t";
                    }
                    cout<<endl;
                    ++i;
                }
            }

            if( remove( "../output/trainingSet.xml" ) != 0 )
            {
                cout<<"Error deleting file: ../output/trainingSet.xml"<<endl;
            }
            else
            {
                cout<<"../output/trainingSet.xml successfully deleted"<<endl;
            }
            parser.writeMat(trainingDescriptors,"../output/trainingSet.xml");
        }
        else
        {
            std::cout<<"Reading training descriptors from file"<<std::endl;
            trainingDescriptors = parser.readMat("../output/trainingSet.xml");
            cout << "Training Descriptors Size"
                    << " is: " << trainingDescriptors.size() << std::endl;
        }


        //  SVM Classifier
        //  Requiremets: 1D matrix describing the image (it may be its hitogram descriptor)
        //  - http://stackoverflow.com/questions/14694810/using-opencv-and-svm-with-images
        if(true)
        {
            //  1) Initializing the training matrix
            int num_files = 1500;  // TO_DO -> Has to check the folder again through a for loop
            int descLenght = dictionarySize;
            Mat trainingSVM_mat = trainingDescriptors;
            //  2) Store descriptors at trainingSVM_mat (1 new descriptor per row)
            //  and set the labelSVM_mat (class) - each row has the class for the descriptor
            //trainingSVM_mat.row(0) = newDescriptors;
            cout<<"trainingSVM_mat size is: "<<trainingSVM_mat.size()<<endl;

            //Mat labelsSVM_mat(num_files,1,CV_32FC1);
            Mat labelsSVM_mat = trainingLabels;

            //  3) Set SVM parameters (for kernel separation - instead of 2d separation line)
            CvSVMParams params = CvSVMParams();
            params.svm_type = CvSVM::C_SVC;
            params.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
            params.degree = 0; // for poly
            params.gamma = 3; // for poly/rbf/sigmoid
            params.coef0 = 0; // for poly/sigmoid

            params.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
            params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
            params.p = 0.0; // for CV_SVM_EPS_SVR

            params.class_weights = NULL; // for CV_SVM_C_SVC
            params.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
            params.term_crit.max_iter = 1000;
            params.term_crit.epsilon = 1e-6;

            //  4) Creating SVM and training data if true ; load data if false
            CvSVM svm;
            if(true)
            {
                svm.train(trainingSVM_mat, labelsSVM_mat, Mat(), Mat(), params);
                svm.save("../output/svm_filename"); // saving
            } else if(false)
            {
                svm.load("../output/svm_filename"); // loading
            }

            //  Testing File
            cout<<endl;
            int classe = 0;
            int predicted = 0;
            int total=0;
            int corrected =0;
            for (auto it = testClassAndFiles.begin(); it != testClassAndFiles.end(); ++it)
            {
                std::cout<<std::endl<<"Testing category: "<<it->first<<std::endl;
                std::string pathToCategory;
                pathToCategory.clear();
                pathToCategory.append("../data/test/");
                pathToCategory.append(it->first);
                int i=1;
                for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    imgToAdd = imread(makePathToImage(pathToCategory, *it2));
                    featureDetector->detect(imgToAdd,imgKeypoints);
                    bowExtractor->compute(imgToAdd, imgKeypoints, newDescriptors,
                            &pointIdxsOfClusters, &descriptors);
                    predicted = svm.predict(newDescriptors);

                    cout<<">> \t";
                    if(i%5)
                    {
                        cout<<predicted;
                        if (predicted == classe)
                        {
                            cout<<"*\t";
                            ++corrected;
                        }
                        else
                            cout<<"\t";
                    }
                    else
                    {
                        cout<<predicted;
                        if (predicted == classe)
                        {
                            cout<<"*\n";
                            ++corrected;
                        }
                        else
                            cout<<"\n";
                    }
                    ++total;
                    ++i;
                }
                cout<<"accuracy: "<<(float)corrected/total<<endl<<endl;
                ++classe;
            }
        }
        parser.writeMatToFile(newDescriptors,"../output/cvNewDescriptors.xml");
        parser.histogramPlot("../output/cvNewDescriptors.xml");


        delete imgPointer;
    }
	return 0;
}
