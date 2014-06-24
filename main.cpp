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
#include "foldsextraction.h"
#include "crossvalidation.h"
#include "svmclassifier.h"
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

    std::cout<<"Starting Application"<<std::endl;
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
        cv::Mat img1 = imread(argv[1]);
        cv::Mat img2 = imread(argv[2]);
        cv::Mat imgToAdd;
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
        cv::Mat vocabulary;

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
        cv::Mat descriptors;
        cv::Mat newDescriptors;
        vector<vector<int> > pointIdxsOfClusters;
        cv::Mat trainingDescriptors(1500,dictionarySize,CV_32FC1);
        cv::Mat trainingLabels = cv::Mat::zeros(1500,1,CV_32FC1);

        for (int i = 0; i < 15; ++i)
        {
            for(int j=i*100; j<i*100+100; j++)
            {
                trainingLabels.at<float>(0,j)=i;
            }
        }

        std::cout<<"trainingLabels.size()"<<trainingLabels.size()<<std::endl;

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

                    std::cout<<"-> "<<trainingLabels.at<float>(i,0)<<"\t";
                    for (int j = 0; j < trainingDescriptors.cols; ++j)
                    {
                        trainingDescriptors.at<float>(i,j)=newDescriptors.at<float>(j);
                        std::cout<<trainingDescriptors.at<float>(i,j)<<"\t";
                    }
                    std::cout<<std::endl;
                    ++i;
                }
            }

            if( remove( "../output/trainingSet.xml" ) != 0 )
            {
                std::cout<<"Error deleting file: ../output/trainingSet.xml"<<std::endl;
            }
            else
            {
                std::cout<<"../output/trainingSet.xml successfully deleted"<<std::endl;
            }
            parser.writeMat(trainingDescriptors,"../output/trainingSet.xml");
        }
        else
        {
            std::cout<<"Reading training descriptors from file"<<std::endl;
            trainingDescriptors = parser.readMat("../output/trainingSet.xml");
            std::cout << "Training Descriptors Size"
                    << " is: " << trainingDescriptors.size() << std::endl;
        }

        // Cross Validation Rotine
        unsigned int numberOfClasses = 15;
        FoldsExtraction my10Folds = FoldsExtraction(trainingDescriptors, trainingLabels, numberOfClasses);


        // Cross Validation SVM
        CrossValidation myCrossValidation(my10Folds);
        std::cout<<"Startin SVM crossvalidaion"<<std::endl;
        myCrossValidation.crossValidadeSVM();
        myCrossValidation.displayResults();



        if(false)
        {
            std::vector<int> indices;
            for(int i=0; i<trainingLabels.rows; ++i)
            {
                indices.push_back(i);
            }
            std::random_device device;
            std::mt19937 generator(device());
            std::shuffle(indices.begin(), indices.end(), generator);

            cv::Mat trainingDescriptorsShuffled(
                trainingDescriptors.rows,
                trainingDescriptors.cols,
                CV_32FC1
            );
            cv::Mat trainingLabelsShuffled(
                trainingLabels.rows,
                trainingLabels.cols,
                CV_32FC1
            );

//            for(int i=0; i<static_cast<int>(indices.size()); ++i)
//            {
//                trainingDescriptorsShuffled.row(i) = trainingDescriptors.row(indices[i]);
//                trainingLabelsShuffled.at<float>(i) = trainingLabels.at<float>(indices[i]);
//            }

            for(int i=0; i<trainingDescriptors.rows; ++i)
            {
                for(int j=0; j<trainingDescriptors.cols; ++j)
                {
                    trainingDescriptorsShuffled.at<float>(i, j) = trainingDescriptors.at<float>(indices[i], j);
                }
                trainingLabelsShuffled.at<float>(i) = trainingLabels.at<float>(indices[i]);
            }

            //  SVM Classifier
            //  Requiremets: 1D matrix describing the image (it may be its hitogram descriptor)
            //  - http://stackoverflow.com/questions/14694810/using-opencv-and-svm-with-images
//            SvmClassifier svm;  // to load the previus trained classfier
            //SvmClassifier svm(trainingDescriptors,trainingLabels);
            SvmClassifier svm(trainingDescriptorsShuffled, trainingLabelsShuffled);
            std::cout << trainingDescriptorsShuffled.rows << " " << trainingDescriptorsShuffled.cols << " | "
                      << trainingLabelsShuffled.rows << " " << trainingLabelsShuffled.cols << std::endl;
                //  Testing File
                std::cout<<std::endl;
                int classe = 0;
                int predicted = 0;
                int total=0;
                int corrected =0;

                if(true)
                {
                    for (int i = 0; i < trainingDescriptors.rows; ++i)
                    {
                        newDescriptors = trainingDescriptors.row(i);
                        predicted = svm.predict(newDescriptors);

                        if(!(i%100))
                        {
                            if(i!=0)
                                std::cout<<"\nAccuracy: "<<(float)corrected/total<<std::endl;
                            std::cout<<"\n\nClass "<<trainingLabels.at<float>(i)<<"\n";
                        }
                        if(!(i%5)&& i!=0)
                        {
                            std::cout<<"\n";
                        }
                        std::cout<<predicted;
                        if (predicted == trainingLabels.at<float>(i))
                        {
                            std::cout<<"*\t";
                            ++corrected;
                        }
                        else
                            std::cout<<"\t";
                        ++total;
                    }
                    std::cout<<std::endl<<"Total accuracy: "<<(float)corrected/total<<std::endl;
                    ++classe;
                }else
                {
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

                            std::cout<<">> \t";
                            if(i%5)
                            {
                                std::cout<<predicted;
                                if (predicted == classe)
                                {
                                    std::cout<<"*\t";
                                    ++corrected;
                                }
                                else
                                    std::cout<<"\t";
                            }
                            else
                            {
                                std::cout<<predicted;
                                if (predicted == classe)
                                {
                                    std::cout<<"*\n";
                                    ++corrected;
                                }
                                else
                                    std::cout<<"\n";
                            }
                            ++total;
                            ++i;
                        }
                        std::cout<<"accuracy: "<<(float)corrected/total<<std::endl<<std::endl;
                        ++classe;
                    }

                    parser.writeMatToFile(newDescriptors,"../output/cvNewDescriptors.xml");
                    parser.histogramPlot("../output/cvNewDescriptors.xml");
                    delete imgPointer;
            }
        }
    }
    return 0;
}


