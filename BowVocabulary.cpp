/*
 * BowVocabulary.cpp
 *
 *  Created on: 27/03/2014
 *      Author: agostini
 */

#include "BowVocabulary.h"


BowVocabulary::BowVocabulary():
m_vocabulary(){
}

BowVocabulary::~BowVocabulary() {

}

void BowVocabulary::addImgToBOW(cv::Mat& img) {
	cv::Ptr<cv::FeatureDetector> featureDetector =
			cv::FeatureDetector::create("SURF");
	cv::Ptr<cv::DescriptorExtractor> descExtractor =
			cv::DescriptorExtractor::create("SURF");
	cv::Ptr<cv::DescriptorMatcher> descMatcher =
			cv::DescriptorMatcher::create("BruteForce");
	std::vector<cv::KeyPoint> imgKeypoints;
	cv::Mat imgDescriptors;
	featureDetector->detect(img, imgKeypoints);
	descExtractor->compute(img, imgKeypoints, imgDescriptors);

	BowVocabulary::updateVocabulary(imgDescriptors,descExtractor,descMatcher);

}

void BowVocabulary::updateVocabulary(cv::Mat imgDescriptors,
		cv::Ptr<cv::DescriptorExtractor> descExtractor,
		cv::Ptr<cv::DescriptorMatcher> descMatcher) {

	cv::TermCriteria tc(1000, 10, 0.1);
	int dictionarySize = 5;
	int retries = 1;
	int flags = cv::KMEANS_PP_CENTERS;
	cv::BOWKMeansTrainer bowTrain(dictionarySize, tc, retries, flags);

	bowTrain.add(imgDescriptors);
	m_vocabulary = bowTrain.cluster();
	cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor;
	bowExtractor = new cv::BOWImgDescriptorExtractor(descExtractor,
			descMatcher);
	bowExtractor->setVocabulary(m_vocabulary);

}

