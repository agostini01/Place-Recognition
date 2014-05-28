/*
 * BowVocabulary.h
 *
 *  Created on: 27/03/2014
 *      Author: agostini
 */

#ifndef BOWVOCABULARY_H_
#define BOWVOCABULARY_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>

class BowVocabulary {
public:
	BowVocabulary();

	virtual ~BowVocabulary();

	void addImgToBOW(cv::Mat& img);
	void updateVocabulary(cv::Mat imgDescriptors,
			cv::Ptr<cv::DescriptorExtractor> descExtractor,
			cv::Ptr<cv::DescriptorMatcher> descMatcher);


private:
	cv::Mat m_vocabulary;

//	const cv::TermCriteria m_termCriteria;
//	cv::BOWKMeansTrainer m_bowTrain;
//	cv::Ptr<cv::FeatureDetector> m_featureDetector ;
//	const cv::Ptr<cv::DescriptorExtractor> m_descExtractor;
//	const cv::Ptr<cv::DescriptorMatcher> m_descMatcher;
//	cv::Ptr<cv::BOWImgDescriptorExtractor> m_bowExtractor;

//cv::vector<cv::vector<int> > m_pointIdxsOfClusters;

};

#endif /* BOWVOCABULARY_H_ */
