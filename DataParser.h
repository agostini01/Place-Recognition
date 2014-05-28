/*
 * DataParser.h
 *
 *  Created on: 04/04/2014
 *      Author: agostini
 */

#ifndef DATAPARSER_H_
#define DATAPARSER_H_

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "gnuplot-iostream.h"
#include "tinydir.h"            //Read Files and Folders

namespace dataParser {

class DataParser {
public:
	DataParser();
	virtual ~DataParser();

    void writeMatToFile(cv::Mat& m, const char* filename);
	void writeVocabulary(cv::Mat& vocabulary, const char* filename);
    void writeMat(cv::Mat& matFile, const char* filename);
	void histogramPlot(const char* filename);
	cv::Mat readVocabulary(const char* filename);
    cv::Mat readMat(const char* filename);
	//cv::Mat getImagesPath(const char* imagesPath);

};

} /* namespace dataParser */
#endif /* DATAPARSER_H_ */
