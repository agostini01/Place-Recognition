/*
 * examples.h
 *
 *  Created on: 27/03/2014
 *      Author: agostini
 */

#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "tinydir.h"
#include <iostream>
#include <stdio.h>

//for gnuplot
#include <map>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h"



class examples {
public:
	examples();
	virtual ~examples();

	void gnuplotExample();
	void gnuplotHistExample();
	void historiogramCalculationExample(std::string path1);
	void kmeansExample();
	void bowExtractorExample(std::string path1,std::string path2);
	void displayPrecedure(cv::Mat image);
    void tinydirExample();
    void svmExample();
};

#endif /* EXAMPLES_H_ */
