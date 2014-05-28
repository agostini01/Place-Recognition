/*
 * DataParser.cpp
 *
 *  Created on: 04/04/2014
 *      Author: agostini
 */

#include "DataParser.h"

namespace dataParser {

DataParser::DataParser() {
	// TODO Auto-generated constructor stub

}

DataParser::~DataParser() {
	// TODO Auto-generated destructor stub
}

void DataParser::writeMatToFile(cv::Mat &m, const char *filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cout << "File Not Opened" << std::endl;
        return;
    }
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            fout << "Cluster-" << j << " " << j << " " << m.at<float>(i, j)
                 << "\n";
        }
        fout << std::endl;
    }
    fout.close();
}

void DataParser::writeVocabulary(cv::Mat& vocabulary, const char* filename) {
    //Mat vocabulary = bowTrain.cluster();
    //vocabulary is parsed to this function
    cv::FileStorage vocabularyFileWrite;
    vocabularyFileWrite.open(filename, cv::FileStorage::WRITE);
    vocabularyFileWrite << "vocabulary" << vocabulary;
    vocabularyFileWrite.release();
}

void DataParser::writeMat(cv::Mat &matFile, const char *filename)
{
    cv::FileStorage matFileWrite;
    matFileWrite.open(filename, cv::FileStorage::WRITE);
    matFileWrite << "matFile" << matFile;
    matFileWrite.release();
}

cv::Mat DataParser::readVocabulary(const char* filename) {
	cv::Mat vocabulary;
	cv::FileStorage vocabularyFileRead;
	vocabularyFileRead.open(filename, cv::FileStorage::READ);
    vocabularyFileRead["vocabulary"] >> vocabulary;
	//bowExtractor.setVocabulary(vocabulary);
	//set must be done after reading;
    return vocabulary;
}

cv::Mat DataParser::readMat(const char *filename)
{
    cv::Mat matFile;
    cv::FileStorage matFileRead;
    matFileRead.open(filename, cv::FileStorage::READ);
    matFileRead["matFile"] >> matFile;
    return matFile;
}

void DataParser::histogramPlot(const char* filename) {
	Gnuplot gp("tee plot.gp | gnuplot -persist");
	gp << "reset\n";
	gp << "set xlabel \"Clusters\"\n";
	gp << "set ylabel \"Probability (%)\"\n";
	gp << "set yrange [0:1]\n";
	gp << "set grid\n";
	gp << "set boxwidth 0.95 relative\n";
	gp << "set xtics nomirror rotate by -45 font \".8\"\n";
	gp << "set style fill transparent solid 0.5 noborder\n";
	gp << "plot \""<<filename<<"\" u 2:3:xtic(1) w boxes lc rgb\"green\" notitle,"
			" \""<<filename<<"\" using 2:(($3+0.05)):3 with labels notitle \n";
}

}
/* namespace dataParser */
