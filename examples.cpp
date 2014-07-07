/*
 * examples.cpp
 *
 *  Created on: 27/03/2014
 *      Author: agostini
 */

#include "examples.h"

using namespace std;
using namespace cv;

examples::examples() {
	// TODO Auto-generated constructor stub

}

examples::~examples() {
	// TODO Auto-generated destructor stub
}

void examples::historiogramCalculationExample(string path1) {
	cout
			<< "Executing void examples::historiogramCalculationExample(string path1)"
			<< endl;
	Mat src = imread(path1);

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange,
			uniform, accumulate);
	cout << "HERE!" << endl;
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange,
			uniform, accumulate);
	cout << "HERE!" << endl;
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange,
			uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double) hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++) {
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	cout << "historiogram size" << histImage.size() << endl;
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	waitKey(0);

}

void examples::kmeansExample() {

	const int MAX_CLUSTERS = 5;
	Scalar colorTab[] = { Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 100,
			100), Scalar(255, 0, 255), Scalar(0, 255, 255) };

	Mat img(500, 500, CV_8UC3);
	RNG rng(12345);

	for (;;) {
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		int i, sampleCount = rng.uniform(1, 1001);
		Mat points(sampleCount, 2, CV_32F), labels;

		clusterCount = MIN(clusterCount, sampleCount);
		Mat centers;

		/* generate random sample from multigaussian distribution */
		for (k = 0; k < clusterCount; k++) {
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k * sampleCount / clusterCount,
					k == clusterCount - 1 ?
							sampleCount : (k + 1) * sampleCount / clusterCount);
			rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y),
					Scalar(img.cols * 0.05, img.rows * 0.05));
		}

		randShuffle(points, 1, &rng);

		kmeans(points, clusterCount, labels,
				TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
				3, KMEANS_PP_CENTERS, centers);

		img = Scalar::all(0);

		for (i = 0; i < sampleCount; i++) {
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx]);
		}

		imshow("clusters", img);

		char key = (char) waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}

void examples::gnuplotExample() {

	//Gnuplot gp;
	// Create a script which can be manually fed into gnuplot later:
	//Gnuplot gp(">script.gp");
	// Create script and also feed to gnuplot:
	Gnuplot gp("tee plot.gp | gnuplot -persist");
	// Or choose any of those options at runtime by setting the GNUPLOT_IOSTREAM_CMD
	// environment variable.
	// Gnuplot vectors (i.e. arrows) require four columns: (x,y,dx,dy)
	std::vector<boost::tuple<double, double, double, double> > pts_A;
	// You can also use a separate container for each column, like so:
	std::vector<double> pts_B_x;
	std::vector<double> pts_B_y;
	std::vector<double> pts_B_dx;
	std::vector<double> pts_B_dy;
	// You could also use:
	//   std::vector<std::vector<double> >
	//   boost::tuple of four std::vector's
	//   std::vector of std::tuple (if you have C++11)
	//   arma::mat (with the Armadillo library)
	//   blitz::Array<blitz::TinyVector<double, 4>, 1> (with the Blitz++ library)
	// ... or anything of that sort
	for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0) {
		double theta = alpha * 2.0 * 3.14159;
		pts_A.push_back(
				boost::make_tuple(cos(theta), sin(theta), -cos(theta) * 0.1,
						-sin(theta) * 0.1));
		pts_B_x.push_back(cos(theta) * 0.8);
		pts_B_y.push_back(sin(theta) * 0.8);
		pts_B_dx.push_back(sin(theta) * 0.1);
		pts_B_dy.push_back(-cos(theta) * 0.1);
	}
	// Don't forget to put "\n" at the end of each line!
	gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
	// '-' means read from stdin.  The send1d() function sends data to gnuplot's stdin.
	gp
			<< "plot '-' with vectors title 'pts_A', '-' with vectors title 'pts_B'\n";
	gp.send1d(pts_A);
	gp.send1d(boost::make_tuple(pts_B_x, pts_B_y, pts_B_dx, pts_B_dy));

}

void examples::gnuplotHistExample() {
	Gnuplot gp("tee plot.gp | gnuplot -persist");

	cv::Mat a = cv::Mat::ones(10, 2, CV_32FC3);
	gp << "set xrange [-10:10]\nset yrange [-10:10]\n";
	gp<< "plot '-' with vectors title 'a'\n";
	//gp.send1d(a);
}

void examples::bowExtractorExample(string path1, string path2) {

	Mat img_1 = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);

	//To detect keypoints
	int minThreshold = 40;
	SurfFeatureDetector detector(minThreshold);
	std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_total;
	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);

	//To extract descriptors
	SurfDescriptorExtractor extractor;
	FlannBasedMatcher matcher;
	Mat descriptors_1, descriptors_2;
	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	//To Create a bag of words
	TermCriteria tc(1000, 10, 0.1);
	int dictionarySize = 5;
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	unsigned int i = 0;
	BOWKMeansTrainer bowTrain(dictionarySize, tc, retries, flags);
	bowTrain.add(descriptors_1);
	i++;
	cout << "Number of descriptors inside BOW after adding descriptors " << i
			<< " is: " << bowTrain.descripotorsCount() << std::endl;
	bowTrain.add(descriptors_2);
	i++;
	cout << "Number of descriptors inside BOW after adding descriptors " << i
			<< " is: " << bowTrain.descripotorsCount() << std::endl;

	Mat vocabulary = bowTrain.cluster(); // BOW core presented

	// Create detector, descriptor, matcher.
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(
			"SURF");
	Ptr<BOWImgDescriptorExtractor> bowExtractor;
	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create(
			"BruteForce");
	bowExtractor = new BOWImgDescriptorExtractor(descExtractor, descMatcher);
	vector<vector<int> > pointIdxsOfClusters;
	Mat descriptors;
	Mat newDescriptors;

	// 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
	bowExtractor->setVocabulary(vocabulary);

	// 2. Compute new descriptor based on vocabulary similarity
	bowExtractor->compute(img_1, keypoints_1, newDescriptors,
			&pointIdxsOfClusters, &descriptors);
	cout << "newDescriptors.size()" << newDescriptors.size() << std::endl;
	cout << "newDescriptors.size()" << pointIdxsOfClusters.size() << std::endl;
	for (i = 0; i < pointIdxsOfClusters.size(); ++i) {
		cout << "pointIdxsOfClusters[" << i << "].size()"
				<< pointIdxsOfClusters[i].size() << " x "
				<< newDescriptors.at<float>(i) << std::endl;
	}

	cout << "descriptors.size()" << descriptors.size() << std::endl
			<< std::endl;

	// 2. Compute new descriptor based on vocabulary similarity
	bowExtractor->compute(img_2, keypoints_2, newDescriptors,
			&pointIdxsOfClusters, &descriptors);

	//Displaying
	cout << "newDescriptors.size()" << newDescriptors.size() << std::endl;
	cout << "newDescriptors.size()" << pointIdxsOfClusters.size() << std::endl;
	for (i = 0; i < pointIdxsOfClusters.size(); ++i) {
		cout << "pointIdxsOfClusters[" << i << "].size()"
				<< pointIdxsOfClusters[i].size() << " x "
				<< newDescriptors.at<float>(i) << std::endl;
	}

	cout << "descriptors.size()" << descriptors.size() << std::endl
			<< std::endl;

	//To display image
	drawKeypoints(img_1, keypoints_1, img_1);
	drawKeypoints(img_2, keypoints_2, img_2);
	//displayPrecedure(img_1);
	//displayPrecedure(img_2);

}

void examples::tinydirExample()
{
    tinydir_dir dir;
    int i;
    tinydir_open_sorted(&dir, "../data/test/bedroom");

    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);

        printf("%s", file.name);
        if (file.is_dir)
        {
            printf("/");
        }
        printf("\n");
    }

    tinydir_close(&dir);
}

void examples::svmExample()
{
//    if(false)
//    {
//        std::vector<int> indices;
//        for(int i=0; i<trainingLabels.rows; ++i)
//        {
//            indices.push_back(i);
//        }
//        std::random_device device;
//        std::mt19937 generator(device());
//        std::shuffle(indices.begin(), indices.end(), generator);

//        cv::Mat trainingDescriptorsShuffled(
//            trainingDescriptors.rows,
//            trainingDescriptors.cols,
//            CV_32FC1
//        );
//        cv::Mat trainingLabelsShuffled(
//            trainingLabels.rows,
//            trainingLabels.cols,
//            CV_32FC1
//        );

////            for(int i=0; i<static_cast<int>(indices.size()); ++i)
////            {
////                trainingDescriptorsShuffled.row(i) = trainingDescriptors.row(indices[i]);
////                trainingLabelsShuffled.at<float>(i) = trainingLabels.at<float>(indices[i]);
////            }

//        for(int i=0; i<trainingDescriptors.rows; ++i)
//        {
//            for(int j=0; j<trainingDescriptors.cols; ++j)
//            {
//                trainingDescriptorsShuffled.at<float>(i, j) = trainingDescriptors.at<float>(indices[i], j);
//            }
//            trainingLabelsShuffled.at<float>(i) = trainingLabels.at<float>(indices[i]);
//        }

//        //  SVM Classifier
//        //  Requiremets: 1D matrix describing the image (it may be its hitogram descriptor)
//        //  - http://stackoverflow.com/questions/14694810/using-opencv-and-svm-with-images
////            SvmClassifier svm;  // to load the previus trained classfier
//        //SvmClassifier svm(trainingDescriptors,trainingLabels);
//        SvmClassifier svm(trainingDescriptorsShuffled, trainingLabelsShuffled);
//        std::cout << trainingDescriptorsShuffled.rows << " " << trainingDescriptorsShuffled.cols << " | "
//                  << trainingLabelsShuffled.rows << " " << trainingLabelsShuffled.cols << std::endl;
//            //  Testing File
//            std::cout<<std::endl;
//            int classe = 0;
//            int predicted = 0;
//            int total=0;
//            int corrected =0;

//            if(true)
//            {
//                for (int i = 0; i < trainingDescriptors.rows; ++i)
//                {
//                    newDescriptors = trainingDescriptors.row(i);
//                    predicted = svm.predict(newDescriptors);

//                    if(!(i%100))
//                    {
//                        if(i!=0)
//                            std::cout<<"\nAccuracy: "<<(float)corrected/total<<std::endl;
//                        std::cout<<"\n\nClass "<<trainingLabels.at<float>(i)<<"\n";
//                    }
//                    if(!(i%5)&& i!=0)
//                    {
//                        std::cout<<"\n";
//                    }
//                    std::cout<<predicted;
//                    if (predicted == trainingLabels.at<float>(i))
//                    {
//                        std::cout<<"*\t";
//                        ++corrected;
//                    }
//                    else
//                        std::cout<<"\t";
//                    ++total;
//                }
//                std::cout<<std::endl<<"Total accuracy: "<<(float)corrected/total<<std::endl;
//                ++classe;
//            }else
//            {
//                for (auto it = testClassAndFiles.begin(); it != testClassAndFiles.end(); ++it)
//                {
//                    std::cout<<std::endl<<"Testing category: "<<it->first<<std::endl;
//                    std::string pathToCategory;
//                    pathToCategory.clear();
//                    pathToCategory.append("../data/test/");
//                    pathToCategory.append(it->first);
//                    int i=1;
//                    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
//                    {
//                        imgToAdd = imread(makePathToImage(pathToCategory, *it2));
//                        featureDetector->detect(imgToAdd,imgKeypoints);
//                        bowExtractor->compute(imgToAdd, imgKeypoints, newDescriptors,
//                                &pointIdxsOfClusters, &descriptors);
//                        predicted = svm.predict(newDescriptors);

//                        std::cout<<">> \t";
//                        if(i%5)
//                        {
//                            std::cout<<predicted;
//                            if (predicted == classe)
//                            {
//                                std::cout<<"*\t";
//                                ++corrected;
//                            }
//                            else
//                                std::cout<<"\t";
//                        }
//                        else
//                        {
//                            std::cout<<predicted;
//                            if (predicted == classe)
//                            {
//                                std::cout<<"*\n";
//                                ++corrected;
//                            }
//                            else
//                                std::cout<<"\n";
//                        }
//                        ++total;
//                        ++i;
//                    }
//                    std::cout<<"accuracy: "<<(float)corrected/total<<std::endl<<std::endl;
//                    ++classe;
//                }

//        }
//    }
}

/**
 * @function dysplayPrecedure
 */
void displayPrecedure(cv::Mat image) {
	namedWindow("Janela 01", WINDOW_AUTOSIZE);
	imshow("Janela 01", image);
	waitKey(0);
	destroyAllWindows();
}
