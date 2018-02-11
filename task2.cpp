#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData );
void load_images( const string & prefix, const string & filename, vector< Mat > & img_lst );
void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size );
void train_RF( const vector< Mat > & gradient_lst, const vector< int > & labels );
void train_DT( const vector< Mat > & gradient_lst, const vector< int > & labels );
void train_svm( const vector< Mat > & gradient_lst, const vector< int > & labels );
inline TermCriteria TC(int iters, double eps);
//void train_own_RF( const vector< Mat > & gradient_lst, const vector< int > & labels, int no_of_trees, int no_of_samples )


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1 );
    vector< Mat >::const_iterator itr = train_samples.begin();
    vector< Mat >::const_iterator end = train_samples.end();
    for( int i = 0 ; itr != end ; ++itr, ++i )
    {
        CV_Assert( itr->cols == 1 ||
            itr->rows == 1 );
        if( itr->cols == 1 )
        {
            transpose( *(itr), tmp );
            tmp.copyTo( trainData.row( i ) );
        }
        else if( itr->rows == 1 )
        {
            itr->copyTo( trainData.row( i ) );
        }
    }
}

void load_images( const string & prefix, const string & filename, vector< Mat > & img_lst )
{
    string line;
    ifstream file;

    file.open( (prefix+filename).c_str() );
    if( !file.is_open() )
    {
        cerr << "Unable to open the list of images from " << filename << " filename." << endl;
        exit( -1 );
    }

    bool end_of_parsing = false;
    while( !end_of_parsing )
    {
        getline( file, line );
        if( line.empty() ) // no more file to read
        {
            end_of_parsing = true;
            break;
        }
        Mat img = imread( (prefix+line).c_str() ); // load the image
        if( img.empty() ) // invalid image, just skip it.
            continue;
#ifdef _DEBUG
        imshow( "image", img );
        waitKey( 10 );
#endif
        img_lst.push_back( img.clone() );
    }
}

void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size )
{
    HOGDescriptor hog;
    hog.winSize = size;
    Mat gray;
    vector< Point > location;
    vector< float > descriptors;

    vector< Mat >::const_iterator img = img_lst.begin();
    vector< Mat >::const_iterator end = img_lst.end();
    for( ; img != end ; ++img )
    {
        cvtColor( *img, gray, COLOR_BGR2GRAY );
        hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ), location );
        gradient_lst.push_back( Mat( descriptors ).clone() );
#ifdef _DEBUG
        imshow( "gradient", get_hogdescriptor_visu( img->clone(), descriptors, size ) );
        waitKey( 10 );
#endif
    }
}

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

void train_RF( const vector< Mat > & gradient_lst, const vector< int > & labels )
{

	Mat train_data;
	convert_to_ml( gradient_lst, train_data );

	clog << "Start training...";

	static Ptr<TrainData> training_data = TrainData::create(train_data, ROW_SAMPLE, Mat(labels));

	Ptr<RTrees> rtrees = RTrees::create();
	rtrees->setMaxDepth(10);
	rtrees->setMinSampleCount(10);
	rtrees->setRegressionAccuracy(0);
	rtrees->setUseSurrogates(false);
	rtrees->setMaxCategories(15);
	rtrees->setPriors(Mat());
	rtrees->setCalculateVarImportance(true);
	rtrees->setActiveVarCount(4);
	rtrees->setTermCriteria(TC(100,0.01f));
	rtrees->train(training_data);
	rtrees->save("object_detector_RF_task3_aug1.yml");

	clog << "...[done]" << endl;

	/*    Ptr<RTrees> rtrees = RTrees::create();
	rtrees->setMaxDepth(4);
	rtrees->setMinSampleCount(2);
	rtrees->setRegressionAccuracy(0.f);
	rtrees->setUseSurrogates(false);
	rtrees->setMaxCategories(16);
	rtrees->setPriors(Mat());
	rtrees->setCalculateVarImportance(false);
	rtrees->setActiveVarCount(1);
	rtrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5, 0));
	rtrees->train(prepare_train_data());
	*/

}

void train_DT( const vector< Mat > & gradient_lst, const vector< int > & labels )
{

	Mat train_data;
	convert_to_ml( gradient_lst, train_data );

	clog << "Start training...";

	static Ptr<TrainData> training_data = TrainData::create(train_data, ROW_SAMPLE, Mat(labels));

	Ptr<DTrees> dtree = DTrees::create();
	dtree->setMaxDepth(10);
	dtree->setMinSampleCount(2);
	dtree->setRegressionAccuracy(0);
	dtree->setUseSurrogates(false);
	dtree->setMaxCategories(16);
	dtree->setCVFolds(0);
	dtree->setUse1SERule(false);
	dtree->setTruncatePrunedTree(false);
	dtree->setPriors(Mat());
	dtree->train(training_data);
	dtree->save("object_detector_DT_task3_aug1.yml");

	clog << "...[done]" << endl;


}
/*
void train_own_RF( const vector< Mat > & gradient_lst, const vector< int > & labels, int no_of_trees, int no_of_samples )
{

	Mat train_data;
	convert_to_ml( gradient_lst, train_data );

	clog << "Start training...";

for(int i=0; i<no_of_trees; i++)
{
}

	static Ptr<TrainData> training_data = TrainData::create(train_data, ROW_SAMPLE, Mat(labels));

	vector <Ptr<DTrees> > trees;

	int counter_tree = 1;
	string name_tree = "object_detector_DT_task3_aug" + counter_tree + ".yml";

	

for(int i=0; i<no_of_trees; i++){

	string name_tree = "object_detector_DT_task3_aug" + counter_tree + ".yml";
	counter_tree++;

	Ptr<DTrees> dtree = DTrees::create();
	dtree->setMaxDepth(10);
	dtree->setMinSampleCount(2);
	dtree->setRegressionAccuracy(0);
	dtree->setUseSurrogates(false);
	dtree->setMaxCategories(16);
	dtree->setCVFolds(0);
	dtree->setUse1SERule(false);
	dtree->setTruncatePrunedTree(false);
	dtree->setPriors(Mat());
	dtree->train(training_data);
	
	dtree->save(name_tree.c_str());

}

	clog << "...[done]" << endl;


}
*/

void train_svm( const vector< Mat > & gradient_lst, const vector< int > & labels )
{

    Mat train_data;
    convert_to_ml( gradient_lst, train_data );

    clog << "Start training...";
    Ptr<SVM> svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-3 ));
    svm->setGamma(0);
    svm->setKernel(SVM::LINEAR);
    svm->setNu(0.5);
    svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
    svm->setC(0.01); // From paper, soft classifier
    svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train(train_data, ROW_SAMPLE, Mat(labels));
    clog << "...[done]" << endl;

    svm->save("object_detector_svm.yml");
}

int main( int argc, char** argv )
{
    vector< Mat > pos_lst0;
    vector< Mat > neg_lst0;
    
    vector< Mat > pos_lst1;
    vector< Mat > neg_lst1;

    vector< Mat > pos_lst2;
    vector< Mat > neg_lst2;

    vector< Mat > pos_lst3;
    vector< Mat > neg_lst3;

    //vector< Mat > pos_lst4;
    //vector< Mat > neg_lst4;

    //vector< Mat > pos_lst5;
    //vector< Mat > neg_lst5;
    
    vector< Mat > gradient_lst;
    vector< int > labels;

    string pos_dir0 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/a/";
    string pos_dir1 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/b/";
    string pos_dir2 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/";
    string pos_dir3 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/3/";
    //string pos_dir4 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/4/";
    //string pos_dir5 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/5/";

    string pos = "pos.txt";

    load_images( pos_dir0, pos, pos_lst0 );
    labels.assign( pos_lst0.size(), 0 );

    load_images( pos_dir1, pos, pos_lst1 );
    labels.insert( labels.end(), pos_lst1.size(), 1 );

    load_images( pos_dir2, pos, pos_lst2 );
    labels.insert( labels.end(), pos_lst2.size(), 2 );

    load_images( pos_dir3, pos, pos_lst3 );
    labels.insert( labels.end(), pos_lst3.size(), 3 );

    //load_images( pos_dir4, pos, pos_lst4 );
    //labels.insert( labels.end(), pos_lst4.size(), 4 );

    //load_images( pos_dir5, pos, pos_lst5 );
    //labels.insert( labels.end(), pos_lst5.size(), 5 );

    compute_hog( pos_lst0, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst1, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst2, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst3, gradient_lst, Size( 96, 96 ) );
    //compute_hog( pos_lst4, gradient_lst, Size( 96, 96 ) );
    //compute_hog( pos_lst5, gradient_lst, Size( 96, 96 ) );

    train_RF( gradient_lst, labels );

    //test_it( Size( 96, 160 ) ); // change with your parameters

    return 0;
}
