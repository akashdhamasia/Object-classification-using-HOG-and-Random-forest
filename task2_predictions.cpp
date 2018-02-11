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
void test_RF( const vector< Mat > & gradient_lst, const vector< int > & labels, vector< vector< int > >& confusion_mat );
void test_DT( const vector< Mat > & gradient_lst, const vector< int > & labels );

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

void test_RF( const vector< Mat > & gradient_lst, const vector< int > & labels, vector< vector< int > >& confusion_mat )
{
	Mat test_data;
	float counter_acc = 0;
	convert_to_ml( gradient_lst, test_data );

	clog << "Start training...";

	Ptr<RTrees> model = StatModel::load<RTrees>( "object_detector_RF.yml" );
	if( model.empty() )
	cout << "Could not read the classifier " << endl;
	else
	cout << "The classifier is loaded.\n";

	for(int i = 0; i < test_data.rows; i++ )
	{

	 float result = model->predict( test_data.row(i));
	 //cout << "prediction " << result << " actual_label "<<labels[i] << endl;
	 int actual = labels[i];
	 int predicted = result;
	 confusion_mat[actual][predicted]++;

	if(actual = result)counter_acc++;

	}

	cout << "Accuracy = " << counter_acc/test_data.rows << endl;

	clog << "...[done]" << endl;
}

void test_DT( const vector< Mat > & gradient_lst, const vector< int > & labels, vector< vector< int > >& confusion_mat )
{
	Mat test_data;
	float counter_acc = 0;
	convert_to_ml( gradient_lst, test_data );

	clog << "Start training...";

	Ptr<DTrees> model = StatModel::load<DTrees>( "object_detector_DT.yml" );
	if( model.empty() )
	cout << "Could not read the classifier " << endl;
	else
	cout << "The classifier is loaded.\n";

	for(int i = 0; i < test_data.rows; i++ )
	{

	 float result = model->predict( test_data.row(i));
	 //cout << "prediction " << result << " actual_label "<<labels[i] << endl;
	 int actual = labels[i];
	 int predicted = result;
	 confusion_mat[actual][predicted]++;

	if(actual == result)counter_acc++;

	}

	cout << "Accuracy = " << counter_acc/test_data.rows << endl;

	clog << "...[done]" << endl;
}

int main( int argc, char** argv )
{
    vector <int> init_conf_mat;
    vector< vector< int > > confusion_mat;
    int k=0;

	for(int i=0;i<6;i++)
	{
		 init_conf_mat.push_back(k);
		
	}

	for(int i=0;i<6;i++)
	{
		 confusion_mat.push_back(init_conf_mat);
		
	}

    vector< Mat > pos_lst0;
    vector< Mat > neg_lst0;
    
    vector< Mat > pos_lst1;
    vector< Mat > neg_lst1;

    vector< Mat > pos_lst2;
    vector< Mat > neg_lst2;

    vector< Mat > pos_lst3;
    vector< Mat > neg_lst3;

    vector< Mat > pos_lst4;
    vector< Mat > neg_lst4;

    vector< Mat > pos_lst5;
    vector< Mat > neg_lst5;
    
    vector< Mat > gradient_lst;
    vector< int > labels;

    string pos_dir0 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task2/test/0/";
    string pos_dir1 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task2/test/1/";
    string pos_dir2 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task2/test/2/";
    string pos_dir3 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task2/test/3/";
    string pos_dir4 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task2/test/4/";
    string pos_dir5 = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task2/test/5/";

    string pos = "pos.txt";

    load_images( pos_dir0, pos, pos_lst0 );
    labels.assign( pos_lst0.size(), 0 );

    load_images( pos_dir1, pos, pos_lst1 );
    labels.insert( labels.end(), pos_lst1.size(), 1 );

    load_images( pos_dir2, pos, pos_lst2 );
    labels.insert( labels.end(), pos_lst2.size(), 2 );

    load_images( pos_dir3, pos, pos_lst3 );
    labels.insert( labels.end(), pos_lst3.size(), 3 );

    load_images( pos_dir4, pos, pos_lst4 );
    labels.insert( labels.end(), pos_lst4.size(), 4 );

    load_images( pos_dir5, pos, pos_lst5 );
    labels.insert( labels.end(), pos_lst5.size(), 5 );

    compute_hog( pos_lst0, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst1, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst2, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst3, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst4, gradient_lst, Size( 96, 96 ) );
    compute_hog( pos_lst5, gradient_lst, Size( 96, 96 ) );

    test_RF( gradient_lst, labels, confusion_mat );

    for(int l=0; l<confusion_mat.size(); l++){

	for(int h=0; h<init_conf_mat.size(); h++)
	    std::cout << confusion_mat[l][h] << " ";

	cout << endl;
    }
    //test_it( Size( 96, 160 ) ); // change with your parameters

    return 0;
}
