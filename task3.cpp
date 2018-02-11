// this program makes a text file and put every data sets in one folder with proper numbering

// fileoperation.cpp is comparative program

#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>

#include "nms.h"

using namespace cv;
using namespace cv::ml;
using namespace std;


unsigned char isFile =0x8;

int main()
  {

    std::vector<cv::Rect> srcRects0;
    std::vector<float> scores0;
    std::vector<cv::Rect> srcRects1;
    std::vector<float> scores1;
    std::vector<cv::Rect> srcRects2;
    std::vector<float> scores2;

    int testInd = 4;
    float minScoresSum = 0;

  vector <Rect> ground_truth;
/*   Rect fake_rect;

     for(int g=0;i<3;i++)
     ground_truth.push_back(fake_rect);
*/
  Mat src;
  ifstream fin;
  string dir, filepath, temp, temp1;
  int num;
  int counter=1;
  DIR *dp;
  struct dirent *dirp;
  struct stat filestat;
  const char* new_name;
  const char* new_name1;
  ostringstream convert;
  ostringstream convert1;
  //ofstream outputFile;
  //outputFile.open("/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/pos.txt");

//  cout << "dir to get files of: " << flush;
//  getline( cin, dir );  // gets everything the user ENTERs

//  dp = opendir( dir.c_str() );

  dir = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/gt";
  dp = opendir("/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/gt");

  if (dp == NULL)
    {
    cout << "Error opening " << dir << endl;
    //outputFile.close();
    return 0;
    }

  dirp = readdir( dp );
 
  while (dirp)
  {

    filepath = dir + "/" + dirp->d_name;

    if( dirp->d_type == isFile )
    {
	temp = dirp->d_name;
	temp1 = (temp.substr(0,4));
	cout << temp1 << endl;
	cout << temp << endl;

    string line;
    ifstream file;

    string image_path = "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/test/" + temp1 + ".jpg";
    //cout << image_path << endl;

    Mat src = imread( image_path.c_str() ); // to check whether file is openable

      if( !src.data )
      { printf(" No data! -- Exiting the program \n");
      continue; }


    file.open( filepath.c_str() );
    if( !file.is_open() )
    {
        cerr << "Unable to open the list of images from " << filepath << " filename." << endl;
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
	cout << line << endl;

    std::string str = line;
    std::istringstream buf(str);
    std::istream_iterator<std::string> beg(buf), end;

    std::vector<std::string> tokens(beg, end); // done!

    for(auto& s: tokens)
        std::cout << s << '\n';	

	Rect temp_gt;

	temp_gt.x = stoi(tokens[1]);
	temp_gt.y = stoi(tokens[2]);
	temp_gt.width = stoi(tokens[3]) - stoi(tokens[1]);
	temp_gt.height = stoi(tokens[4]) - stoi(tokens[2]);

	ground_truth.push_back(temp_gt);

        //rectangle( src, region_of_interest, Scalar( 0, 0, 255 ), 2 );

/////////////////////Implementation of Sliding window/////////////////////////////

	vector< Rect > locations;
        int WndWidth = 96;
        int WndHeight = 96;
	int StepSizeX = 20;
	int StepSizeY = 20;
        float SCALE_FACTOR = 5;

	Mat dst;
	Mat temp_img;
	for (float scale = 1; scale >= 1; scale = scale / SCALE_FACTOR)
	{
		resize(src, dst, Size(cvRound(src.cols*scale), cvRound(src.rows*scale)));
		//pyrDown(img, dst, Size(img.cols*scale, img.rows*scale));
		int ImRows = dst.rows;
		int ImCols = dst.cols;
		for (int i = 0; i < ImRows - WndHeight + 1; i = i + StepSizeY)
		{
			for (int j = 0; j < ImCols - WndWidth + 1; j = j + StepSizeX)
			{
				Rect region_of_interest = Rect(j, i, WndWidth, WndHeight);
				temp_img = dst(region_of_interest);
				//Apply Neural Network

				  if( !temp_img.data )
				  { printf(" No data! -- Exiting the program \n");
				  return -1; }

			    HOGDescriptor hog;
			    hog.winSize = Size( 96, 96 );
			    Mat gray;
			    vector< Point > location;
			    vector< float > descriptors;

				cvtColor( temp_img, gray, COLOR_BGR2GRAY );
		 
				hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ), location );

			Mat hog_descriptor =  Mat( descriptors ).clone();

			//trainData.push_back(hog_descriptor);

	
			      //--Convert data
			    const int rows = 1;
			    const int cols = (int)std::max( hog_descriptor.cols, hog_descriptor.rows );
			    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
			    Mat trainData = cv::Mat(rows, cols, CV_32FC1 );

				if( hog_descriptor.cols == 1 )
				{
				    transpose( hog_descriptor, tmp );
				    tmp.copyTo( trainData.row( 0 ) );
				}
				else if( hog_descriptor.rows == 1 )
				{
				    hog_descriptor.copyTo( trainData.row( 0 ) );
				}

			    cout << trainData.rows << " " << trainData.cols << endl;

				clog << "Start testing...";

				Ptr<RTrees> model = StatModel::load<RTrees>( "object_detector_RF_task3.yml" );
				if( model.empty() )
				cout << "Could not read the classifier " << endl;
				else
				cout << "The classifier is loaded.\n";

				float result = model->predict( trainData );
				cout << "result " << result << endl;

					imshow( "Thumbdetector", temp_img );
		  			//waitKey(100);


				if (result == (0 || 1 || 2))//If Neural Network Responds positive)
				{
//
//					imshow( "Thumbdetector", temp_img );
//		  			waitKey(100);
					region_of_interest.x *= scale;
					region_of_interest.y *= scale;
					region_of_interest.width*= scale;
					region_of_interest.height *= scale;
					locations.push_back(region_of_interest);

					if(result == 0){
					srcRects0.push_back(region_of_interest);
        				scores0.push_back(result);}

					if(result == 1){
					srcRects1.push_back(region_of_interest);
        				scores1.push_back(result);}

					if(result == 2){
					srcRects2.push_back(region_of_interest);
        				scores2.push_back(result);}
				}
			}
		}
	}

    if( !locations.empty() )
    {
        vector< Rect >::const_iterator loc = locations.begin();
        vector< Rect >::const_iterator end = locations.end();
        for( ; loc != end ; ++loc )
        {
            rectangle( src, *loc, Scalar( 0, 0, 255 ), 2 );
        }
    }

//rectangle( src, ground_truth, Scalar( 255, 0, 0 ), 2 );
    }


    std::vector<cv::Rect> resRects0;
    std::vector<cv::Rect> resRects1;
    std::vector<cv::Rect> resRects2;

    if (srcRects0.size() == scores0.size())
nms(srcRects0, resRects0, 0.3f, 1);
        //nms2(srcRects0, scores0, resRects0, 0.3f, 1, minScoresSum);

    for (auto r : resRects0)
        cv::rectangle(src, r, cv::Scalar(0, 255, 0), 2);

   if (srcRects1.size() == scores1.size())
nms(srcRects1, resRects1, 0.3f, 1);
       // nms2(srcRects1, scores1, resRects1, 0.3f, 1, minScoresSum);

    for (auto r : resRects1)
        cv::rectangle(src, r, cv::Scalar(0, 255, 0), 2);

   if (srcRects2.size() == scores2.size())
	nms(srcRects2, resRects2, 0.3f, 1);
//        nms2(srcRects2, scores2, resRects2, 0.3f, 1, minScoresSum);

    for (auto r : resRects2)
        cv::rectangle(src, r, cv::Scalar(0, 255, 0), 2);


for(int p=0;p<3;p++)
rectangle( src, ground_truth[p], Scalar( 255, 0, 0 ), 2 );


        imshow( "image", src );
        waitKey(0);

    }

    dirp = readdir( dp );
  
    if(!dirp)
    {

	   cout << "dir to get files of: " << flush;

	   getline( cin, dir );  // gets everything the user ENTERs

	   if(dir.c_str() == "quit")
	   break;

	   dp = opendir( dir.c_str() );
	   
	   if (dp == NULL)
	   {
	    cout << "Error opening " << dir << endl;
	    //outputFile.close();
	    return 0;
	   }

	   dirp = readdir( dp );
  
    }

 }

  closedir( dp );
  //outputFile.close();

  return 0;

}
