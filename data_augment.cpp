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

using namespace cv;
using namespace std;

unsigned char isFile =0x8;

int main()
  {

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
  ofstream outputFile;
  outputFile.open("/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/pos.txt");

  cout << "dir to get files of: " << flush;
  getline( cin, dir );  // gets everything the user ENTERs

  dp = opendir( dir.c_str() );

  if (dp == NULL)
    {
    cout << "Error opening " << dir << endl;
    outputFile.close();
    return 0;
    }

  dirp = readdir( dp );
 
  while (dirp)
  {

    filepath = dir + "/" + dirp->d_name;

    if( dirp->d_type == isFile )
    {

	   	      temp = dirp->d_name;

		      temp1 = (temp.substr(temp.find_last_of(".") + 1));

	    //if(temp1 != "txt"){

	    //  if(temp1 == "png"){
	      
	      src = imread( filepath.c_str() ); // to check whether file is openable

	      if( !src.data )
	      { printf(" No data! -- Exiting the program \n");
		dirp = readdir( dp );
	      continue; }

              resize( src, src, Size(96, 96) );


/*	//////////for storing the negative patches////////////////

for(int c=0; c<10; c++){			resize( tmp, dst, Size( tmp.cols*1.2, tmp.rows*1.2 ) );

	Rect box;
	box.width = 64;
	box.height = 128;

	const int size_x = box.width;
	const int size_y = box.height;

	box.x = rand() % (src.cols - size_x);
	box.y = rand() % (src.rows - size_y);

	Mat roi = (src)(box);


/////////////////////////////////////////////////////////////////////////////////

	      //if( src.cols > 100 && src.rows > 100 )
	      //{

*/		      		
		      convert << "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/pos" << counter << "." << temp1;
		      convert1 << "pos" << counter << "." << temp1;

		      new_name = convert.str().c_str(); 
		      new_name1 = convert1.str().c_str();

		      //rename(filepath.c_str(),new_name);
		      imwrite(new_name,src);
		      //waitKey(100);	
		      counter++;

		      outputFile << new_name1 << endl;
		      cout << new_name1 << endl;

		      convert.str("");
		      convert.clear();
	 	      convert1.str("");
		      convert1.clear();



///Data Augmentation

	//rotation

			Mat tmp = src.clone();
			Mat dst = src.clone();
	
	  	for(int i=0; i<5; i++)
		{
			Point center = Point( tmp.cols/2, tmp.rows/2 );
			double angle = -3;
			double scale = 1;
			Mat rot_mat = getRotationMatrix2D( center, angle, scale );
			warpAffine( tmp, dst, rot_mat, tmp.size() );

		      convert << "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/pos" << counter << "." << temp1;
		      convert1 << "pos" << counter << "." << temp1;

		      new_name = convert.str().c_str(); 
		      new_name1 = convert1.str().c_str();

		      //rename(filepath.c_str(),new_name);
		      imwrite(new_name,dst);
		      //waitKey(100);	
		      counter++;

		      outputFile << new_name1 << endl;
		      cout << new_name1 << endl;

		      convert.str("");
		      convert.clear();
	 	      convert1.str("");
		      convert1.clear();

			tmp = dst.clone();


		}
			tmp = src.clone();
			dst = src.clone();
	
	  	for(int i=0; i<5; i++)
		{

		Point center = Point( tmp.cols/2, tmp.rows/2 );
		double angle = 3;
		double scale = 1;
		Mat rot_mat = getRotationMatrix2D( center, angle, scale );
		warpAffine( tmp, dst, rot_mat, tmp.size() );

		      convert << "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/pos" << counter << "." << temp1;
		      convert1 << "pos" << counter << "." << temp1;

		      new_name = convert.str().c_str(); 
		      new_name1 = convert1.str().c_str();

		      //rename(filepath.c_str(),new_name);
		      imwrite(new_name,dst);
		      //waitKey(100);	
		      counter++;

		      outputFile << new_name1 << endl;
		      cout << new_name1 << endl;

		      convert.str("");
		      convert.clear();
	 	      convert1.str("");
		      convert1.clear();

			tmp = dst.clone();


		}

	//Mirroring
		tmp = src.clone();
		dst = src.clone();
             // dst must be a different Mat
	      flip(tmp, dst, 1);     // because you can't flip in-place (leads to segfault)


		      convert << "/home/sprva/AAkash/TUM/TDCV/HW2/data/task3/train/c/pos" << counter << "." << temp1;
		      convert1 << "pos" << counter << "." << temp1;

		      new_name = convert.str().c_str(); 
		      new_name1 = convert1.str().c_str();

		      //rename(filepath.c_str(),new_name);
		      imwrite(new_name,dst);
		      //waitKey(100);	
		      counter++;

		      outputFile << new_name1 << endl;
		      cout << new_name1 << endl;

		      convert.str("");
		      convert.clear();
	 	      convert1.str("");
		      convert1.clear();



	//contrast -> it will have no effects -> as gradient will be uneffected 



//	}

		//}

	      //}
	//}
	    
    //fin.close();
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
	    outputFile.close();
	    return 0;
	   }

	   dirp = readdir( dp );
  
    }

 }

  closedir( dp );
  outputFile.close();

  return 0;

}
