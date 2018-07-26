#ifndef LIGHT_FIELD_H_
#define LIGHT_FIELD_H_

#define M_PI       3.14159265358979323846   // pi

#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<string>
#include<vector>
#include<numeric>
#include<cmath>
#include<iostream>
#include<boost\filesystem.hpp>
namespace fs = boost::filesystem;

// Global Variables
std::string windowName;
std::vector<cv::Mat> images;
cv::Mat dstImage;
int sideNum;
double s;
double t;
int sliderDisp = 3;
int sliderAper = 0;
double sigma = 1;

void readImage(std::string imPath)
{
	for (auto i = fs::directory_iterator(imPath); i != fs::directory_iterator(); i++)
	{
		if (!fs::is_directory(i->path()))
		{
			std::string filename = imPath + i->path().filename().string();
			cv::Mat image = cv::imread(filename);
			images.push_back(image);
		}
	}
	sideNum = sqrt(images.size());
	dstImage = cv::Mat::zeros(images[0].size(), images[0].type());
}

void findCams(int &camCount, std::vector<int> &camIds, std::vector<double> &camWeights, std::vector<double> &sFlags, std::vector<double> &tFlags)
{
	if (0 == sliderAper)
	{
		int side = 2;
		for (int i = 0; i < side; i++)
			for (int j = 0; j < side; j++)
			{
				int sId = floor(s) + i;
				int tId = floor(t) + j;
				camIds.push_back(tId * sideNum + sId);
				camWeights.push_back((1 - abs(s - sId)) * (1 - abs(t - tId)));
				int temp = sId < s ? -1 : 1;
				sFlags.push_back((sId == s ? 0 : (sId < s ? -1 : 1)) * abs(s - sId));

				temp = tId < t ? 1 : -1;
				tFlags.push_back((tId == t ? 0 : (tId < t ? 1 : -1)) * abs(t - tId));
				camCount++;
				std::cout << "camId: " << camIds[i * 2 + j] << "; camWeight: " << camWeights[i * 2 + j]
					<< "; sFlag: " << sFlags[i * 2 + j] << "; tFlag: " << tFlags[i * 2 + j] << std::endl;
			}
	}
}

void BilinearInterpolation()
{
	int width = images[0].cols;
	int height = images[0].rows;
	cv::Mat sumweight = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat sumImage0 = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat sumImage1 = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat sumImage2 = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < images.size(); i++)
	{
		int sId = sideNum - i % sideNum - 1;
		int tId = floor(i / sideNum);
		double dist = sqrt((sId - s) * (sId - s) + (tId - t)*(tId - t));
		if (abs(sId - s) < 1 && abs(tId - t) < 1)
		{
			double weight = (1 - abs(sId - s)) * (1 - abs(tId - t));
			double sk = sId - s;
			double tk = tId - t;
			for (int row = 0; row < images[0].rows; row++)
			{
				for (int col = 0; col < images[0].cols; col++)
				{
					int dist_row = row + floor(tk*sliderDisp);
					int dist_col = col + floor(sk*sliderDisp);
					if (dist_row >= 0 && dist_row < images[0].rows && dist_col >= 0 && dist_col < images[0].cols)
					{
						sumweight.at<float>(dist_row, dist_col) += weight;
						sumImage0.at<float>(dist_row, dist_col) += weight * images[i].at<cv::Vec3b>(row, col)[0];
						sumImage1.at<float>(dist_row, dist_col) += weight * images[i].at<cv::Vec3b>(row, col)[1];
						sumImage2.at<float>(dist_row, dist_col) += weight * images[i].at<cv::Vec3b>(row, col)[2];
					}
				}
			}
		}
	}
	sumImage0 = sumImage0 / sumweight;
	sumImage1 = sumImage1 / sumweight;
	sumImage2 = sumImage2 / sumweight;
	for (int row = 0; row < images[0].rows; row++)
	{
		for (int col = 0; col < images[0].cols; col++)
		{
			dstImage.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(sumImage0.at<float>(row, col));
			dstImage.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(sumImage1.at<float>(row, col));
			dstImage.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(sumImage2.at<float>(row, col));
		}
	}
}

void GaussionInterpolation()
{
	int width = images[0].cols;
	int height = images[0].rows;
	cv::Mat sumweight = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat sumImage0 = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat sumImage1 = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat sumImage2 = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < images.size(); i++)
	{
		int sId = sideNum - i % sideNum - 1;
		int tId = floor(i / sideNum);
		double dist = sqrt((sId - s) * (sId - s) + (tId - t)*(tId - t));
		if (dist < sliderAper)
		{
			double weight = 1 / (sigma * sqrt(2 * M_PI)) * exp(dist * dist / (-2.0 * sigma * sigma));
			double sk = sId - s;
			double tk = tId - t;
			for (int row = 0; row < images[0].rows; row++)
			{
				for (int col = 0; col < images[0].cols; col++)
				{
					int dist_row = row + floor(tk*sliderDisp);
					int dist_col = col + floor(sk*sliderDisp);
					if (dist_row >= 0 && dist_row < images[0].rows && dist_col >= 0 && dist_col < images[0].cols)
					{
						sumweight.at<float>(dist_row, dist_col) += weight;
						sumImage0.at<float>(dist_row, dist_col) += weight * images[i].at<cv::Vec3b>(row, col)[0];
						sumImage1.at<float>(dist_row, dist_col) += weight * images[i].at<cv::Vec3b>(row, col)[1];
						sumImage2.at<float>(dist_row, dist_col) += weight * images[i].at<cv::Vec3b>(row, col)[2];
					}
				}
			}
		}
	}
	sumImage0 = sumImage0 / sumweight;
	sumImage1 = sumImage1 / sumweight;
	sumImage2 = sumImage2 / sumweight;
	for (int row = 0; row < images[0].rows; row++)
	{
		for (int col = 0; col < images[0].cols; col++)
		{
			dstImage.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(sumImage0.at<float>(row, col));
			dstImage.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(sumImage1.at<float>(row, col));
			dstImage.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(sumImage2.at<float>(row, col));
		}
	}
}

void render()
{	
	if (0 == sliderAper)
		BilinearInterpolation();
	else
		GaussionInterpolation();
}


static void onTrackDisp(int Disp, void*)
{
	render();
	cv::imshow(windowName, dstImage);
}

static void onTrackAper(int Aper, void*)
{
	render();
	cv::imshow(windowName, dstImage);
}

static void mouseCallBack(int event, int x, int y, int flags, void* userdata)
{
	int width = images[0].cols;
	int height = images[0].rows;
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		s = double(x) / width * (sideNum - 1);
		t = double(y) / height * (sideNum - 1);
		std::cout << "s: " << s << "; t: " << t << std::endl;
		render();
		cv::imshow(windowName, dstImage);
	}
}

void initWindow(std::string name)
{
	windowName = name;
	const int sliderDispMax = 12;
	const int sliderAperMax = 8;
	cv::namedWindow(name);
	cv::imshow(name, images[0]);
	cv::setMouseCallback(name, mouseCallBack, NULL);
	cv::createTrackbar("Disparity", name, &sliderDisp, sliderDispMax, onTrackDisp);
	cv::createTrackbar("Aperture", name, &sliderAper, sliderAperMax, onTrackAper);
	cv::waitKey();
}


#endif // LIGHT_FIELD_H_
