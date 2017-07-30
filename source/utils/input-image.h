// =============
// input-image.h
// =============

#ifndef INPUTIMAGE_H
#define INPUTIMAGE_H

#include "utils.h"

#include <opencv2/opencv.hpp>

#include <vector>

class InputImage
{
public:
	InputImage(std::string& fileName);

	std::vector<utils::Vec4f> getPixels();
	void printInfo();

	unsigned int getWidth()
	{
		return _imageWidth;
	}

	unsigned int getHeight()
	{
		return _imageHeight;
	}

private:
	cv::Mat _image;

	std::string _fileName;

	int _imageWidth;
	int _imageHeight;
};

#endif
