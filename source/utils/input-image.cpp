// ===============
// input-image.cpp
// ===============

#include "input-image.h"

#include <iostream>

InputImage::InputImage(std::string& fileName)
{
	_fileName = fileName;

	_image = cv::imread(_fileName, CV_LOAD_IMAGE_COLOR);

	if (!_image.data)
		std::cout << "[i] Could not open image" << _fileName << std::endl;
		
	_imageWidth = _image.cols;
	_imageHeight = _image.rows;
}

std::vector<utils::Vec4f> InputImage::getPixels()
{
	std::vector<utils::Vec4f> imageRGBA(_image.cols * _image.rows);

	int c = _image.channels();
	int num_pixels = _image.cols * _image.rows;

	for (int i = 0; i < num_pixels; i++)
	{
		imageRGBA[i].r = static_cast<float>(_image.data[i * c + 2]) / 255.0f; // r
		imageRGBA[i].g = static_cast<float>(_image.data[i * c + 1]) / 255.0f; // g
		imageRGBA[i].b = static_cast<float>(_image.data[i * c + 0]) / 255.0f; // b
		imageRGBA[i].a = 1.0f;                                                // a
	}

	return imageRGBA;
}

void InputImage::printInfo()
{
	std::cout << "[i] image file: "   << _fileName    << std::endl;
	std::cout << "[i] image width: "  << _imageWidth  << std::endl;
	std::cout << "[i] image height: " << _imageHeight << std::endl;
}
