// ===============
// input-video.cpp
// ===============

#include "input-video.h"

#include <iostream>

using namespace input;

InputVideo::InputVideo()
{

}

InputVideo::InputVideo(std::string& fileName)
{
	_fileName = fileName;

	_vidcap.open(_fileName);

	if (!_vidcap.isOpened())
		std::cerr << "[i] Could not open capture: " << _fileName << std::endl;

	_vidcapLength = _vidcap.get(cv::CAP_PROP_FRAME_COUNT);
	_vidcapWidth  = _vidcap.get(cv::CAP_PROP_FRAME_WIDTH);
	_vidcapHeight = _vidcap.get(cv::CAP_PROP_FRAME_HEIGHT);
	_vidcapFPS    = _vidcap.get(cv::CAP_PROP_FPS);

}

InputVideo::~InputVideo()
{

}

std::vector<utils::Vec4f> InputVideo::getNextFramePixels()
{
	cv::Mat frame;

	_vidcap >> frame;

	if(frame.empty())
	{
		_vidcap.set(cv::CAP_PROP_POS_AVI_RATIO , 0);
		_vidcap >> frame;
	}

	std::vector<utils::Vec4f> frameRGBA(frame.cols * frame.rows);

	int c = frame.channels();
	int num_pixels = frame.cols * frame.rows;

	for (int i = 0; i < num_pixels; i++)
	{
		frameRGBA[i].x = static_cast<float>(frame.data[i * c + 2]) / 255.0f; // r
		frameRGBA[i].y = static_cast<float>(frame.data[i * c + 1]) / 255.0f; // g
		frameRGBA[i].z = static_cast<float>(frame.data[i * c + 0]) / 255.0f; // b
		frameRGBA[i].w = 1.0f;                                               // a
	}

	return frameRGBA;
}

void InputVideo::run()
{

	cv::Mat frame;

	cv::namedWindow("window", CV_WINDOW_AUTOSIZE);

	while (_vidcap.isOpened())
	{
		_vidcap >> frame; // get frame from video
		if (frame.empty())
			break;

			cv::imshow("window", frame);
			cv::waitKey(20);
	}
}

void InputVideo::printInfo()
{
	std::cout << "[i] video file: "   << _fileName     << std::endl;
	std::cout << "[i] video length: " << _vidcapLength << std::endl;
	std::cout << "[i] video width: "  << _vidcapWidth  << std::endl;
	std::cout << "[i] video height: " << _vidcapHeight << std::endl;
	std::cout << "[i] video fps: "    << _vidcapFPS    << std::endl;
}
