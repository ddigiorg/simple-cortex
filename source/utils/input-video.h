// =============
// input-video.h
// =============

#ifndef INPUTVIDEO_H
#define INPUTVIDEO_H

#include "../utils/utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

namespace input
{
	class InputVideo
	{
	public:
		InputVideo();
		InputVideo(std::string& fileName);
		~InputVideo();

		std::vector<utils::Vec4f> getNextFramePixels();
		void run();
		void printInfo();

		unsigned int getWidth()
		{
			return _vidcapWidth;
		}

		unsigned int getHeight()
		{
			return _vidcapHeight;
		}

	private:
		cv::VideoCapture _vidcap;

		std::string _fileName;

		int _vidcapLength;
		int _vidcapWidth;
		int _vidcapHeight;
		int _vidcapFPS;
	};
}

#endif
