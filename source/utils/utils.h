// =======
// utils.h
// =======

#ifndef UTILS_H
#define UTILS_H

#include "compute/compute-system.h"

#include <random>

namespace utils
{
	typedef struct Vec2i
	{
		int x, y;

		Vec2i(){};

		Vec2i(int initX, int initY)
		{
			x = initX;
			y = initY;
		}
	} Vec2i;

	typedef struct Vec3i
	{
		int x, y, z;

		Vec3i(){};

		Vec3i(int initX, int initY, int initZ)
		{
			x = initX;
			y = initY;
			z = initZ;
		}
	} Vec3i;

	typedef struct Vec2f
	{
		float x, y;

		Vec2f(){};

		Vec2f(float initX, float initY)
		{
			x = initX;
			y = initY;
		}
	} Vec2f;

	typedef struct Vec4f
	{
		float r, b, g, a;

		Vec4f(){};

		Vec4f(float initR, float initB, float initG, float initA)
		{
			r = initR;
			g = initG;
			b = initB;
			a = initA;
		}
	} Vec4f;

	inline float getRandomFloat(float min, float max)
	{
		float ran = (float) rand() / RAND_MAX;
		return min + (max - min) * ran;
	}

//	inline cl::Buffer createBuffer(ComputeSystem &cs, 
//	{
//		return cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, 
//	}

	inline cl::Image1D createImage1D(ComputeSystem &cs, cl_int  size, cl_channel_order channelOrder, cl_channel_type channelType)
	{	
		return cl::Image1D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size);
	}

	inline cl::Image2D createImage2D(ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType)
	{	
		return cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y);
	}

	inline cl::Image3D createImage3D(ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType)
	{
		return cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y, size.z);
	}
}

#endif
