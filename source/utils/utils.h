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

	inline int getRandomInt(int min, int max)
	{
		int ran = rand() % (max - min + 1) + min;

		return ran;
	}
}

#endif
