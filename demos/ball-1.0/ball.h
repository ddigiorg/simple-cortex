// ======
// ball.h
// ======

#ifndef BALL_H
#define BALL_H

#include "utils/utils.h"

#include <vector>

class Ball
{
public:
	Ball(unsigned int sizeSceneX, unsigned int sizeSceneY, unsigned int radius)
	{
		_sizeSceneX = sizeSceneX;
		_sizeSceneY = sizeSceneY;

		_radius = radius;

		_binaryVec.resize(_sizeSceneX * _sizeSceneY);

		_resetFlag = true;
	}
 
	void step()
	{
		if (_resetFlag)
			reset();
		else
			computePhysics();

		setBinaryVector();
	}

	std::vector<unsigned char> getBinaryVector()
	{
		return _binaryVec;
	}

	bool getStartSequence()
	{
		return _startSequence;
	}

private:
	void reset()
	{
		float percent = 0.0;

		unsigned int limitX = _sizeSceneX * percent;
		unsigned int limitY = _sizeSceneY * percent;

		_positionX = (float)(utils::getRandomInt(0 + _radius + limitX, _sizeSceneX - 1 - _radius - limitX));
		_positionY = (float)(utils::getRandomInt(0 + _radius + limitY, _sizeSceneY - 1 - _radius - limitY));

		_velocityX = (float)(utils::getRandomInt(-1, 1) * 2);
		_velocityY = 0.0f;

		_acceleration = 1.0f;

		_resetFlag = false;
		_startSequence = true;
	}

	void computePhysics()
	{
		_velocityY += _acceleration;

		_positionX += _velocityX;
		_positionY += _velocityY;

		bool hitGround    = _positionY + _radius >= _sizeSceneY - 1.0f;
		bool hitWallLeft  = _positionX - _radius <  0.0f;
		bool hitWallRight = _positionX + _radius >= _sizeSceneX - 1.0f;

		if (hitGround)
		{
			_positionY = _sizeSceneY - 1.0f - _radius;

			// apply bounce friction 
			_velocityX *=  0.75f;
			_velocityY *= -0.75f;

			// check if ball has slowed down enough
			float vSquaredX = _velocityX * _velocityX;
			float vSquaredY = _velocityY * _velocityY;

			if (vSquaredX + vSquaredY < 0.25f)
				_resetFlag = true;
		}

		if (hitWallLeft)
		{
			_velocityX = -_velocityX;
			_positionX = 0.0f + _radius;
		}

		if (hitWallRight)
		{
			_velocityX = -_velocityX;
			_positionX = _sizeSceneY - 1.0f - _radius;
		}

		_startSequence = false;
	}

	void setBinaryVector()
	{
		// Clear binary vector
		for (unsigned int y = 0; y < _sizeSceneY; y++)
			for (unsigned int x = 0; x < _sizeSceneX; x++)
				_binaryVec[x + _sizeSceneX * y] = 0;

		// Set circle
		for (int y = -_radius; y <= _radius; y++)
		{
		    for (int x = -_radius; x <= _radius; x++)
			{
		        if (x * x + y * y <= _radius * _radius)
				{
					unsigned int cX = _positionX + x;
					unsigned int cY = _positionY + y;

					unsigned int i = cX + _sizeSceneX * cY;

					_binaryVec[i] = 1;
				}
			}
		}
	}

private:
	std::vector<unsigned char> _binaryVec;

	unsigned int _sizeSceneX;
	unsigned int _sizeSceneY;

	int _radius;

	float _positionX;
	float _positionY;
	float _velocityX;
	float _velocityY;
	float _acceleration;

	bool _resetFlag;
	bool _startSequence;
};

#endif
