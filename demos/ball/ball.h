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
	Ball(
		unsigned int sizeSceneX, 
		unsigned int sizeSceneY, 
		unsigned int ballRadius,
		unsigned int posType,
		unsigned int velType)
	{
		_sizeSceneX = sizeSceneX;
		_sizeSceneY = sizeSceneY;
		_radius = ballRadius;
		_posType = posType;
		_velType = velType;

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
		switch (_posType)
		{
			// Ball Position - Middle of Screen
			case 0:
				_positionX = (float)(_sizeSceneX / 2);
				_positionY = (float)(_sizeSceneY / 2);
				break;

			// Ball Position - Random Ints
			case 1:
				unsigned int limitX = _sizeSceneX * 0.25;
				unsigned int limitY = _sizeSceneX * 0.25;

				_positionX = (float)(utils::getRandomInt(0 + _radius + limitX, _sizeSceneX - 1 - _radius - limitX));
				_positionY = (float)(utils::getRandomInt(0 + _radius + limitY, _sizeSceneY - 1 - _radius - limitY));
				break;
		}

		switch (_velType)
		{
			// Ball Velocity - Zero
			case 0:
				_velocityX = 0;
				_velocityY = 0;
				break;

			// Ball Velocity - Random Ints
			case 1:
				_velocityX = (float)(utils::getRandomInt(-1, 1) * 5);
				_velocityY = (float)(utils::getRandomInt(-1, 1) * 5);
				break;

			// Ball Velocity - Random Floats
			case 2:
				_velocityX = utils::getRandomFloat(-5.0f, 5.0f);
				_velocityY = utils::getRandomFloat(-5.0f, 5.0f);
				break;
		}

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
	unsigned int _posType;
	unsigned int _velType;

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
