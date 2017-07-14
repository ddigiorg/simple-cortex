// ======
// ball.h
// ======

#ifndef BALL_H
#define BALL_H

#include "utils/utils.h"

#include <vector>
#include <iostream>

class Ball
{
public:
	Ball(utils::Vec2i sizeScene)
	{
		_sizeScene = sizeScene;

		_binaryVec.resize(_sizeScene.x * _sizeScene.y);

		_resetFlag = true;
	}
 
	void step()
	{
		if (_resetFlag)
		{
//			reset();
//			resetRandom();
			resetRandomSimple();

			_startSequence = true;
		}
		else
		{
			computeSimplePhysics();
//			computeNewtonianPhysics();

			_startSequence = false;
		}

		setBinaryVector();
	}

	std::vector<char> getBinaryVector()
	{
		return _binaryVec;
	}

	bool getStartSequence()
	{
		return _startSequence;
	}

private:
	std::vector<float> _pixels;
	std::vector<char> _binaryVec;

	utils::Vec2i _sizeScene;
	utils::Vec2f _position;
	utils::Vec2f _velocity;

	float _acceleration;

	bool _resetFlag;
	bool _startSequence;

	int counter;


	void reset()
	{
		_position.x = _sizeScene.x / 2;
		_position.y = _sizeScene.y / 2;

		_velocity.x = 0.0f;
		_velocity.y = 0.0f;

		_resetFlag = true;
	}

	void resetRandom()
	{
		_position.x = (int)(utils::getRandomFloat(2.0f, (float)_sizeScene.x - 2.0f));
		_position.y = (int)(utils::getRandomFloat(2.0f, (float)_sizeScene.y - 2.0f));

		_velocity.x = (int)(2.0f - utils::getRandomFloat(0.0f, 4.0f));
		_velocity.y = (int)(2.0f - utils::getRandomFloat(0.0f, 4.0f));

		_resetFlag = false;
	}

	void resetRandomSimple()
	{
		float temp = utils::getRandomFloat(-1.0f, 1.0f);

		if (temp < 0)
			counter = 0;
		else
			counter = 1;

		if (counter == 0)
		{
			_position.x = 2.0f;
			_position.y = 5.0f;

			_velocity.x = 0.0f;
			_velocity.y = 1.0f;
		}
		else
		{
			_position.x = 6.0f;
			_position.y = 5.0f;

			_velocity.x = 0.0f;
			_velocity.y = 1.0f;
		}

//		_position.x = (int)(utils::getRandomFloat(0.0f, (float)_sizeScene.x));
//		_position.y = (int)(utils::getRandomFloat(0.0f, (float)_sizeScene.y));

//		_velocity.x = (int)(1.0f - utils::getRandomFloat(0.0f, 2.0f));
//		_velocity.y = (int)(1.0f - utils::getRandomFloat(0.0f, 2.0f));

		_resetFlag = false;
	}

	void computeSimplePhysics()
	{
		if (_velocity.y == 0.0f)
			_velocity.y = 1.0f;

		_position.x += _velocity.x;
		_position.y += _velocity.y;

		bool boundaryUpper = _position.y <= 0.0f;
		bool boundaryLower = _position.y >= _sizeScene.y - 1.0f; //2
		bool boundaryLeft  = _position.x <=  0.0f;
		bool boundaryRight = _position.x >= _sizeScene.x - 1.0f;

		if (boundaryLower)
			_velocity.y *= -1.0f;
//			_resetFlag = true;
		if (boundaryUpper)
			_resetFlag = true;
//			_velocity.y *= -1.0f;
//		if (boundaryLeft || boundaryRight)
//			_velocity.x *= -1.0f;
	}

	void computeNewtonianPhysics()
	{
		_velocity.y += _acceleration;

		_position.x += _velocity.x;
		_position.y += _velocity.y;

		bool hitGround    = _position.y >= _sizeScene.y - 1.0f;
		bool hitWallLeft  = _position.x <  0.0f;
		bool hitWallRight = _position.x >= _sizeScene.x - 1.0f;

		if (hitGround)
		{
			_position.y = _sizeScene.y - 1.0f;

			// apply bounce friction 
			_velocity.x *=  0.90f;
			_velocity.y *= -0.90f;

			// check if ball has slowed down enough
			float vSquaredX = _velocity.x * _velocity.x;
			float vSquaredY = _velocity.y * _velocity.y;

			if (vSquaredX + vSquaredY < 0.25f)
				_resetFlag = true;
		}

		if (hitWallLeft)
		{
			_velocity.x = -_velocity.x;
			_position.x = 0.0f;
		}

		if (hitWallRight)
		{
			_velocity.x = -_velocity.x;
			_position.x = _sizeScene.y - 1.0f;
		}
	}

	void setBinaryVector()
	{
		for (int y = 0; y < _sizeScene.y; y++)
		{
			for (int x = 0; x < _sizeScene.x; x++)
			{
				int i = x + _sizeScene.x * y;

				float value;

				if (x == (int)_position.x && y == (int)_position.y)
					value = 1;
				else
					value = 0;

				_binaryVec[i] = value;
			}
		}
	}
};

#endif
