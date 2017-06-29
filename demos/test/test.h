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

		_pixelData.resize(_sizeScene.x * _sizeScene.y);

		_sizeBall.x = 1;
		_sizeBall.y = 1;

		_radius = 1;

		reset();
	}
 
	void reset()
	{
		_position.x = _sizeScene.x / 2;
		_position.y = _sizeScene.y / 2;

//		_position.x = (int)(utils::getRandomFloat(12.0f,  36.0f));
//		_position.y = (int)(utils::getRandomFloat(10.0f, 35.0f));

		_velocity.x = 0.0f;
		_velocity.y = 0.0f;

//		_velocity.x = (int)(2.0f - utils::getRandomFloat(0.0f, 4.0f));
//		_velocity.y = (int)(2.0f - utils::getRandomFloat(0.0f, 4.0f));
	}

	void step()
	{
		for (int y = 0; y < _sizeScene.y; y++)
		{
			for (int x = 0; x < _sizeScene.x; x++)
			{
				int i = x + _sizeScene.x * y;

				float value;

				// border
				value = (x == 0 || x == _sizeScene.x - 1 || y == 0 || y == _sizeScene.y - 1) ? 1.0f : 0.0f;

				// ball position
				if (
					x >= (int)_position.x - _radius &&
					x <  (int)_position.x + _radius &&
					y >= (int)_position.y - _radius &&
					y <  (int)_position.y + _radius)
					{
						value = 1.0f;
					}

					_pixelData[i] = value;
			}
		}

		_velocity.y += _acceleration;

		_position.x += _velocity.x;
		_position.y += _velocity.y;

		bool hitGround    = _position.y + _radius >= _sizeScene.y - 1;
		bool hitWallLeft  = _position.x + _radius <  1;
		bool hitWallRight = _position.x + _radius >= _sizeScene.x - 1;

		if (hitWallLeft)
		{
			_velocity.x = -_velocity.x;
			_position.x = 1 + _radius;
		}

		if (hitWallRight)
		{
			_velocity.x = -_velocity.x;
			_position.x = _sizeScene.y - 1 - _radius;
		}

		if (hitGround)
		{
			float vSquaredX = _velocity.x * _velocity.x;
			float vSquaredY = _velocity.y * _velocity.y;

			if (vSquaredX < 1.0 && vSquaredY < 2.0)
			{
				reset();
			}
			else
			{
				_velocity.x *=  0.75f;
				_velocity.y *= -0.75f;
				_position.y = _sizeScene.y - 1.0f - _radius;
			}
		}
		
	}

	std::vector<float> getPixelData()
	{
		return _pixelData;
	}

private:
	std::vector<float> _pixelData;

	utils::Vec2i _sizeScene;
	utils::Vec2i _sizeBall;

	utils::Vec2f _position;
	utils::Vec2f _velocity;
	float _acceleration = 1.0f;

	int _radius;
};

#endif
