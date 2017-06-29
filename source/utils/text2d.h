// ========
// text2d.h
// ========

#ifndef TEXT2D_H
#define TEXT2D_H

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <string>

class Text2D
{
public:
	Text2D(utils::Vec2i size)
	{
		_size = size;

		_font.loadFromFile("resources/FreeMono.ttf");

		_text.setFont(_font);
		_text.setCharacterSize(12);
		_text.setFillColor(sf::Color::White);
		_text.setOutlineColor(sf::Color::White);
	}

	void setText(std::vector<float> textData)
	{
		std::string textString;

		for (int y = 0; y < _size.y; y++)
		{
			for (int x = 0; x < _size.x; x++)
			{
 				unsigned int i = x + _size.x * y;

//				if (textData[i] > 9)
					textString += std::to_string(static_cast<int>(textData[i])) + " ";
//				else
//					textString += " " + std::to_string(static_cast<int>(textData[i])) + " ";
			}

			textString += "\n";
		}

		_text.setString(textString);
	}

	void setPosition(utils::Vec2i position)
	{   
		_text.setOrigin(sf::Vector2f(108.0f, 108.0f));
		_text.setPosition(position.x, position.y);
	}

	void setScale(float scale)
	{
		_text.setScale(sf::Vector2f(scale, scale));
	}

	sf::Text getText()
	{   
		return _text;
	}

private:
	utils::Vec2i _size;

	sf::Font _font;
	sf::Text _text;
};

#endif
