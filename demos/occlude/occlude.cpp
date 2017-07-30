// ===========
// occlude.cpp
// ===========

#include "occlude.h"
#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "utils/utils.h"
#include "utils/input-image.h"
#include "utils/render2d.h"
#include "cortex/pattern.h"
#include "cortex/area.h"

#include <vector>

int main()
{
	// Setup SFML render window and ball simulation
	unsigned int sizeSceneX = 100;
	unsigned int sizeSceneY = 100;
	unsigned int scaleScene = 2;
	unsigned int sizeDisplayX = sizeSceneX * scaleScene;
	unsigned int sizeDisplayY = sizeSceneY * scaleScene;
	unsigned int numPixels = sizeSceneX * sizeSceneY;

	sf::RenderWindow window;
	window.create(sf::VideoMode(sizeDisplayX, sizeDisplayY), "Simple Cortex - Ball Demo 2.0", sf::Style::Default); // title

	Render2D scene(sizeSceneX, sizeSceneY);
	scene.setPosition(sizeDisplayX / 2, sizeDisplayY / 2);
	scene.setScale((float) scaleScene);

	std::vector<float> rVec(numPixels);
	std::vector<float> gVec(numPixels);
	std::vector<float> bVec(numPixels);

	// Setup input images
	std::string fileBox      = "resources/box.png";
	std::string fileCircle   = "resources/circle.png";
	std::string fileCircle2  = "resources/circle2.png";
	std::string fileDuck     = "resources/duck.png";
	std::string fileDude     = "resources/dude.png";
	std::string fileHeart    = "resources/heart.png";
	std::string fileSquare   = "resources/square.png";
	std::string fileTriangle = "resources/triangle.png";
	std::string fileUparrow  = "resources/uparrow.png";
	std::string fileX        = "resources/x.png";

	InputImage imgBox(fileBox);


	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;

	std::string kernels_cl = "source/cortex/area.cl";

	cs.init(ComputeSystem::_gpu);
	cs.printCLInfo();
	cp.loadFromSourceFile(cs, kernels_cl);

	// Setup Simple Cortex Area
	unsigned int numNeurons = 20;
//	unsigned int numForecasts = 20;

//	std::vector<unsigned char> resetNeuronsVec(numNeurons);
//	resetNeuronsVec[numNeurons - 1] = 1;

	std::vector<Pattern> patterns(4);
	patterns[0].init(cs, numPixels);  // input - current binary scene state
	patterns[1].init(cs, numNeurons); // input - previous neuron activations
	patterns[2].init(cs, numNeurons); // input - storage of current neuron activations for forecasting
	patterns[3].init(cs, numPixels);  // output - predicted future binary scene state

	Area area;
	area.init(cs, cp, numNeurons, {50, 1});

	// Render loop
	bool quit = false;
	bool step = true;
	bool pause = false;

	printf("\nPress 'Space' to step algorithms");
	printf("\nPress 'p' to pause/unpause running algorithms");
	printf("\nPress 'Esc' to quit application");
	printf("\n");

	while (!quit)
	{
		// Handle SFML window events
		sf::Event windowEvent;
		while (window.pollEvent(windowEvent))
		{
			if (windowEvent.type == sf::Event::Closed)
				quit = true;

			if (windowEvent.type == sf::Event::KeyPressed)
			{
				switch (windowEvent.key.code)
				{
					case sf::Keyboard::Escape:
						quit = true;
						break;

					case sf::Keyboard::Space:
						step = true;
						break;

					case sf::Keyboard::P:
						if (pause == true)
							pause = false;
						else
							pause = true;
						break;
				}
			}
		}

		if (step || pause)
		{
			for (unsigned int p = 0; p < numPixels; p++)
			{
				rVec[p] = 0.0f;
				gVec[p] = 0.0f;
				bVec[p] = 0.0f;
			}

			/*
			patterns[0].setStates(cs, ball.getBinaryVector());

			if (ball.getStartSequence())
				patterns[1].setStates(cs, resetNeuronsVec);

			area.encode(cs, {patterns[0], patterns[1]});
			area.learn(cs, {patterns[0], patterns[1]});

			patterns[1].setStates(cs, area.getStates(cs));

			// Forecast the future
			for (unsigned int i = 0; i < numForecasts; i++)
			{
				patterns[2].setStates(cs, area.getStates(cs));

				area.predict(cs, {patterns[2]}, {1});
				area.decode(cs, {patterns[3]}, {0});

				std::vector<unsigned char> prediction = patterns[3].getStates(cs);

				for (unsigned int p = 0; p < numPixels; p++)
				{
					if (prediction[p] > 0)
						bVec[p] = 0.2f + 0.04f * i;
				}
			}
			*/

//			std::vector<unsigned char> input = patterns[0].getStates(cs);
			std::vector<utils::Vec4f> test = imgBox.getPixels();

			for (unsigned int p = 0; p < numPixels; p++)
			{
//				if (input[p] > 0)
					gVec[p] = test[p].g;
			}

			scene.setPixels(rVec, gVec, bVec);

			window.clear(sf::Color::Black);
			window.draw(scene.getSprite());

			step = false;
		}

		window.display();
	}
}
