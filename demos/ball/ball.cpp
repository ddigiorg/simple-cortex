// ========
// ball.cpp
// ========

#include "ball.h"
#include "utils/render2d.h"
#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "cortex/area.h"

#include <vector>

int main()
{
	int scale = 20;

	utils::Vec2i sizeScene(21, 21);
	utils::Vec2i sizeDisplay(sizeScene.x * scale, sizeScene.y * scale);

	// Setup SFML render window
	sf::RenderWindow window;
	window.create(sf::VideoMode(sizeDisplay.x, sizeDisplay.y), "Simple Cortex - Ball Demo", sf::Style::Default);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;

	std::string kernels_cl = "source/cortex/area.cl";

	cs.init(ComputeSystem::_gpu);
	cs.printCLInfo();
	cp.loadProgramFromSourceFile(cs, kernels_cl);  // change to loadFromSourceFile

	// Setup Ball Simulation
	Ball ball(sizeScene);
	Render2D scene(sizeScene);

	scene.setPosition(utils::Vec2i(sizeDisplay.x / 2, sizeDisplay.y / 2));
	scene.setScale((float) scale);

	// Setup Simple Cortex Area
	unsigned int numPixels = sizeScene.x * sizeScene.y;
	unsigned int numNeurons = 65000; // note: cant go over 65535 neurons unless _sAddrMax is modified
	unsigned int numForecasts = 10;

	std::vector<unsigned int> numVperP(4); // 4 patterns
	std::vector<unsigned int> numSperD(2); // 2 dendrites (per neuron)

	// Patterns
	numVperP = {numPixels,  // input - current binary scene state
                numNeurons, // input - previous neuron activations
				numNeurons, // input - storage of current neuron activations or predictions for forecasting
                numPixels}; // output - predicted future binary scene state

	// Dendrites
	numSperD = {1,  // learns current ball location address
                1}; // learns previous active neuron address

	Area area(cs, cp, numNeurons, numVperP, numSperD);

	std::vector<char> resetPreActNeuVec(numNeurons);
	resetPreActNeuVec[numNeurons - 1] = 1;

	// Color vectors
	std::vector<float> r(numPixels);
	std::vector<float> g(numPixels);
	std::vector<float> b(numPixels);

	// Loop
	bool quit = false;
	bool pause = false;

	while (!quit)
	{
		// Handle SFML window events
		sf::Event windowEvent;
		while (window.pollEvent(windowEvent))
		{
			if (windowEvent.type == sf::Event::Closed)
				quit = true;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
				quit = true;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
				pause = false;
		}
 
		if (!pause)
		{
			// Reset color vectors
			for (unsigned int p = 0; p < numPixels; p++)
			{
				r[p] = 0.0f;
				g[p] = 0.0f;
				b[p] = 0.0f;
			}

			// Step ball simulation
			ball.step();

			// Set cortical inputs
			area.setPattern(cs, 0, ball.getBinaryVector());
			area.setPatternFromActiveNeurons(cs, 1);

			if (ball.getStartSequence())
				area.setPattern(cs, 1, resetPreActNeuVec);

			// Activate neurons and learn
			area.encode(cs, {0, 1}, {0, 1});
			area.learn(cs, {0, 1}, {0, 1});

			// Forecast the future
			for (unsigned int i = 0; i < numForecasts; i++)
			{
				if (i == 0)
					area.setPatternFromActiveNeurons(cs, 2);
				else
					area.setPatternFromPredictNeurons(cs, 2);

				area.predict(cs, {2}, {1});
				area.decode(cs, {3}, {0});

				std::vector<char> prediction = area.getPattern(cs, 3);

				for (unsigned int p = 0; p < numPixels; p++)
				{
					if (prediction[p] > 0)
						b[p] = 1.0f - 0.1f * i;
				}
			}

			std::vector<char> input = area.getPattern(cs, 0);

			for (unsigned int p = 0; p < numPixels; p++)
			{
				if (input[p] > 0)
					g[p] = 1.0f;
			}

			scene.setPixels(r, g, b);

			window.clear(sf::Color::Black);
			window.draw(scene.getSprite());

			pause = true;
		}

		window.display();
	}
}
