// ========
// ball.cpp
// ========

#include "ball.h"
#include "utils/render2d.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "app/region.h"

#include <vector>
#include <random>
#include <time.h>

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

	std::string kernels_cl = "source/app/region.cl";

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

	std::vector<unsigned int> numVperP(4); // 4 patterns
	std::vector<unsigned int> numSperD(2); // 2 dendrites (per neuron)

	// Patterns
	numVperP = {numPixels,  // input - current binary scene state
                numNeurons, // input - previous neuron activations
                numPixels}; // output - predicted future binary scene state

	// Dendrites
	numSperD = {1,  // learns current ball location address
                1}; // learns previous active neuron address

	Region region(cs, cp, numNeurons, numVperP, numSperD);

	std::vector<char> resetPreActNeuVec(numNeurons);
	resetPreActNeuVec[numNeurons - 1] = 1;

	// Loop
	bool quit = false;
	bool pause = false;
	int counter = 0;

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
			ball.step();

			region.setPattern(cs, 0, ball.getBinaryVector());
			region.setPatternFromActiveNeurons(cs, 1);

			if (ball.getStartSequence())
			{
				region.setPattern(cs, 1, resetPreActNeuVec);
				region.setPattern(cs, 2, resetPreActNeuVec);
			}

			region.encode(cs, {0, 1}, {0, 1});
			region.learn(cs, {0, 1}, {0, 1});
			region.predict(cs, {1}, {1});
			region.decode(cs, {2}, {0});

			//PUT FORECASTING HERE

			window.clear(sf::Color::Black);

			scene.setPixelsFromBinaryVector('g', false, region.getPattern(cs, 0));
			scene.setPixelsFromBinaryVector('b', false, region.getPattern(cs, 2));

			window.draw(scene.getSprite());

			pause = true;
		}

		window.display();
	}
}
