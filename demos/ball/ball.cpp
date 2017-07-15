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
	unsigned int numNeurons = 1000;

	std::vector<unsigned int> numVperP(4); // 4 patterns
	std::vector<unsigned int> numSperD(2); // 2 dendrites (per neuron)

	// Patterns
	numVperP = {numPixels,  // input - current scene state
                numNeurons, // input - previous active neurons
                numNeurons, // input - previous winner neurons
                numPixels}; // output - predicted future scene state

	// Dendrites
	numSperD = {1,  // stores current active scene state values
                1}; // stores previous winner neurons

	Region region(cs, cp, numNeurons, numVperP, numSperD);

	std::vector<unsigned int> pEncode(2);
	pEncode = {0, 1};

	std::vector<unsigned int> pLearn(2);
	pLearn = {0, 2};

	std::vector<char> resetVec(numNeurons);
	resetVec[numNeurons - 1] = 1;

	// Loop
	bool quit = false;
	bool pause = false;
	int counter = 0;

//	sf::Clock clock;
//	sf::Time time;

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
			region.setPatternFromWinnerNeurons(cs, 2);

			if (ball.getStartSequence())
			{
				region.setPattern(cs, 1, resetVec);
				region.setPattern(cs, 2, resetVec);
			}

//			clock.restart();
//			time = clock.getElapsedTime();

			region.encode(cs, pEncode);
			region.learn(cs, pLearn);
			region.predict(cs);
			region.decode(cs);

//			time = clock.getElapsedTime();

//			region.print(cs);

//			printf("\nTime(us): %i", time.asMicroseconds()); 

			window.clear(sf::Color::Black);

//			scene.setPixelsFromBinaryVector('g', false, ball.getBinaryVector());
			scene.setPixelsFromBinaryVector('g', false, region.getGoodPrediction(cs));
			scene.setPixelsFromBinaryVector('r', false, region.getBadPrediction(cs));
			scene.setPixelsFromBinaryVector('b', false, region.getPattern(cs, 3));

			window.draw(scene.getSprite());

			pause = true;
		}

		window.display();
	}
}
