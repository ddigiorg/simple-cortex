// ========
// ball.cpp
// ========

#include "ball.h"
#include "utils/render2d.h"
//#include "utils/text2d.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "app/region.h"

#include <iostream>
#include <vector>
#include <random>
#include <time.h>

int main()
{
	std::mt19937 rng(time(nullptr)); 
	srand(time(NULL)); 

	int scale = 20;

	utils::Vec2i sizeScene(11, 11);
	utils::Vec2i sizeDisplay(sizeScene.x * scale, sizeScene.y * scale);

	// Setup SFML render window
	sf::RenderWindow window;
	window.create(sf::VideoMode(sizeDisplay.x, sizeDisplay.y), "Simple Cortex - Ball Demo", sf::Style::Default);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;
	std::string kernels_cl = "source/app/region.cl";
	cs.init(ComputeSystem::_gpu);
	//cs.printCLInfo();
	cp.loadProgramFromSourceFile(cs, kernels_cl);  // change to loadFromSourceFile

	// Setup Ball Simulation
	Ball ball(sizeScene);
	Render2D scene(sizeScene);

	scene.setPosition(utils::Vec2i(sizeDisplay.x / 2, sizeDisplay.y / 2));
	scene.setScale((float) scale);

	// Setup Simple Cortex area
	unsigned int numPixels = sizeScene.x * sizeScene.y;
	unsigned int numN = 50; // number of neurons

	std::vector<unsigned int> numVperP(4); // 4 patterns
	std::vector<unsigned int> numSperD(2); // 2 dendrites (per neuron)

	numVperP = {numPixels,  // input - current scene state
                numN,       // input - previous active neurons
                numN,       // input - previous winner neurons
                numPixels}; // output - predicted future scene state

	numSperD = {1,  // stores current active scene state values
                1}; // stores previous winner neurons

	Region region(cs, cp, rng, numN, numVperP, numSperD);

	std::vector<unsigned int> pEncode(2);
	std::vector<unsigned int> pLearn(2);

	pEncode = {0, 1};
	pLearn = {0, 2};

	// Loop
	bool quit = false;
	bool pause = false;
	int counter = 0;

	sf::Clock clock;
	sf::Time activateTime;
//	sf::Time tmTime;

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
			region.copyActiveNeuronsToPattern(cs, 1);
			region.copyWinnerNeuronsToPattern(cs, 2);

			region.encode(cs, pEncode);
			region.learn(cs, pLearn);
			region.predict(cs);
			region.decode(cs);

//			clock.restart();
//			region.activate(cs, true);
//			activateTime = clock.getElapsedTime();

			region.print(cs);

//			std::cout << "Activate(us): " << activateTime.asMicroseconds() << std::endl;
//			std::cout << std::endl << "TM(us): " << tmTime.asMicroseconds();
//			std::cout << std::endl << "Total(us): " << spTime.asMicroseconds() + tmTime.asMicroseconds() << std::endl;

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
