// ========
// test.cpp
// ========

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

	// Setup SFML render window
	sf::RenderWindow window;
	utils::Vec2i displaySize(400, 300);
	window.create(sf::VideoMode(displaySize.x, displaySize.y), "HTM - Test", sf::Style::Default);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;
	std::string kernels_cl = "source/app/region.cl";
	cs.init(ComputeSystem::_gpu);
	//cs.printCLInfo();
	cp.loadProgramFromSourceFile(cs, kernels_cl);  // change to loadFromSourceFile

	// Setup HTM Region
	unsigned int numI     = 8; //2500 number of inputs
	unsigned int numC     = 2; //2048 number of columns
	unsigned int numNpC   = 2; //40 number of neurons per column
	unsigned int numDDpN  = 2; //10 number of distal dendrites per neuron
	unsigned int numDSpDD = 2; //128 number of distal synapses per distal dendrite

	std::vector<char> inputs0(numI);
	std::vector<char> inputs1(numI);

	inputs0 = {1, 1, 0, 0, 0, 0, 0, 0};
	inputs1 = {0, 0, 1, 1, 0, 0, 0, 0};

	Region region(cs, cp, rng, numI, numC, numNpC, numDDpN, numDSpDD);
//	region.initSynapsesRandom(cs);
//	region.initSynapsesTest(cs);

	// Loop
	bool quit = false;
	bool pause = false;
	int counter = 0;

	sf::Clock clock;
	sf::Time spTime;
	sf::Time tmTime;

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
			if (counter == 0)
			{
				region.setInputs(cs, inputs0);
				counter = 1;
			}
			else
			{
				region.setInputs(cs, inputs1);
				counter = 0;
			}

			clock.restart();
			region.sp(cs, true);
			spTime = clock.getElapsedTime();

			clock.restart();
			region.tm(cs, true);
			tmTime = clock.getElapsedTime();

			region.print(cs);
			std::cout << "SP(us): " << spTime.asMicroseconds();
			std::cout << std::endl << "TM(us): " << tmTime.asMicroseconds();
			std::cout << std::endl << "Total(us): " << spTime.asMicroseconds() + tmTime.asMicroseconds() << std::endl;

			window.clear(sf::Color::Black);

			pause = true;
		}

		window.display();
	}
}
