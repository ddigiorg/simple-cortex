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
	unsigned int numIn0   = 4;
	unsigned int numIn1   = 4;
	unsigned int numN     = 4;
	unsigned int numSpD0  = 1;
	unsigned int numSpD1  = 1;

	std::vector<char> inputs00(numIn0);
	std::vector<char> inputs10(numIn1);
	std::vector<char> inputs01(numIn0);
	std::vector<char> inputs11(numIn1);
	std::vector<char> inputs02(numIn0);
	std::vector<char> inputs12(numIn1);
	std::vector<char> inputs03(numIn0);
	std::vector<char> inputs13(numIn1);
	std::vector<char> inputs04(numIn0);
	std::vector<char> inputs14(numIn1);
	std::vector<char> inputs05(numIn0);
	std::vector<char> inputs15(numIn1);
	std::vector<char> inputs06(numIn0);
	std::vector<char> inputs16(numIn1);
	std::vector<char> inputs07(numIn0);
	std::vector<char> inputs17(numIn1);

	inputs00 = {1, 0, 0, 0};
	inputs10 = {1, 0, 0, 0};
	inputs01 = {0, 1, 0, 0};
	inputs11 = {0, 1, 0, 0};
	inputs02 = {0, 0, 1, 0};
	inputs12 = {0, 0, 1, 0};
	inputs03 = {0, 0, 0, 1};
	inputs13 = {0, 0, 0, 1};
	inputs04 = {1, 0, 0, 0};
	inputs14 = {1, 0, 0, 0};
	inputs05 = {0, 1, 0, 0};
	inputs15 = {0, 1, 0, 0};
	inputs06 = {1, 0, 0, 0};
	inputs16 = {0, 0, 0, 1};
	inputs07 = {1, 0, 0, 0};
	inputs17 = {1, 0, 0, 0};

	Region region(cs, cp, rng, numIn0, numIn1, numN, numSpD0, numSpD1);
//	region.initSynapsesRandom(cs);
//	region.initSynapsesTest(cs);

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

			printf("TEST");

			switch (counter)
			{
				case 0:
					region.setInputs0(cs, inputs00);
					region.setInputs1(cs, inputs10);
					region.activate(cs, true);
					counter++;
					break;

				case 1:
					region.setInputs0(cs, inputs01);
					region.setInputs1(cs, inputs11);
					region.activate(cs, true);
					counter++;
					break;

				case 2:
					region.setInputs0(cs, inputs02);
					region.setInputs1(cs, inputs12);
					region.activate(cs, true);
					counter++;
					break;

				case 3:
					region.setInputs0(cs, inputs03);
					region.setInputs1(cs, inputs13);
					region.activate(cs, true);
					counter++;
					break;

				case 4:
					region.setInputs0(cs, inputs04);
					region.setInputs1(cs, inputs14);
					region.activate(cs, true);
					counter++;
					break;

				case 5:
					region.setInputs0(cs, inputs05);
					region.setInputs1(cs, inputs15);
					region.activate(cs, true);
					counter++;
					break;

				case 6:
					region.setInputs0(cs, inputs06);
					region.setInputs1(cs, inputs16);
					region.activate(cs, true);
					counter++;
					break;

				case 7:
					region.setInputs0(cs, inputs07);
					region.setInputs1(cs, inputs17);
					region.activate(cs, true);
					counter++;
					break;

				default:
					region.predict(cs);
			}

//			clock.restart();
//			region.activate(cs, true);
//			activateTime = clock.getElapsedTime();

//			region.print(cs);
			std::cout << "Activate(us): " << activateTime.asMicroseconds() << std::endl;
//			std::cout << std::endl << "TM(us): " << tmTime.asMicroseconds();
//			std::cout << std::endl << "Total(us): " << spTime.asMicroseconds() + tmTime.asMicroseconds() << std::endl;

			window.clear(sf::Color::Black);

			pause = true;
		}

		window.display();
	}
}
