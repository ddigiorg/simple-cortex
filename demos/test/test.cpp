// ========
// test.cpp
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
	unsigned int numNeurons = 35; // note: cant go over 65535 neurons unless _sAddrMax is modified

	std::vector<unsigned int> numVperP(4); // 4 patterns
	std::vector<unsigned int> numSperD(2); // 2 dendrites (per neuron)

	// Patterns
	numVperP = {numPixels,  // input - current scene state
                numNeurons, // input - previous active neurons
                numPixels}; // output - predicted future scene state

	// Dendrites
	numSperD = {1,  // stores current active scene state values
                1}; // stores previous winner neurons

	Region region(cs, cp, numNeurons, numVperP, numSperD);

	std::vector<char> resetPreActNeuVec(numNeurons);
	resetPreActNeuVec[numNeurons - 1] = 1;

	// for printing neuron information
	std::vector<unsigned short> nBoostsVec(numNeurons);
	std::vector<char> nPredictsVec(numNeurons);
	std::vector<char> nActivesVec(numNeurons);
	std::vector<unsigned short> sAddrs0Vec(numNeurons);
	std::vector<char> sPerms0Vec(numNeurons);
	std::vector<char> pattern0Vec(numPixels);
	std::vector<unsigned short> sAddrs1Vec(numNeurons);
	std::vector<char> sPerms1Vec(numNeurons);
	std::vector<char> pattern1Vec(numNeurons);

	// Loop
	bool quit = false;
	bool pause = false;
	int counter = 0;

	sf::Clock clock;
	sf::Time time;

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

			clock.restart();
			time = clock.getElapsedTime();

			region.encode(cs, {0, 1}, {0, 1});
			region.learn(cs, {0, 1}, {0, 1});
			region.predict(cs, {1}, {1});
			region.decode(cs, {2}, {0});

			time = clock.getElapsedTime();

			// print neuron information
			printf("\nNEURO ");
			for (int i = 0; i < numNeurons; i++)
			{
				if (i < 10)
					printf("  %i ", i);
				else
					printf(" %i ", i);
			}

			printf("\nNBOOS ");
			nBoostsVec = region.getNeuronBoosts(cs);
			for(int i = 0; i < nBoostsVec.size(); i++)
			{
				if (nBoostsVec[i] < 10)
					printf("  %i ", nBoostsVec[i]);
				else if (nBoostsVec[i] < 100)
					printf(" %i ", nBoostsVec[i]);
				else if (nBoostsVec[i] > 999)
					printf("XXX ");
				else
					printf("%i ", nBoostsVec[i]);
			}

			printf("\nNPRED ");
			nPredictsVec = region.getNeuronPredicts(cs);
			for(int i = 0; i < nPredictsVec.size(); i++)
			{
				printf("  %i ", nPredictsVec[i]);
			}

			printf("\nNACTI ");
			nActivesVec = region.getNeuronActives(cs);
			for(int i = 0; i < nActivesVec.size(); i++)
			{
				printf("  %i ", nActivesVec[i]);
			}

			printf("\nSADR0 ");
			sAddrs0Vec = region.getSynapseAddrs(cs, 0);
			for(int i = 0; i < sAddrs0Vec.size(); i++)
			{
				if (sAddrs0Vec[i] < 10)
					printf("  %i ", sAddrs0Vec[i]);
				else if (sAddrs0Vec[i] < 100)
					printf(" %i ", sAddrs0Vec[i]);
				else if (sAddrs0Vec[i] > 999)
					printf("XXX ");
				else
					printf("%i ", sAddrs0Vec[i]);
			}

			printf("\nSPER0 ");
			sPerms0Vec = region.getSynapsePerms(cs, 0);
			for(int i = 0; i < sPerms0Vec.size(); i++)
			{
				if (sPerms0Vec[i] < 10)
					printf("  %i ", sPerms0Vec[i]);
				else
					printf(" %i ", sPerms0Vec[i]);
			}

			printf("\nINPU0 ");
			pattern0Vec = region.getPattern(cs, 0);
			for(int i = 0; i < pattern0Vec.size(); i++)
			{
				if (pattern0Vec[i] > 0)
					printf("%i ", i);
			}

			printf("\nSADR1 ");
			sAddrs1Vec = region.getSynapseAddrs(cs, 1);
			for(int i = 0; i < sAddrs1Vec.size(); i++)
			{
				if (sAddrs1Vec[i] < 10)
					printf("  %i ", sAddrs1Vec[i]);
				else if (sAddrs1Vec[i] < 100)
					printf(" %i ", sAddrs1Vec[i]);
				else if (sAddrs1Vec[i] > 999)
					printf("XXX ");
				else
					printf("%i ", sAddrs1Vec[i]);
			}

			printf("\nSPER1 ");
			sPerms1Vec = region.getSynapsePerms(cs, 1);
			for(int i = 0; i < sPerms1Vec.size(); i++)
			{
				if (sPerms1Vec[i] < 10)
					printf("  %i ", sPerms1Vec[i]);
				else
					printf(" %i ", sPerms1Vec[i]);
			}

			printf("\nINPU1 ");
			pattern1Vec = region.getPattern(cs, 1);
			for(int i = 0; i < pattern1Vec.size(); i++)
			{
				printf("  %i ", pattern1Vec[i]);
			}

			printf("\nTime(us): %i", time.asMicroseconds()); 
			printf("\n");

			window.clear(sf::Color::Black);

			scene.setPixelsFromBinaryVector('g', false, region.getPattern(cs, 0));
			scene.setPixelsFromBinaryVector('b', false, region.getPattern(cs, 2));

			window.draw(scene.getSprite());

			pause = true;
		}

		window.display();
	}
}
