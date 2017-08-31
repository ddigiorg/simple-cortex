// ========
// ball.cpp
// ========

#include "ball.h"
#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "utils/render2d.h"
#include "cortex/stimulae.h"
#include "cortex/forest.h"
#include "cortex/area.h"

#include <vector>

int main()
{
	// Setup SFML Render Window
	unsigned int sizeSceneX = 100; // pixels
	unsigned int sizeSceneY = 100; // pixels
	unsigned int scaleScene = 4;
	unsigned int sizeDisplayX = sizeSceneX * scaleScene; // pixels
	unsigned int sizeDisplayY = sizeSceneY * scaleScene; // pixels
	unsigned int numPixels = sizeSceneX * sizeSceneY;

	sf::RenderWindow window;
	window.create(sf::VideoMode(sizeDisplayX, sizeDisplayY), "Simple Cortex - Ball Demo", sf::Style::Default);

	// Setup Scene Renderer
	std::vector<float> rVec(numPixels);
	std::vector<float> gVec(numPixels);
	std::vector<float> bVec(numPixels);

	Render2D scene(sizeSceneX, sizeSceneY);
	scene.setPosition(sizeDisplayX / 2, sizeDisplayY / 2);
	scene.setScale((float) scaleScene);

	// Setup Ball Simulation
	unsigned int ballRadius = 4;
	unsigned int posType = 0;
		// 0 - ball position resets to middle of screen
		// 1 - ball position resets to random int
	unsigned int velType = 2;
		// 0 - ball velocity resets to zero
		// 1 - ball velocity resets to random int
		// 2 - ball velocity resets to random float

	Ball ball(sizeSceneX, sizeSceneY, ballRadius, posType, velType);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;

	cs.init(ComputeSystem::_gpu);
	cs.printCLInfo();

	std::string kernels_cl = "source/cortex/behavior.cl";

	cp.loadFromFile(cs, kernels_cl);

	// Setup Simple Cortex Area
	unsigned int numStimulae = 4;
	unsigned int numForests = 2;
	unsigned int numNeurons = 1500000; // 1,500,000 reccomended maxumum

	std::vector<Stimulae> vecStimulae(numStimulae);
	vecStimulae[0].init(cs, numPixels);  // input - current binary scene state
	vecStimulae[1].init(cs, numNeurons); // input - previous neuron activations
	vecStimulae[2].init(cs, numNeurons); // input - storage of current neuron activations for forecasting
	vecStimulae[3].init(cs, numPixels);  // output - predicted future binary scene state

	std::vector<Forest> vecForest(numForests);
	vecForest[0].init(cs, cp, numNeurons, 50, 0.25f);
	vecForest[1].init(cs, cp, numNeurons,  1, 1.00f);

	Area area;
	area.init(cs, cp, numNeurons);

	std::vector<unsigned char> vecResetNeurons(numNeurons);
	vecResetNeurons[numNeurons - 1] = 1;

	// Render Loop
	bool forecast = false;
	bool learn = true;
	bool stepMode = true;
	bool step = false;
	bool quit = false;

	printf("\nLearning %i Neurons with %i Dendrites each", numNeurons, numForests);
	printf("\nPress 'f' to enable/disable forecasting (default disabled)");
	printf("\nPress 'l' to enable/disable learning    (default enabled )");
	printf("\nPress 'p' to enable/disable step mode   (default enabled )");
	printf("\nPress 'Space' to step algorithms");
	printf("\nPress 'Esc' to quit application");
	printf("\n");

	/*
	sf::Clock clock;
	sf::Time time;

	float msum = 0.0f;
	float mean = 0.0f;
	float vsum = 0.0f;
	float diff = 0.0f;
	float vari = 0.0f;

	unsigned int numTimes = 100000;
	unsigned int count = 0;

	std::vector<float> times;
	times.resize(numTimes);
	*/

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

					case sf::Keyboard::F:
						forecast = !forecast;
						break;

					case sf::Keyboard::L:
						learn = !learn;
						break;

					case sf::Keyboard::P:
						stepMode = !stepMode;
						break;
				}
			}
		}

		if (!stepMode || step)
		{
			for (unsigned int p = 0; p < numPixels; p++)
			{
				rVec[p] = 0.0f;
				gVec[p] = 0.0f;
				bVec[p] = 0.0f;
			}

			ball.step();

			vecStimulae[0].setStates(cs, ball.getBinaryVector());

			if (ball.getStartSequence())
				vecStimulae[1].setStates(cs, vecResetNeurons);

//			clock.restart();

			area.encode(cs, {vecStimulae[0], vecStimulae[1]}, {vecForest[0], vecForest[1]});
				// Getting segfaults beyond 1,500,000 neurons (Note: using cl_uint for neuron addressing)
				// Forest 0: 1.5 mil neurons x 1 dendrite/neuron x 50 synapse/dendrite x 32 bits/synapse = 300 MB per uint address buffer <-- could be this?
				// Forest 1: 1.5 mil neurons x 1 dendrite/neuron x  1 synapse/dendrite x 32 bits/synapse =   6 MB per uint address buffer

			if (learn)
				area.learn(cs, {vecStimulae[0], vecStimulae[1]}, {vecForest[0], vecForest[1]});

			vecStimulae[1].setStates(cs, area.getStates(cs));

			// Forecast 20 time steps into the future
			if (forecast)
			{
				for (unsigned int i = 0; i < 20; i++)
				{
					vecStimulae[2].setStates(cs, area.getStates(cs));

					area.predict(cs, {vecStimulae[2]}, {vecForest[1]});
					area.decode(cs, {vecStimulae[3]}, {vecForest[0]});

					std::vector<unsigned char> prediction = vecStimulae[3].getStates(cs);

					for (unsigned int p = 0; p < numPixels; p++)
					{
						if (prediction[p] > 0)
							bVec[p] = 0.2f + 0.04f * i;
					}
				}
			}

			std::vector<unsigned char> input = vecStimulae[0].getStates(cs);

			/*
			time = clock.getElapsedTime();

			if (count < numTimes)
			{
				times[count] = time.asSeconds();
				count++;
			}
			else if (count == numTimes) 
			{
				for (int i = 0; i < numTimes; i++)
					msum += times[i];

				mean = msum / numTimes;

				for (int i = 0; i < numTimes; i++)
				{
					diff = times[i] - mean;
					vsum += diff * diff;
				}

				vari = vsum / numTimes;

				printf("\nmean: %f", mean);
				printf("\nvari: %f", vari);
				printf("\n");

				count++;
			}
			*/

			for (unsigned int p = 0; p < numPixels; p++)
			{
				if (input[p] > 0)
					gVec[p] = 1.0f;
			}

			scene.setPixels(rVec, gVec, bVec);

			window.clear(sf::Color::Black);
			window.draw(scene.getSprite());
			window.display();

			step = false;
		}
	}
}
