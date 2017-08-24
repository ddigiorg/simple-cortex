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
	// Setup SFML render window and ball simulation
	unsigned int sizeSceneX = 100; // pixels
	unsigned int sizeSceneY = 100; // pixels
	unsigned int scaleScene = 4;
	unsigned int ballRadius = 4;
	unsigned int sizeDisplayX = sizeSceneX * scaleScene; // pixels
	unsigned int sizeDisplayY = sizeSceneY * scaleScene; // pixels
	unsigned int numPixels = sizeSceneX * sizeSceneY;

	sf::RenderWindow window;
	window.create(sf::VideoMode(sizeDisplayX, sizeDisplayY), "Simple Cortex - Ball Demo 2.0", sf::Style::Default);

	Ball ball(sizeSceneX, sizeSceneY, ballRadius);

	Render2D scene(sizeSceneX, sizeSceneY);
	scene.setPosition(sizeDisplayX / 2, sizeDisplayY / 2);
	scene.setScale((float) scaleScene);

	std::vector<float> rVec(numPixels);
	std::vector<float> gVec(numPixels);
	std::vector<float> bVec(numPixels);

	// Setup OpenCL
	ComputeSystem cs;
	ComputeProgram cp;

	std::string kernels_cl = "source/cortex/behavior.cl";

	cs.init(ComputeSystem::_gpu);
	cs.printCLInfo();
	cp.loadFromSourceFile(cs, kernels_cl);

	// Setup Simple Cortex Area
	unsigned int numStimulae = 4;
	unsigned int numForests = 2;
	unsigned int numNeurons = 1500000; 

	// NEED TO FIGURE THIS OUT!!!
	// Getting segfaults at random beyond 1,600,000 neurons...
	// Using unsigned int for neuron addressing
	// 1.5 mil neurons x 32 bits = 6 MB per uint buffer
	// well under GTX 1070 max buffer size...
	// Otherwise smooth operating

	std::vector<Stimulae> vecStimulae(numStimulae);
	vecStimulae[0].init(cs, numPixels);  // input - current binary scene state
	vecStimulae[1].init(cs, numNeurons); // input - previous neuron activations
	vecStimulae[2].init(cs, numNeurons); // input - storage of current neuron activations for forecasting
	vecStimulae[3].init(cs, numPixels);  // output - predicted future binary scene state

	std::vector<Forest> vecForest(numForests);
	vecForest[0].init(cs, cp, numNeurons, 50, 0.75f);
	vecForest[1].init(cs, cp, numNeurons,  1, 1.00f);

	Area area;
	area.init(cs, cp, numNeurons);

	std::vector<unsigned char> vecResetNeurons(numNeurons);
	vecResetNeurons[numNeurons - 1] = 1;

	// Render loop
	bool forecast = false;
	bool stepMode = true;
	bool step = true;
	bool quit = false;

	printf("\nLearning %i Neurons each with %i Dendrites", numNeurons, numForests);
	printf("\nPress 'f' to enable/disable forecasting");
	printf("\nPress 'p' to enable/disable step mode");
	printf("\nPress 'Space' to step algorithms");
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
						stepMode = !stepMode;
						break;

					case sf::Keyboard::F:
						forecast = !forecast;
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

			area.encode(cs, {vecStimulae[0], vecStimulae[1]}, {vecForest[0], vecForest[1]});
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

			for (unsigned int p = 0; p < numPixels; p++)
			{
				if (input[p] > 0)
					gVec[p] = 1.0f;
			}

			scene.setPixels(rVec, gVec, bVec);

			window.clear(sf::Color::Black);
			window.draw(scene.getSprite());

			step = false;
		}

		window.display();
	}
}
