# Simple Cortex (SC)

Simple Cortex (SC) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex.  Additionally, it provides a brief theory on the interaction and behavior of cortical structures (synapses, dendrites, and neurons).  Currently SC solves unsupervised and supervised on-line machine learning problems.  Simple Cortex is:

- **Simple**: No sophisticated knowledge of mathematics is required to understanding SC theory, architecture, and algorithms
- **Flexible**: SC can observe and learn from any spatio-temporal sensory-motor input.
- **Dynamic**: Stored knowledge/memories adapt based on new knowledge and observations.  If a SC area becomes full it will keep learning but only remember the most observed data.
- **Predictive**: SC is able to predict neuron states many time steps into the future.  It can also translate these neuron states back to observable data.
- **Fast**: SC is coded using OpenCL for fast GPU parallel processing.  Calculation speeds of 10 billion synapses/sec with a capacity of up to 1.5 million neurons on a GTX 1070 GPU.

## Demos
[YouTube - Ball Demo](https://www.youtube.com/watch?v=iRt8sVPZkss)

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/ball-demo.gif)

## Theory
- Concurrent stimulae are detectable changes in an environment that occour within a time range.  To get an intuition for concurrent stimulae, think about the computer screen.  All pixels must activate in a very narrow window in time (concurrently) to display recognizeable objects.
- Neurons are sensors that respond to concurrent stimulae and have memory in the form of synapses that grow and connect with commonly reoccouring stimulae.  Neuron activations are the language of the neocortex and represent spatio-temporal sensory-motor patterns and/or sequences of stimulae.

## Architecture

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/sc.png)

- **Stimulae**: Detectable changes in an environment represented by a binary data vector.  Examples include light brightness, color, sound frequency, muscle contraction, and neuron activations.
- **Synapse**: Connects towards, responds to, and stores knowledge from observed stimulae
- **Dendrite**: A collection of synapses forming a coincidence detector
- **Forest**: A collection of dendrites used to organize OpenCL data buffers that respond to a particular stimulae
- **Neuron**: A collection of dendrites forming a sensor that responds to dendrite activation
- **Area**: A collection of neurons with activation and inhibition rules

## Algorithms
- **Encode**: Converts observed stimulae to neuron activations
  - Overlap stimulae with synapses
  - Activate and inhibit neurons if overlaps beat neuron activation thresholds
  - If no inhibition, activate neurons with highest boost value

- **Learn**: Store knowledge of observed stimulae
  - For all active neurons: grow, shrink, or move synapses to active stimulae

- **Predict**: Predict neurons based on stimulae
  - Overlap stimulae with synapses
  - Activate neurons if overlaps beat neuron prediction threshold
  
- **Decode**: Converts neuron activations to stimulae using synapse memories
  - Activate stimulae based on the synapse addresses of active neurons

## Future Improvements
- Benchmark performance vs. NUPIC, Ogmaneo, and LSTM
- Upgrade algorithms to allow for observing and learning from scalar stimulae data vectors, which could represent the number of action potentials outputted by a presynaptic-neurons within a time step
- Upgrade algorithms to allow neurons to share similar dendrites.  Resource sharing could allow for a smaller memory footprint
- Optimize "learnSynapses" kernel
- Implement some sort of "saveSynapses" and "loadSynapses" for data storage and retrieval
- Create reinforcement learning demos by adding a simple neurotransmitter model

## Inspiration
- **Numenta**
  - [Hierarchical Temporal Memory (HTM)](https://numenta.com/papers/)
  - [Biological and Machine Intelligence (BAMI)](https://numenta.com/biological-and-machine-intelligence/)
  - [NuPIC](https://github.com/numenta/nupic)
  - [HTM Forum](https://discourse.numenta.org/)
    
- **Ogma**
  - [Feynman Machine](https://arxiv.org/abs/1609.03971)
  - [OgmaNeo](https://github.com/ogmacorp/OgmaNeo)
  
- **Rebel Science**
  - [Rebel Cortex](http://www.rebelscience.org/download/rebelcortex.pdf)
  - [Rebel Speech](http://www.rebelscience.org/download/rebelspeech.pdf)
