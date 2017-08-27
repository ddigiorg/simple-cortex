# Simple Cortex (SC)

Simple Cortex (SC) is an unsupervised on-line learning machine intelligence architecture based on intelligence principles of the mammalian neocortex and a simple theory of neurons.  SC is:

- **General**: Algorithms can observe and learn any spatio-temporal sensory-motor input
- **Dynamic**: Stored knowledge/memories adapt based on new knowledge and observations
- **Predictive**: Algorithms are able to predict multiple future states based on learned experience
- **Fast**: Algorithms are parallelized in OpenCL for GPU processing (10 billion synapses/sec on GTX 1070 GPU)

## Demos
[YouTube - Ball Demo](https://www.youtube.com/watch?v=iRt8sVPZkss)

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/ball-demo.gif)

## Theory
- Neurons are sensors respond to concurrent stimulae and have memory.
- Concurrent stimulae are detectable changes in an environment within a time range.
- Concurrent neuron activations (a form of stimulae) represent spatio-temporal sensory-motor patterns or sequences.

## Architecture

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/sc.png)

- **Stimulae**: Detectable changes in an environment represented by a binary data vector.  Example stimulae include light brightness, color, sound frequency, muscle contraction, and neuron activations
- **Synapse**: Connects towards, responds to and stores knowledge from observed stimulae
- **Dendrite**: A collection of synapses forming a coincidence detector
- **Forest**: A collection of dendrites used to organize OpenCL data buffers that respond to a particular stimulae
- **Neuron**: A collection of dendrites forming a sensor that responds to dendrite activation
- **Area**: A collection of neurons with activation and inhibition rules

## Algorithms
- **Encode**: Converts observed stimulae to neuron activations
  - Overlap stimulae with synapses
  - Activate neurons and inhibit if applicable
  - If no inhibition, select neuron with highest boost value

- **Learn**: Store knowledge of observed stimulae
  - For all active neurons: grow, shrink, or move synapses to currently active stimulae

- **Predict**: Predict neurons based on Stimulae
  - Overlap stimulae with synapses
  - Predict neurons
  
- **Decode**: Converts neuron activations to stimulae using synapse memories
  - Retrieve synapse addresses of active neurons

## Future Improvements
- Upgrade compute-system and compute-program to use cl2.hpp
- Benchmark performance vs. NUPIC and Ogmaneo
- Upgrade algorithms to allow for observing and learning from scalar data vectors
- Optimize learnSynapses kernel

## Inspiration:
- **Numenta**: Hierarchical Temporal Memory (HTM)
- **Ogma**: Feynman Machine
- **Rebel Science**: Rebel Cortex
