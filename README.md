# Simple Cortex (SC)
- **Simple**: Algorithms do not require sophisticated understanding of mathematics
- **Neocortical**: Architecture inspired by intelligence principles of the mammalian neocortex
- **Unsupervised**: Algorithms acquire knowledge by observation (no labeled data necessary)
- **On-line**: Algorithms observe and learn data continuously
- **General**: Algorithms can process any form of stimulation (spatio-temporal sensory-motor)
- **Dynamic Memory**: Stored knowledge adapts based on new knowledge
- **Predictive**: Algorithms are able to predict multiple future states based on learned experience
- **Fast**: Algorithms are parallelized in OpenCL for GPU processing (10 billion synapses/sec)

## Demos
- [Ball 1.0](https://www.youtube.com/watch?v=Az5HldJHbKc)
- [Ball 2.0](https://www.youtube.com/watch?v=iRt8sVPZkss)

## Theory
- Neurons are sensors with memory that respond concurrent stimulae
- Concurrent stimulae are multiple detectable changes in an environment within a time range
- Concurrent stimulae may represent patterns or sequences

## Architecture

- **Stimulae**: Detectable changes in an environment represented by a binary data vector
- **Synapse**: Fundamental memory storage structure
- **Dendrite**: A collection of Synapses forming a coincidence detector
- **Forest**: A collection of Dendrites used to organize OpenCL buffers
- **Neuron**: A collection of Dendrites forming a sensor that responds to stimulae
- **Area**: A collection of Neurons with activation and inhibition rules

## Algorithms
- **Encode**: Activates Neurons based on Stimulae
  - Overlap Synapses with Stimulae
  - Activate Neurons (and Inhibit if applicable)
  - If no Inhibition, select highest Boost Neurons

- **Learn**: Synapses store knowledge of Stimulae
  - For all Active Neurons: Grow, Shrink, or Move Synapses to Active Stimulae

- **Predict**: Predict Neurons based on Stimulae
  - Overlap Synapses with Stimulae
  - Predict Neurons
  
- **Decode**: Retrieve Stimulae based on Neuron Activations
  - Retrieve Synapse addresses of Active Neurons

## Future Improvements
- Upgrade algorithms to allow for observing and learning from scalar data vectors
- Optimize learnSynapses kernel

## Inspiration:
- **Numenta**: Hierarchical Temporal Memory(HTM)
- **Ogma**: Feynman Machine
- **Rebel Science**: Rebel Speech
