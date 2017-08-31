# Simple Cortex (SC)

Simple Cortex (SC) is a Machine Intelligence architecture based on intelligence principles of the mammalian neocortex.  Additionally, it provides a brief theory on the interaction and behavior of cortical structures (synapses, dendrites, and neurons).  Currently SC solves unsupervised and supervised on-line machine learning problems.  Simple Cortex is:

- **Simple**: No sophisticated knowledge of mathematics is required to understanding SC theory, architecture, and algorithms
- **Flexible**: SC can observe and learn from any spatio-temporal sensory-motor input.
- **Dynamic**: Stored knowledge/memories adapt based on new knowledge and observations.  If a SC area becomes full it will keep learning but only remember the most observed data.
- **Predictive**: SC is able to predict neuron states many time steps into the future.  It can also translate these neuron states back to observable data.
- **Fast**: SC is coded using OpenCL for fast GPU parallel processing

## Demos
[YouTube - Ball Demo](https://www.youtube.com/watch?v=iRt8sVPZkss)

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/ball-demo.gif)

## Theory
- Concurrent stimulae are detectable changes in an environment that occour within a time range.  To get an intuition for concurrent stimulae, think about the computer screen.  All pixels must activate in a very narrow window in time (concurrently) to display recognizeable objects.
- Neurons are sensors that respond to concurrent stimulae and have memory in the form of synapses that grow and connect with commonly reoccouring stimulae.  Neuron activations are the language of the neocortex and represent spatio-temporal sensory-motor patterns and/or sequences of stimulae.

## Architecture

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/sc.png)

- **Stimulae**: Detectable changes in an environment represented by a binary data vector.  Examples include light brightness, color, sound frequency, muscle contraction, and neuron activations.
  - sState: Stimulus state represents whether a stimulus is active or inactive
- **Synapse**: Connects towards, responds to, and stores knowledge from observed stimulae
  - sAddrs: Synapse address represents what stimulus a synapse is connected to
  - sPerms: Synapse permanence represents how strongly a synapse is connected to it's stimulus
- **Dendrite**: A collection of synapses forming a coincidence detector
  - dOverlap: Dendrite overlap represents how many synapses are active (connected to active stimulae) during a time step
  - dThresh: Dendrite threshold represents how many active synapses are needed to activate the dendrite
- **Forest**: A collection of dendrites used to organize OpenCL data buffers that respond to a particular stimulae
- **Neuron**: A collection of dendrites forming a sensor that responds to dendrite activation
  - nOverlap: Neuron overlap represents how many dendrites are active during a time step
  - nThresh: Neuron threshold represents how many active dendrites are needed to activate the neuron
  - nBoost: Neuron boost represents how often a neuron is activated.  Neurons activated less frequently are more likely to learn new patterns
  - nState: Neuron state represents whether a neuron is active or inactive
- **Area**: A collection of neurons with activation and inhibition rules

Note: SC may have as many synapses per dendrite, dendrites per neuron, and neurons per area as required.  The number of forests should be equal to the number of dendrites per neuron, although the exact arrangement is up to the user.

## Algorithms
- **Encode**: Converts observed stimulae to neuron activations (pattern recognition)
  - Overlap stimulae with synapses
  - Activate and inhibit neurons if overlaps beat neuron activation thresholds
  - If no inhibition, activate neurons with highest boost value
  - Increment all boost values, then set active neuron boosts to 0

- **Learn**: Active neurons store knowledge of observed stimulae
  - Grow: Increment synapse permanence if connected stimulae is active
  - Shrink: Decrement synapse permanence if connected stimulae is inactive
  - Move: If synapse permanence decreases to 0, change synapse address to an unused active stimulae and set permanence to 1

- **Predict**: Predict neurons based on stimulae
  - Overlap stimulae with synapses
  - Activate neurons if overlaps beat neuron prediction threshold
  
- **Decode**: Converts neuron activations to stimulae using synapse memories
  - Activate stimulae based on the synapse addresses of active neurons

## Benchmarks

#### Compute Speed per Timestep

- Performed Ball demo on Nvidia GTX 1070 GPU
- Mean and Standard Deviation used 100,000 time step samples
- 1.5 million neurons with 2 dendrites each
  - Dendrite 1: 50 synapses
  - Dendrite 2: 1 synapse
  - Total synapses: 76.5 million
- 20 predicts per time step

|            Executed Algorithms             |    Mean (ms)   | Standard Deviation (ms) |
| ------------------------------------------ |:--------------:|:-----------------------:|
| **Encode**                                 |       6.72     |           0.00*         |
| **Encode, Learn**                          |       7.44     |           4.24**        |
| **Encode, Learn, x20 Predict, x20 Decode** |      21.76     |           4.47**        |

- *Low std. dev. of Encode was less than 1/100th of a millisecond and therefore written as 0.00 ms
- **High std. dev. of Learn due to unoptimized "move synapse" algorithm

## Results

Simple Cortex can encode and learn **~10 billion synapses/second** according to compute speed benchmark (76.5 million synapses in 7.44 ms).

## Future Improvements
- Benchmark performance vs. NUPIC, Ogmaneo, and LSTM
- Figure out why "encode" function segfaults over 1.5 million neurons
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
