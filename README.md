# Simple Cortex

Simple Cortex (SC) is an unsupervised online learning machine intelligence architecture based on intelligence principles of the mammalian neocortex.  The work is heavily inspired by Numenta's Hierarchical Temporal Memory (HTM) Theory, a theoretical framework of neocortex for both biological and machine intelligence.  Simple Cortex use OpenCL for fast parallel computation.

## Demos

- [Ball 1.0](https://www.youtube.com/watch?v=Az5HldJHbKc)
- [Ball 2.0](https://www.youtube.com/watch?v=iRt8sVPZkss)

## Architecture

Simple Cortex offers a straightforward and robust architecture for learning and predicting spatio-temporal sensory-motor patterns and sequences.  The algorithms are inspired by the hierarchy of interacting structures found in the mammalian neocortex.  Fundamentally, these structures are networked sensors with memory that respond to concurrent stimulation.  Advanced network architectures like the neocortex not only observe an environment through "bottom-up" sensing, but also make sense of the environment through "top-down" perception, expectation, and prediction of what's in the environment.  This "top-down" perception allows intelligent systems to have sophisticated awareness of and interaction in an environment.  Simple Cortex may use both "bottom-up" and "top-down" stimulation.  Therefore, the possibile solutions of a suitably sophisticated SC implementation are endless.

As the name implies, the architecture is simple:

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/sc.png)

#### Stimulation

A SC stimulae is simply a data vector that represents a detectable change in an environment.  This can be literally anything: light brightness, sound waves, actuator positions, etc.  The interacting network of sensors in the brain discussed above communicate with each other by using neuron activations stimulae.  Therefore because the data is so versitile, SC can learn any type of pattern or sequence in an environment or within the intelligent architecture itself.  For now SC only processes binary(0 or 1) stimulae, but a future upgrade will allow SC to observe and learn from scalar data vectors.

#### Synapse

A SC synapse is the most fundamental memory storage and learning unit.  Like the neocortex and HTM theory, SC synapses grow towards active neurons and shrink away from inactive neurons.  A "connected" synapse responds to neuronal stimulus while a "unconnected" synapse is unaffected by neuronal stimulus.  Synapses grow towards consistantly reoccouring neurons tend to be very strongly locked to that neuron while unconnected synapses often change their connection.  This is the basis of Hebbian Learning.

Each synapse has:
- Address: represents the presynaptic stimulus a synapse is connected to.
- Permanence: represents how well the synapse is connected to its address.  Uses scalar value from 0 to 99.

Synaptic learning rules include:
- Grow: Increment the synapse permanence if a pre-synaptic connection is active,.
- Shrink: Decrement the synapse permanence if a pre-synaptic connection is inactive.
- Switch: If a synapse permanence is 0 (representing an unconnected synapse), set the synapse address to an unused pre-synaptic connection and reset the synapse permanance to 1.

#### Dendrite

A SC dendrite is a collection of synapses and a threshold that forms a coincidence detector.  When enough of these synapses become active at one time step and are greater than the dendrite's threshold value, the dentrite is active.  This represents the occourance of a pattern, or reoccouring stimulae.  A dendrite may have as many synpases as required.

#### Forest

A SC forest is a collection of dendrites that respond to and learn from a common stimulae.  The forest structure simplifies the code and allows for more efficient parallelization.

#### Neuron

A SC neuron is loosely modeled by pyramidal neurons found in the mammalian neocortex and in HTM theory.  Much like the dendrite model explained above, a neuron is a collection of dendrites and a threshold that forms a coincidence detector.  Since an active dendrite represents the occourance of a pattern, when enough of these patterns are recognized at one time step the neuron is active.  An active neuron represents the occourance of one or more patterns depending on how many dendrites connect to a neuron.  This means that even when a dendrite was not activated by its observed stimulae, the activation of the neuron implies that the stimulae was expected or predicted.  A neuron may have as many dendrites as required.

#### Area

A SC area is simply a collection of neurons.  In the neocortex this is equivalent to "cortical columns", or "macrocolumns".  An area may have as many neurons as required.

#### Region

A SC region is a collection of areas.  In the neocortex this is equivalent to brain regions like V1 or M1, etc.

## Future Improvements
- Upgrade algorithms to allow for observing and learning from scalar data vectors
- Optimize learnSynapses kernel
- Figure out better inhibitFlag variable
