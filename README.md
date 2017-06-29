# HTM OpenCL

This project is an OpenCL implementation of Numenta's Hierarchical Temporal Memory (HTM), a theoretical framework of the neocortex for both biological and machine intelligence.

## Architecture

- Columns: A collection of neurons that respond to the same receptive field of the pre-synaptic input.
- Neurons: Represent the occourance of an input pattern in the context of another pattern or patterns (i.e. previous input patterns gives temporal context)
- Dendrites: A collection of synapses that form a coincidence detector.
	- Proximal: Bottom-up input in current time step lead to column activations (represents spatial pattern)
	- Distal: Lateral neuron activations in current time step lead to neuron activations in next time step (represents temporal pattern)
- Synapses: The fundamental memory storage of HTM theory.  Each synapse has an address, representing where the synapse is connected, and a permanence value, representing how well the synapse is connected.

## Core HTM Functions

### Spatial Pooling

When HTM is given an input pattern, Spatial Pooling uses proximal dendrites to select active columns.  It does so in three steps:

1. Set proximal dendrite states based on input
2. Inhibit proximal dendrites to select active columns
3. Learn proximal synapses

### Temporal Memory

Temporal Memory uses distal dendrites to select active neurons based on previously predicted neurons.

1. For each active column
	a. If neuron has previous active distal dendrites, activate neurons
	b. If no previous active distal dendrites, burst column (set activate all neurons in column).
2. Learn distal dendrites
3. Set distal dendrite states based on previous neuron activations

## OpenCL Kernels

### Set Dendrite States

A synapse is actived if its presynaptic connection is active and its permenence is greater than or equal to the synaptic permanence threshold.  A dendrite is active if the number of active synapses is greater than or equal to the dendritic activation threshold.

### Learn Synapses

If learning is enabled and when a dendrite segment is marked for learning its synapses undergo one of four learning rules:

1. Grow: If the pre-synaptic connection is active, increase the synapse permanence by the learning rate.
2. Shrink: If the pre-synaptic connection is inactive, decrease the synapse permanence by the learning rate.
3. Birth: If the synapse is unused (address at max value) and there's an unused active input, insert the synapse by setting the synapse address to the unused input and the synapse permanence to the threshold.
4. Death: If the synapse permanence falls to zero, remove the synapse (set synapse address to max address value).