# Simple Cortex

This project is a simple unsupervised on-line learning machine intelligence architecture based on principles of neocortical funtion.  The work is heavily inspired by Numenta's Hierarchical Temporal Memory (HTM), a theoretical framework of the neocortex for both biological and machine intelligence.

## Architecture

- Neurons: A unit who's activation represents the occourance of one or more input patterns.
	- Feed-forward:
	- Lateral:
	- Feed-back:
- Dendrites: A collection of synapses that form a coincidence detector.
- Synapses: The fundamental memory storage unit.  Each synapse has:
	- Address: represents where the synapse is connected
	- Permanence: represents how well the synapse is connected

## Core Functions

### Set Dendrite States

A synapse is actived if its presynaptic connection is active and its permenence is greater than or equal to the synaptic permanence threshold.  A dendrite is active if the number of active synapses is greater than or equal to the dendritic activation threshold.

### Learn Synapses

If learning is enabled and when a dendrite segment is marked for learning its synapses undergo one of four learning rules:

1. Grow: If the pre-synaptic connection is active, increase the synapse permanence by the learning rate.
2. Shrink: If the pre-synaptic connection is inactive, decrease the synapse permanence by the learning rate.
3. Birth: If the synapse is unused (address at max value) and there's an unused active input, insert the synapse by setting the synapse address to the unused input and the synapse permanence to the threshold.
4. Death: If the synapse permanence falls to zero, remove the synapse (set synapse address to max address value).
