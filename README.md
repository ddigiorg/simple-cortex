# Simple Cortex

Simple Cortex (SC) is an unsupervised on-line learning machine intelligence architecture based on intelligence principles of the mammalian neocortex.  The work is heavily inspired by Numenta's Hierarchical Temporal Memory (HTM), a theoretical framework of neocortex for both biological and machine intelligence.  Simple Cortex uses OpenCL for fast parallel computation.

## Functionality

#### Learn Patterns

- sensory patterns
- motor patterns

#### Learn Sequences

#### Long Term Prediction

Beyond one time step into the future.

## Architecture

#### Area

A SC area is a collection of neurons modeled by pyramidal neurons found in HTM Theory and the mammalian neocortex.  However unlike HTM theory, these neurons are not pre-arranged in minicolumns of shared input receptive fields.  Rather the neurons will respond to and learn the receptive fields dynamically, allowing for a much simpler set of algorithms.

#### Neurons

A SC neuron is a unit who's activation represents the occourance of one or more patterns.  Each neuron has 3 states:
- Inactive: Not enough dendrites are active to set the neuron into activate or predict states.
- Active: When enough dendrites are active and pass a threshold value the neuron is active and may learn its input patterns.
- Predicted: When one or a few dendrites are active, the neuron is predicted.  This means that after sufficient learning, the observation of just one pattern implies the occourance of other patterns even though they have not been directly observed. 

#### Dendrites

A SC dendrite is a collection of synapses that form a coincidence detector and represents the occourance of a pattern.  When enough synapses are active and pass a threshold value the dentrite is active.

#### Synapses

A SC synapse is the fundamental memory storage unit.  Like HTM Theory each synapse has:
- Address: represents the presynaptic connection, or what input node or neuron the synapse is connected to.
- Permanence: represents how well the synapse is connected.  Unlike HTM Theory synapses are always connected to the presynaptic connection.

## Core Algorithms

#### Overlap Dendrites

A synapse is actived if its presynaptic connection is active and its permenence is greater than or equal to the synaptic permanence threshold.  A dendrite is active if the number of active synapses is greater than or equal to the dendritic activation threshold.

#### Learn Synapses

If learning is enabled and when a dendrite segment is marked for learning its synapses undergo one of four learning rules:

1. Grow: If the pre-synaptic connection is active, increase the synapse permanence by the learning rate.
2. Shrink: If the pre-synaptic connection is inactive, decrease the synapse permanence by the learning rate.
3. Birth: If the synapse is unused (address at max value) and there's an unused active input, insert the synapse by setting the synapse address to the unused input and the synapse permanence to the threshold.
4. Death: If the synapse permanence falls to zero, remove the synapse (set synapse address to max address value).

#### Activate Neurons

inhibition

#### Predict Neurons

## To Do List
- Optimize learnSynapses kernel
- Figure out better inhibitFlag variable
