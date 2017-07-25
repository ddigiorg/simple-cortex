# Simple Cortex

Simple Cortex (SC) is an unsupervised on-line learning machine intelligence architecture based on intelligence principles of the mammalian neocortex.  The work is heavily inspired by Numenta's Hierarchical Temporal Memory (HTM), a theoretical framework of neocortex for both biological and machine intelligence.  Simple Cortex uses OpenCL for fast parallel computation.

## Functionality

Simple Cortex can learn and predict any type of binary input pattern.

Input patterns may contain sensory or motor information as well as temporal information

Prediction Beyond one time step into the future.

Intention of upgrading the capabilities to learning floating-point inputs.

## Architecture

Simple Cortex architecture is inspired by the networked hierarchy of interacting structures found in the mammalian neocortex.

#### Synapse

A SC synapse is the most fundamental memory storage unit.  In the neocortex synapses grow towards and connect to 

Each synapse has:
- Address: represents the presynaptic connection, or what input node or neuron the synapse is connected to.
- Permanence: represents how well the synapse is connected.  Unlike HTM Theory synapses are always connected to the presynaptic connection.

#### Dendrite

A SC "dendrite" is a collection of synapses formimg a coincidence detector.  When enough of these synapses become active at one time step, the dentrite is active and represents the occourance of a pattern.

#### (Dendritic) Tree

A SC "tree" is a collection of dendrites that observe and learn from a common pattern.

#### Neuron

A SC "neuron" is a collection of dendrites whos activation implies the occourance of 

modeled by pyramidal neurons found in the mammalian neocortex and modeled in HTM theory.

#### Area

A SC "area" is a collection of neurons 

#### Region

A SC "region" is a collection of areas.  

The equivalent of V1 or M1, etc

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
- make ball demo video
- write paper
