# Simple Cortex (SC)

Simple Cortex is a Machine Intelligence neural model based on intelligence principles of the mammalian neocortex.  The details the model's architecture and algorithms may be found in this [paper](https://arxiv.org/abs/1710.01347 " Simple Cortex: A Model of Cells in the Sensory Nervous System").  Specifically, the paper discusses the fundamentals of sensation and perception and how those conecpts relate to the architecture and behavior of cortical structures (synapses, dendrites, and neurons).  SC uses these basic intelligence principles in its an implementation, found in this repository, which observes, learns, and predicts spatio-temporal sensory-motor patterns and sequences.  Simple Cortex is:

- **Simple**: No sophisticated knowledge of mathematics required
- **Unsupervised**: Stores knowledge through observation with no labeled data necessary
- **On-line**: Observes and learns continuously like biological intelligence
- **Dynamic**: memories adapt to new observations, prioritizing commonly recurring knowledge
- **Predictive**: SC is able to predict neuron states many time steps into the future. It can also translate these neuron states back to observable data
- **Fast**: GPU parallel processing encodes and learns billions of synapses per second

## Demos
[YouTube - Ball Demo](https://www.youtube.com/watch?v=iRt8sVPZkss)

![alt tag](https://raw.githubusercontent.com/ddigiorg/neuroowl.github.io/master/webpages/technology/simple-cortex/ball-demo.gif)

## Future Improvements
- Benchmark performance vs. NUPIC, Ogmaneo, and LSTM
- Optimize the "move synapses" portion of the "learnSynapses" kernel
- Optimize aquiring the neuron index with mamximum boost value for selecting neurons that see new input patterns.  Do this via a two-stage reduction.
- Upgrade algorithms to allow for observing and learning from scalar stimuli data vectors.
- Upgrade algorithms to allow neurons to share similar dendrites.  Resource sharing could allow for a smaller memory footprint.
- Implement some sort of "saveSynapses" and "loadSynapses" for data storage and retrieval.
