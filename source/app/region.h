// ========
// region.h
// ========

#ifndef REGION_H
#define REGION_H

#include "utils/utils.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include <vector>
#include <random>
#include <iostream>

class Synapses
{
	public:
		cl_uint numIn;     // number of inputs
		cl_uint numSpD;    // number of synapses per dendrite
		cl_uint numS;      // number of synapses
		cl_uint dThresh;   // dendrite activation threshold

		cl::Buffer inputs; // OpenCL buffer of chars (values from 0 to 1)
		cl::Buffer addrs;  // OpenCL buffer of ushorts (values from 0 to 65535)
		cl::Buffer perms;  // OpenCL buffer of chars (values from 0 to 99)
};

class Region
{
public:
	Region(
		ComputeSystem &cs,
		ComputeProgram &cp,
		std::mt19937 rng,
		unsigned int numN,
		std::vector<unsigned int> numIn,
		std::vector<unsigned int> numSpD
	);

	void encode(ComputeSystem& cs, bool learn);
	void predict(ComputeSystem& cs);
	void decode(ComputeSystem& cs);

	void print(ComputeSystem& cs);

	void setInputs(ComputeSystem& cs, unsigned int d, std::vector<char> vec);

	void copyInputsToInputs(ComputeSystem& cs, unsigned int dFrom, unsigned int dTo);
	void copyNeuronsToInputs(ComputeSystem& cs, unsigned int d);

	std::vector<char> getInputs(ComputeSystem &cs, unsigned int d);
	std::vector<char> getOutputs(ComputeSystem &cs);

private:
	std::mt19937 _rng;

	cl::NDRange _range;

	cl_uint _numN;       // number of neurons
	cl_uint _numAN;      // number of active neurons at each time step
	cl_uint _numDpN;     // number of dendrites per neuron
	cl_uint _nActThresh; // neuron activation threshold
	cl_uint _nPreThresh; // neuron prediction threshold
	cl_uint _sPermMax;   // synapse permanence max value (99)
	cl_uint _sAddrMax;   // synapse address max value (65535)

	cl::Buffer _outputs;   // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nPredicts; // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nLearns;   // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nActives;  // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nOverlaps; // OpenCL buffer of chars (values from 0 to 255)
	cl::Buffer _nBoosts;   // OpenCL buffer of ushorts (values from 0 to 65535)

	std::vector<Synapses> _dendrites;

	cl::Kernel _overlapDendrites;
	cl::Kernel _learnSynapses;
	cl::Kernel _predictNeurons;
	cl::Kernel _decodeNeurons;
};

#endif
