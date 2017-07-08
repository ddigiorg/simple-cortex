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
		cl_uint numIn;
		cl_uint numSpD;
		cl_uint numS;
		cl_uint dThresh;

		cl::Buffer inputs;
		cl::Buffer addresses;
		cl::Buffer permenances;
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

	void activate(ComputeSystem& cs, bool learn);
	void predict(ComputeSystem& cs);

	void print(ComputeSystem& cs);

	void setInputs0(ComputeSystem& cs, std::vector<char> vec);
	void setInputs1(ComputeSystem& cs, std::vector<char> vec);

	std::vector<char> getInputs0(ComputeSystem &cs);
	std::vector<char> getInputs1(ComputeSystem &cs);
	std::vector<char> getOutputs(ComputeSystem &cs);

private:
	std::mt19937 _rng;

	cl::NDRange _range;

	unsigned int _numDpN;

	cl_uint _numIn0;   // number of inputs
	cl_uint _numIn1;   // number of inputs
	cl_uint _numN;     // number of neurons
	cl_uint _numAN;    // number of active neurons at each time step
	cl_uint _numSpD0;  // number of synapses per dendrite
	cl_uint _numSpD1;  // number of synapses per dendrite
	cl_uint _numS0;    // number of synapses total
	cl_uint _numS1;    // number of synapses total
	cl_uint _sPermMax; // synapse permanence max value
	cl_uint _nThresh;  // neuron activation threshold
	cl_uint _dThresh0; // dendrite activation threshold
	cl_uint _dThresh1; // dendrite activation threshold

	cl::Buffer _inputs0;   // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _inputs1;   // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _outputs;   // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nActives;  // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nPredicts; // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nOverlaps; // OpenCL buffer of chars (values from 0 to 255)
	cl::Buffer _nBoosts;   // OpenCL buffer of ushorts (values from 0 to 65535)
	cl::Buffer _sAddrs0;   // OpenCL buffer of ushorts (values from 0 to 65535)
	cl::Buffer _sAddrs1;   // OpenCL buffer of ushorts (values from 0 to 65535)
	cl::Buffer _sPerms0;   // OpenCL buffer of chars (values from 0 to 99)
	cl::Buffer _sPerms1;   // OpenCL buffer of chars (values from 0 to 99)

	std::vector<Synapses> _synapses;

	cl::Kernel _overlapDendrites;
	cl::Kernel _learnSynapses;
	cl::Kernel _predictNeurons;
};

#endif
