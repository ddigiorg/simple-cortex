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

// v V Value
// p P Pattern
// s S Synapse
// d D Dendrite
// n N Neuron

class Pattern
{
	public:
		cl_uint numV; // number of values

		cl::Buffer values; // OpenCL buffer of chars (values from 0 to 1)
};

class Dendrite
{
	public:
		cl_uint numSperD; // number of synapses per dendrite
		cl_uint numS;     // number of synapses
		cl_uint dThresh;  // dendrite activation threshold

		cl::Buffer sAddrs; // OpenCL buffer of ushorts (values from 0 to 65535)
		cl::Buffer sPerms; // OpenCL buffer of chars (values from 0 to 99)
};

class Region
{
public:
	Region(
		ComputeSystem &cs,
		ComputeProgram &cp,
		std::mt19937 rng,
		unsigned int numN,
		std::vector<unsigned int> numVperP, // number of values per pattern
		std::vector<unsigned int> numSperD  // number of synapses per dendrite
	);

	void encode(ComputeSystem& cs, std::vector <unsigned int> pEncode);
	void learn(ComputeSystem& cs, std::vector<unsigned int> pLearn);
	void predict(ComputeSystem& cs);
	void decode(ComputeSystem& cs);

	void print(ComputeSystem& cs);

	void setPattern(ComputeSystem& cs, unsigned int p, std::vector<char> vec);
//	void copyInputsToInputs(ComputeSystem& cs, unsigned int dFrom, unsigned int dTo);
	void copyActiveNeuronsToPattern(ComputeSystem& cs, unsigned int p);
	void copyWinnerNeuronsToPattern(ComputeSystem& cs, unsigned int p);

	std::vector<char> getPattern(ComputeSystem &cs, unsigned int d);

private:
	std::mt19937 _rng;

	cl::NDRange _range;

	cl_uint _numN;       // number of neurons per area
	cl_uint _numW;       // number of winner neurons during one time step
	cl_uint _numP;       // number of patterns per area
	cl_uint _numDperN;   // number of dendrites per neuron
	cl_uint _nActThresh; // neuron activation threshold
	cl_uint _nPreThresh; // neuron prediction threshold
	cl_uint _sPermMax;   // synapse permanence max value (99)
	cl_uint _sAddrMax;   // synapse address max value (65535)

	cl::Buffer _inhibitFlag;

	cl::Buffer _nBoosts;   // OpenCL buffer of ushorts (values from 0 to 65535)
	cl::Buffer _nPredicts; // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nWinners;  // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nActives;  // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nOverlaps; // OpenCL buffer of chars (values from 0 to 255)

	cl::Buffer _DOVE0;
	cl::Buffer _DOVE1;

	std::vector<Pattern> _patterns;
	std::vector<Dendrite> _dendrites;

	cl::Kernel _overlapDendrites;
	cl::Kernel _learnSynapses;
	cl::Kernel _activateNeurons;
	cl::Kernel _predictNeurons;
	cl::Kernel _decodeNeurons;
};

#endif
