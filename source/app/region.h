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

class Region
{
public:
	Region(
		ComputeSystem &cs,
		ComputeProgram &cp,
		std::mt19937 rng,
		unsigned int numI,
		unsigned int numC,
		unsigned int numNpC,
		unsigned int numDDpN,
		unsigned int numDSpDD
	);

	void initSynapsesTest(ComputeSystem& cs);
	void initSynapsesRandom(ComputeSystem& cs);

	void sp(ComputeSystem& cs, bool learn);
	void tm(ComputeSystem& cs, bool learn);
	void print(ComputeSystem& cs);

	void setInputs(ComputeSystem& cs, std::vector<char> inputsVec);

	std::vector<char> getInputs(ComputeSystem &cs);

private:
	std::mt19937 _rng;

//	cl_float2 _initialMemoryRange = {0.0f, 0.0001f};

	cl::NDRange _range;

	cl_uint _numI;     // number of inputs
	cl_uint _numC;     // number of columns
	cl_uint _numAC;    // number of active columns at each time step
	cl_uint _numNpC;   // number of neurons per column
	cl_uint _numN;     // number of neurons
	cl_uint _numPDpC;  // number of proximal dendrites per column
	cl_uint _numPD;    // number of proximal dendrites
	cl_uint _numPSpPD; // number of proximal synapses per proximal dendrite
	cl_uint _numPSpC;  // number of proximal synapses per column
	cl_uint _numPS;    // number of proximal synapses
	cl_uint _numDDpN;  // number of distal dendrites per neuron
	cl_uint _numDDpC;  // number of distal dendrites per column
	cl_uint _numDD;    // number of distal dendrites
	cl_uint _numDSpDD; // number of distal synapses per distal dendrite
	cl_uint _numDSpN;  // number of distal synapses per neuron
	cl_uint _numDS;    // number of distal synapses
	cl_uint _sMaxAddr; // maximum address of synapses
	cl_uint _sMaxPerm; // mamimum permanence of synapses
	cl_uint _dThresh;  // dendrite threshold
	cl_uint _sThresh;  // synapse threshold
	cl_uint _sLearn;   // synapse learning rate

	cl::Buffer _inputs;       // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nActives;     // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nWinners;     // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nActivesPrev; // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nWinnersPrev; // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _pdActives;    // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _ddActives;    // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _ddLearns;     // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _psPerms;      // OpenCL buffer of chars (values from 0 to 99)
	cl::Buffer _dsPerms;      // OpenCL buffer of chars (values from 0 to 99)
	cl::Buffer _psAddrs;      // OpenCL buffer of ushorts (values from 0 to _numI)
	cl::Buffer _dsAddrs;      // OpenCL buffer of ushorts (values from 0 to _numN)
	cl::Buffer _pdOverlaps;   // OpenCL buffer of ushorts (values from 0 to _numPSpPD)
	cl::Buffer _ddOverlaps;   // OpenCL buffer of ushorts (values from 0 to _numDSpDD)

	cl::Kernel _randomizeAddresses;
	cl::Kernel _activateDendrites;
	cl::Kernel _learnDendrites;
	cl::Kernel _setNeuronStates;
};

#endif
