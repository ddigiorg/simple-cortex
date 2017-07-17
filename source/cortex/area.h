// ======
// area.h
// ======

#ifndef AREA_H
#define AREA_H

#include "utils/utils.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include <vector>

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

class Area
{
public:
	Area(
		ComputeSystem &cs,
		ComputeProgram &cp,
		unsigned int numN,
		std::vector<unsigned int> numVperP, // number of values per pattern
		std::vector<unsigned int> numSperD  // number of synapses per dendrite
	);

	void encode(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums);
	void learn(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums);
	void predict(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums);
	void decode(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums);

	void setPattern(ComputeSystem& cs, unsigned int p, std::vector<char> vec);
	void setPatternFromActiveNeurons(ComputeSystem& cs, unsigned int p);
	void setPatternFromPredictNeurons(ComputeSystem& cs, unsigned int p);

	std::vector<char> getPattern(ComputeSystem &cs, unsigned int p);
	std::vector<unsigned short> getSynapseAddrs(ComputeSystem &cs, unsigned int d);
	std::vector<char> getSynapsePerms(ComputeSystem &cs, unsigned int d);
	std::vector<char> getNeuronActives(ComputeSystem &cs);
	std::vector<char> getNeuronPredicts(ComputeSystem &cs);
	std::vector<unsigned short> getNeuronBoosts(ComputeSystem &cs);


private:
	cl::NDRange _range;

	std::vector<Pattern> _patterns;
	std::vector<Dendrite> _dendrites;

	cl_uint _numN;       // number of neurons per area
	cl_uint _numAN;      // number of active neurons during a single time step
	cl_uint _sPermMax;   // synapse permanence max value (99)
	cl_uint _sAddrMax;   // synapse address max value (65535)

	cl::Buffer _nBoosts;   // OpenCL buffer of ushorts (values from 0 to 65535)
	cl::Buffer _nPredicts; // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nActives;  // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nOverlaps; // OpenCL buffer of chars (values from 0 to 255)
	cl::Buffer _inhibitFlag; // OpenCL buffer of char (value from 0 to 1)

	cl::Kernel _overlapDendrites;
	cl::Kernel _learnSynapses;
	cl::Kernel _activateNeurons;
	cl::Kernel _predictNeurons;
	cl::Kernel _decodeNeurons;
};

#endif
