// ======
// area.h
// ======

#ifndef AREA_H
#define AREA_H

#include "utils/utils.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "pattern.h"

#include <vector>

class Forest
{
	public:
		cl_uint numSpD;  // number of synapses per dendrite
		cl_uint numSpF;  // number of synapses per forest
		cl_uint dThresh; // dendrite activation threshold

		size_t sizeAddrs; // number of bits in buffer
		size_t sizePerms; // number of bits in buffer

		cl::Buffer sAddrs; // OpenCL buffer of ushorts (values from 0 to 65535)
		cl::Buffer sPerms; // OpenCL buffer of chars (values from 0 to 99)
};

class Area
{
public:

	void init(ComputeSystem &cs, ComputeProgram &cp, unsigned int numN, std::vector<unsigned int> numSpDs);

	void encode(ComputeSystem& cs, std::vector<Pattern> patterns);
	void learn(ComputeSystem& cs, std::vector<Pattern> patterns);
	void predict(ComputeSystem& cs, std::vector<Pattern> patterns, std::vector<unsigned int> forests);
	void decode(ComputeSystem& cs, std::vector<Pattern> patterns, std::vector<unsigned int> forests);

	std::vector<unsigned char> getStates(ComputeSystem& cs);
	std::vector<unsigned short> getBoosts(ComputeSystem& cs);
	std::vector<unsigned short> getAddrs(ComputeSystem& cs, unsigned int f);
	std::vector<unsigned char> getPerms(ComputeSystem& cs, unsigned int f);

	void printStates(ComputeSystem& cs);
	void printBoosts(ComputeSystem& cs);
	void printAddrs(ComputeSystem& cs, unsigned int f);
	void printPerms(ComputeSystem& cs, unsigned int f);

private:
	const cl_uint _MAX_ADDR = static_cast<cl_ushort>(65535);
	const cl_uint _MAX_PERM = static_cast<cl_char>(99);

	cl_ushort _zeroBoosts     = static_cast<cl_ushort>(0);
	cl_uchar _zeroStates      = static_cast<cl_uchar>(0);
	cl_uchar _zeroOverlaps    = static_cast<cl_uchar>(0);
	cl_uchar _zeroInhibitFlag = static_cast<cl_uchar>(0);
	cl_uchar _zeroAddrs       = static_cast<cl_uchar>(_MAX_ADDR);
	cl_uchar _zeroPerms       = static_cast<cl_uchar>(0);

	cl_uint _numNpA; // number of neurons per area (cant go over 65535 neurons unless synapse address variables are modified)
	cl_uint _numDpN; // number of dendrites per neuron
	cl_uint _numAN;  // number of active neurons during a single time step

	size_t _sizeBoosts;      // number of bits in buffer
	size_t _sizeStates;      // number of bits in buffer
	size_t _sizeOverlaps;    // number of bits in buffer
	size_t _sizeInhibitFlag; // number of bits in buffer

	cl::Buffer _nBoosts;     // OpenCL buffer of ushorts (values from 0 to 65535)
	cl::Buffer _nStates;     // OpenCL buffer of chars (values from 0 to 1)
	cl::Buffer _nOverlaps;   // OpenCL buffer of chars (values from 0 to 255)
	cl::Buffer _inhibitFlag; // OpenCL buffer of char (value from 0 to 1)

	std::vector<Forest> _forests;

	cl::Kernel _overlapDendrites;
	cl::Kernel _learnSynapses;
	cl::Kernel _activateNeurons;
	cl::Kernel _predictNeurons;
	cl::Kernel _decodeNeurons;

	cl::NDRange _range;
};

#endif
