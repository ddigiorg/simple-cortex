// ========
// forest.h
// ========

#ifndef FOREST_H
#define FOREST_H

#include "compute/compute-system.h"
#include "compute/compute-program.h"

class Forest
{
public:
	void init(
		ComputeSystem& cs, ComputeProgram& cp, unsigned int numDenPFor, unsigned int numSynPDen, float threshPercent);

	void encode(ComputeSystem& cs, cl::Buffer bufferStimulae);
	void learn (ComputeSystem& cs);

public:
	const cl_uint _ZERO_UINT = static_cast<cl_uint>(0);
	const cl_uint _S_ADDR_MAX = static_cast<cl_uint>(4294967295);
	const cl_uchar _ZERO_UCHAR = static_cast<cl_uchar>(0);
	const cl_uchar _S_PERM_MAX = static_cast<cl_char>(99);

	cl_uint numDpF;  // number of dendrites per forest
    cl_uint numSpD;  // number of synapses per dendrite
    cl_uint numSpF;  // number of synapses per forest
    cl_uint dThresh; // dendrite activation threshold (how many active synapses needed to activate dendrite)

    size_t numbytesSAddrs;
    size_t numbytesSPerms;

    cl::Buffer bufferSAddrs;  // uints (values from 0 to 4,294,967,295)
    cl::Buffer bufferSPerms;  // uchars (values from 0 to 99)
};

#endif
