// ==========
// forest.cpp
// ==========

#include "forest.h"

void Forest::init(ComputeSystem& cs, ComputeProgram& cp, unsigned int numDenPFor, unsigned int numSynPDen, float threshPercent)
{
	numDpF = static_cast<cl_uint>(numDenPFor);
	numSpD = static_cast<cl_uint>(numSynPDen);
	numSpF = static_cast<cl_uint>(numSpD * numDpF);

	unsigned int threshold = numSpD * threshPercent - 1;

	if (threshold == 0)
		threshold = 1;

	dThresh = static_cast<cl_uint>(threshold);

	numbytesSAddrs  = sizeof(cl_uint)  * numSpF;
	numbytesSPerms  = sizeof(cl_uchar) * numSpF;

	bufferSAddrs  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numbytesSAddrs);
	bufferSPerms  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numbytesSPerms);

	cs.getQueue().enqueueFillBuffer(bufferSAddrs, _S_ADDR_MAX, 0, numbytesSAddrs);
	cs.getQueue().enqueueFillBuffer(bufferSPerms, _ZERO_UCHAR, 0, numbytesSPerms);
}
