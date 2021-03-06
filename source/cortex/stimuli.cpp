// ===========
// stimuli.cpp
// ===========

#include "stimuli.h"

void Stimuli::init(ComputeSystem& cs, unsigned int numStimulus)
{
	numS = static_cast<cl_uint>(numStimulus);

	_numbytesSStates = sizeof(cl_uchar) * numS;

	bufferSStates = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _numbytesSStates);

	cs.getQueue().enqueueFillBuffer(bufferSStates, _ZERO_UCHAR, 0, _numbytesSStates);
}

void Stimuli::clearStates(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(bufferSStates, _ZERO_UCHAR, 0, _numbytesSStates);
}	

void Stimuli::setStates(ComputeSystem& cs, std::vector<unsigned char> vec)
{
	cs.getQueue().enqueueWriteBuffer(bufferSStates, CL_TRUE, 0, _numbytesSStates, vec.data());
} 

std::vector<unsigned char> Stimuli::getStates(ComputeSystem &cs)
{
	std::vector<unsigned char> vecStates(numS);
	cs.getQueue().enqueueReadBuffer(bufferSStates, CL_TRUE, 0, _numbytesSStates, vecStates.data(), NULL);
	return vecStates;
}

void Stimuli::printStates(ComputeSystem& cs)
{
	std::vector<unsigned char> vecStates(numS);
	cs.getQueue().enqueueReadBuffer(bufferSStates, CL_TRUE, 0, _numbytesSStates, vecStates.data(), NULL);

	for (unsigned int s = 0; s < numS; s++)
		printf("%i ", vecStates[s]);
}
