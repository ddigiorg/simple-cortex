// ===========
// pattern.cpp
// ===========

#include "pattern.h"

void Pattern::init(ComputeSystem& cs, unsigned int numNo)
{
	numN = static_cast<cl_uint>(numNo);

	sizeStates = sizeof(cl_char) * numN;

	nStates = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * numN);

	cs.getQueue().enqueueFillBuffer(nStates, static_cast<cl_char>(0), 0, sizeof(cl_char) * numN);
}

void Pattern::setStates(ComputeSystem& cs, std::vector<unsigned char> vec)
{
	cs.getQueue().enqueueWriteBuffer(nStates, CL_TRUE, 0, sizeof(cl_char) * numN, vec.data());
}   

std::vector<unsigned char> Pattern::getStates(ComputeSystem &cs)
{
	std::vector<unsigned char> vec(numN);
	cs.getQueue().enqueueReadBuffer(nStates, CL_TRUE, 0, sizeof(cl_char) * numN, vec.data(), NULL);
	return vec;
}

void Pattern::printStates(ComputeSystem& cs)
{
	std::vector<unsigned char> vec(numN);
	cs.getQueue().enqueueReadBuffer(nStates, CL_TRUE, 0, sizeStates, vec.data(), NULL);

	printf("\nStates: ");

	for (unsigned int i = 0; i < numN; i++)
		printf("%i ", vec[i]);
}
  
