// ==========
// stimulae.h
// ==========

#ifndef STIMULAE_H
#define STIMULAE_H

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include <vector>

class Stimulae
{
public:
	void init(ComputeSystem& cs, unsigned int numStimulus);

	void clearStates(ComputeSystem& cs);

	void setStates(ComputeSystem& cs, std::vector<unsigned char> vec);

	std::vector<unsigned char> getStates(ComputeSystem &cs);

	void printStates(ComputeSystem& cs);

public:
	cl_uint numS; // number of stimulae

	cl::Buffer bufferSStates; // uchars (values from 0 to 1)

private:
	const cl_uchar _ZERO_UCHAR = static_cast<cl_uchar>(0);

	size_t _numbytesSStates;
};

#endif
