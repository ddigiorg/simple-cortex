// =========
// pattern.h
// =========

#ifndef PATTERN_H
#define PATTERN_H

#include "compute/compute-system.h"
#include "compute/compute-program.h"

#include <vector>

class Pattern
{
	public:
		void init(ComputeSystem& cs, unsigned int numNo);

		void setStates(ComputeSystem& cs, std::vector<unsigned char> vec);

		std::vector<unsigned char> getStates(ComputeSystem &cs);

		void printStates(ComputeSystem& cs);

	public:
		cl_uint numN; // number of nodes

		size_t sizeStates; // number of bits in buffer 

		cl::Buffer nStates; // OpenCL buffer of chars (values from 0 to 1)
};

#endif
