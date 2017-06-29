/*
=================
compute_program.h
=================
*/

#ifndef COMPUTE_PROGRAM_H
#define COMPUTE_PROGRAM_H

#include "compute-system.h"

class ComputeProgram
{
public:
	bool loadProgramFromSourceFile(ComputeSystem& cs, const std::string& fileName);

	cl::Program getProgram()
	{
		return _program;
	}

private:
	cl::Program _program;
};

#endif
