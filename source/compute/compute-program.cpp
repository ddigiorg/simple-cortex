// ===================
// compute-program.cpp
// ===================

#include "compute-program.h"

#include <fstream>
#include <iostream>

bool ComputeProgram::loadFromFile(ComputeSystem &cs, const std::string &fileName)
{
	std::ifstream sourceFile(fileName);

	if (!sourceFile.is_open())
	{
		std::cerr << "[compute] Could not open file " << fileName << "!" << std::endl;
		return false;
	}

	std::string kernel = "";

	while (!sourceFile.eof() && sourceFile.good())
	{
		std::string line;

		std::getline(sourceFile, line);

		kernel += line + "\n";
	}

	_program = cl::Program(cs.getContext(), kernel);

	if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS)
	{
		std::cerr << "[compute] Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << std::endl;
		return false;
	}

	sourceFile.close();

	return true;
}
