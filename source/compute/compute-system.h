// ================
// compute-system.h
// ================

#ifndef COMPUTE_SYSTEM_H
#define COMPUTE_SYSTEM_H

#define __CL_HPP_ENABLE_EXCEPTIONS

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

class ComputeSystem
{
public:
	enum DeviceType {_cpu, _gpu, _all};

	bool init(DeviceType type);
	void printCLInfo();

	cl::Platform getPlatform()
	{
		return _platform;
	}

	cl::Device getDevice()
	{
		return _device;
	}

	cl::Context getContext()
	{
		return _context;
	}

	cl::CommandQueue getQueue()
	{
		return _queue;
	}

private:
	cl::Platform _platform;
	cl::Device _device;
	cl::Context _context;
	cl::CommandQueue _queue;
};

#endif
