// ==================
// compute-system.cpp
// ==================

#include "compute-system.h"

#include <iostream>

bool ComputeSystem::init(DeviceType type)
{
	// =============
	// Load Platform
	// =============
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	if (allPlatforms.empty())
	{
		std::cerr << "[compute] No platforms found.  Check your OpenCL installation." << std::endl;
		return false;
	}

	_platform = allPlatforms.front();

	// ===========
	// Load Device
	// ===========
	std::vector<cl::Device>allDevices;

	switch (type)
	{
	case _cpu:
		_platform.getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
		break;
	case _gpu:
		_platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
		break;
	case _all:
		_platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
		break;
    }

	if (allDevices.empty())
	{
		std::cerr << "[compute] No devices found. Check your OpenCL installation." << std::endl;
		return false;
	}

	_device = allDevices.front();

	// ============
	// Load Context
	// ============
	_context = _device;

	// ==========
	// Load Queue
	// ==========
	_queue = cl::CommandQueue(_context, _device);

	return true;
}

void ComputeSystem::printCLInfo()
{
	std::cout << "[compute] OpenCL version: "  << _platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "[compute] OpenCL platform: " << _platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "[compute] OpenCL device: "   << _device.getInfo<CL_DEVICE_NAME>() << std::endl;
}
