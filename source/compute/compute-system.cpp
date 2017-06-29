// ==================
// compute-system.cpp
// ==================

#include "compute-system.h"

#if defined __APPLE__ || defined(MACOSX)
#else
    #if defined _WIN32
		#include <windows.h> // needed?
    #else
		#include <GL/glx.h>
    #endif
#endif

#include <iostream>

bool ComputeSystem::init(DeviceType type)
{
	if (type == _none)
	{
		std::cerr << "[cs] No OpenCL context created." << std::endl;
		return true;
	}

	// =============
	// Load Platform
	// =============
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	if (allPlatforms.empty())
	{
		std::cerr << "[cs] No platforms found.  Check your OpenCL installation." << std::endl;
		return true;
	}

//	std::cout << "[cs] "<< allPlatforms.size() << " platforms found." << std::endl;

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
		std::cerr << "[cs] No devices found. Check your OpenCL installation." << std::endl;
		return false;
	}

	_device = allDevices.front();

	// ============
	// Load Context
	// ============

	// See if cl_khr_gl_sharing is in the platform extensions list
	// std::cout << "[cs] Platform extensions: " << _platform.getInfo<CL_PLATFORM_EXTENSIONS>() << std::endl;
	// See if cl_khr_gl_sharing is in the device extensions list
	// std::cout << "[cs] Device extensions: " << _device.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;

	#if defined (__APPLE__) || defined(MACOSX)
		CGLContextObj kCGLContext = CGLGetCurrentContext();
		CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
		cl_context_properties props[] =
		{
			CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
			0
		};
		try
		{
			_context = cl::Context(props);
		}
		catch (cl::Error er)
		{
			std::cout << "ERROR: " <<  er.what() << ": " << er.err() << std::endl;
		}
	#else
		#if defined WIN32
			cl_context_properties props[] =
			{
				CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // wgl Context
				CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),         // wgl Display
				CL_CONTEXT_PLATFORM, (cl_context_properties)(_platform)(),        // OpenCL Platform
				0
			};
			try
			{
				_context = cl::Context(_device, props);
			}
			catch (cl::Error er)
			{
				std::cout << "ERROR: " <<  er.what() << ": " << er.err() << std::endl;
			}
		#else
			cl_context_properties props[] =
			{
				CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),  // glX Context
				CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), // glX Display
				CL_CONTEXT_PLATFORM, (cl_context_properties)(_platform)(),         // OpenCL Platform
				0
			};
			try
			{
				_context = cl::Context(_device, props);
			}
			catch (cl::Error er)
			{
				std::cout << "ERROR: " <<  er.what() << ": " << er.err() << std::endl;
			}
		#endif
	#endif

	// ==========
	// Load Queue
	// ==========
	_queue = cl::CommandQueue(_context, _device);
}

void ComputeSystem::printCLInfo()
{
	std::cout << "[c] OpenCL version: " << _platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "[c] OpenCL platform: " << _platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "[c] OpenCL device: " << _device.getInfo<CL_DEVICE_NAME>() << std::endl;
}
