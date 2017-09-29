// ======
// area.h
// ======

#ifndef AREA_H
#define AREA_H

#include "utils/utils.h"

#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "stimuli.h"
#include "forest.h"

#include <vector>

class Area
{
public:
	void init(ComputeSystem& cs, ComputeProgram& cp, unsigned int numNpA);

	void encode (ComputeSystem& cs, std::vector<Stimuli> vecStimuli, std::vector<Forest> vecForest);
	void learn  (ComputeSystem& cs, std::vector<Stimuli> vecStimuli, std::vector<Forest> vecForest);
	void predict(ComputeSystem& cs, std::vector<Stimuli> vecStimuli, std::vector<Forest> vecForest);
	void decode (ComputeSystem& cs, std::vector<Stimuli> vecStimuli, std::vector<Forest> vecForest);

	std::vector<unsigned char>  getStates(ComputeSystem& cs);

	void printStates(ComputeSystem& cs);

private:
	void clearNBoosts(ComputeSystem& cs);
	void clearNStates(ComputeSystem& cs);
	void clearNOverlaps(ComputeSystem& cs);
	void clearNInhibit(ComputeSystem& cs);

	void overlapSynapses(ComputeSystem& cs, Stimuli stimuli, Forest forest);
	void activateNeurons(ComputeSystem& cs);
	void learnSynapses(ComputeSystem& cs, Stimuli stimuli, Forest forest);
	void predictNeurons(ComputeSystem& cs);
	void decodeNeurons(ComputeSystem& cs, Stimuli stimuli, Forest forest);

private:
	const cl_uint _ZERO_UINT = static_cast<cl_uint>(0);
	const cl_uint _S_ADDR_MAX = static_cast<cl_uint>(4294967295);
	const cl_uchar _ZERO_UCHAR = static_cast<cl_uchar>(0);
	const cl_uchar _S_PERM_MAX = static_cast<cl_char>(99);

	cl_uint _numNpA;  // number of neurons per area (cant go over 65535 neurons unless synapse address variables are modified)i // !!!
	cl_uint _numDpN;  // number of dendrites per neuron
	cl_uint _numAN;   // number of active neurons during a single time step
	cl_uint _nThresh; // neuron activation threshold (how many active dendrites needed to activate neuron)

	cl_uchar _inhibit;

	size_t _numbytesNBoosts;
	size_t _numbytesNStates;
	size_t _numbytesNOverlaps;
	size_t _numbytesInhibit;

	cl::Buffer _bufferNBoosts;   // uints (values from 0 to 4,294,967,295)
	cl::Buffer _bufferNStates;   // uchars (values from 0 to 1)
	cl::Buffer _bufferNOverlaps; // uchars (values from 0 to 255)
	cl::Buffer _bufferInhibit;   // uchar (value from 0 to 1)

	cl::Kernel _kernelOverlapSynapses;
	cl::Kernel _kernelActivateNeurons;
	cl::Kernel _kernelLearnSynapses;
	cl::Kernel _kernelPredictNeurons;
	cl::Kernel _kernelDecodeNeurons;

	cl::NDRange _range;
};

#endif
