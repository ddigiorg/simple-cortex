// ========
// area.cpp
// ========

#include "area.h"

void Area::init(ComputeSystem& cs, ComputeProgram& cp, unsigned int numNpA)
{
	_numNpA = static_cast<cl_uint>(numNpA);
	_numAN  = static_cast<cl_uint>(1);

	_numbytesNBoosts   = sizeof(cl_uint)  * _numNpA;
	_numbytesNStates   = sizeof(cl_uchar) * _numNpA;
	_numbytesNOverlaps = sizeof(cl_uchar) * _numNpA;
	_numbytesInhibit   = sizeof(cl_uchar);

	_bufferNBoosts   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _numbytesNBoosts);
	_bufferNStates   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _numbytesNStates);
	_bufferNOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _numbytesNOverlaps);
	_bufferInhibit   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _numbytesInhibit);

	clearNBoosts(cs);
	clearNStates(cs);
	clearNOverlaps(cs);
	clearNInhibit(cs);

	_kernelOverlapSynapses = cl::Kernel(cp.getProgram(), "overlapSynapses");
	_kernelActivateNeurons = cl::Kernel(cp.getProgram(), "activateNeurons");
	_kernelLearnSynapses   = cl::Kernel(cp.getProgram(), "learnSynapses");
	_kernelPredictNeurons  = cl::Kernel(cp.getProgram(), "predictNeurons");
	_kernelDecodeNeurons   = cl::Kernel(cp.getProgram(), "decodeNeurons");
}

void Area::encode(ComputeSystem &cs, std::vector<Stimulae> vecStimulae, std::vector<Forest> vecForest)
{
	// Overlap Synapses
	clearNOverlaps(cs);

	for (unsigned int f = 0; f < vecForest.size(); f++)
		overlapSynapses(cs, vecStimulae[f], vecForest[f]);

	// Activate (and potentially Inhibit) Neurons
	clearNStates(cs);
	clearNInhibit(cs);

	_nThresh = static_cast<cl_uint>(vecForest.size());

	activateNeurons(cs);

	cs.getQueue().enqueueReadBuffer(_bufferInhibit, CL_TRUE, 0, _numbytesInhibit, &_inhibit, NULL);

	// If no Neuron Inhibition, activate Neuron with highest Boost value
	if (_inhibit == 0)
	{
		cl_uint arrNBoosts[_numNpA];
		cl_uchar arrNStates[_numNpA];

		cs.getQueue().enqueueReadBuffer(_bufferNBoosts, CL_TRUE, 0, _numbytesNBoosts, &arrNBoosts, NULL);
		cs.getQueue().enqueueReadBuffer(_bufferNStates, CL_TRUE, 0, _numbytesNStates, &arrNStates, NULL);

		for (unsigned int an = 0; an < _numAN; an++)
		{
			// Get max index
			unsigned int maxValue = 0;
			unsigned int maxIndex = 0;

			for (unsigned int n = 0; n < _numNpA; n++)
			{
				if (arrNBoosts[n] > maxValue)
				{
					maxValue = arrNBoosts[n];
					maxIndex = n;
				}
			}

			arrNBoosts[maxIndex] = 0;
			arrNStates[maxIndex] = 1;
		}

		cs.getQueue().enqueueWriteBuffer(_bufferNBoosts, CL_TRUE, 0, _numbytesNBoosts, arrNBoosts);
		cs.getQueue().enqueueWriteBuffer(_bufferNStates, CL_TRUE, 0, _numbytesNStates, arrNStates);
	}
}

void Area::learn(ComputeSystem& cs, std::vector<Stimulae> vecStimulae, std::vector<Forest> vecForest)
{
	for (unsigned int f = 0; f < vecForest.size(); f++)
		learnSynapses(cs, vecStimulae[f], vecForest[f]);
}

void Area::predict(ComputeSystem& cs, std::vector<Stimulae> vecStimulae, std::vector<Forest> vecForest)
{
	// Overlap Synapses
	clearNOverlaps(cs);

	for (unsigned int f = 0; f < vecForest.size(); f++)
		overlapSynapses(cs, vecStimulae[f], vecForest[f]);

	// Predict Neurons
	clearNStates(cs);

	_nThresh = static_cast<cl_uint>(vecForest.size());

	predictNeurons(cs);
}

void Area::decode(ComputeSystem& cs, std::vector<Stimulae> vecStimulae, std::vector<Forest> vecForest)
{
	for (unsigned int f = 0; f < vecForest.size(); f++)
	{
		vecStimulae[f].clearStates(cs);

		decodeNeurons(cs, vecStimulae[f], vecForest[f]);
	}
}

std::vector<unsigned char> Area::getStates(ComputeSystem &cs)
{
	std::vector<unsigned char> vec(_numNpA);
	cs.getQueue().enqueueReadBuffer(_bufferNStates, CL_TRUE, 0, _numbytesNStates, vec.data(), NULL);
	return vec;
}

void Area::printStates(ComputeSystem& cs)
{
	std::vector<unsigned char> vec(_numNpA);
	cs.getQueue().enqueueReadBuffer(_bufferNStates, CL_TRUE, 0, _numbytesNStates, vec.data(), NULL);

	printf("\nnStates: ");

	for (unsigned int i = 0; i < _numNpA; i++)
		printf("%i ", vec[i]);
}

// Private Functions

void Area::clearNBoosts(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_bufferNBoosts, _ZERO_UINT, 0, _numbytesNBoosts);
}

void Area::clearNStates(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_bufferNStates, _ZERO_UCHAR, 0, _numbytesNStates);
}

void Area::clearNOverlaps(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_bufferNOverlaps, _ZERO_UCHAR, 0, _numbytesNOverlaps);
}

void Area::clearNInhibit(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_bufferInhibit, _ZERO_UCHAR, 0, _numbytesInhibit);
}

void Area::overlapSynapses(ComputeSystem& cs, Stimulae stimulae, Forest forest)
{
	_kernelOverlapSynapses.setArg(0, _bufferNOverlaps);
	_kernelOverlapSynapses.setArg(1, stimulae.bufferSStates);
	_kernelOverlapSynapses.setArg(2, forest.bufferSAddrs);
	_kernelOverlapSynapses.setArg(3, forest.bufferSPerms);
	_kernelOverlapSynapses.setArg(4, forest.numSpD);
	_kernelOverlapSynapses.setArg(5, forest.dThresh);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_kernelOverlapSynapses, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Area::activateNeurons(ComputeSystem& cs)
{
	_kernelActivateNeurons.setArg(0, _bufferNBoosts);
	_kernelActivateNeurons.setArg(1, _bufferNStates);
	_kernelActivateNeurons.setArg(2, _bufferNOverlaps);
	_kernelActivateNeurons.setArg(3, _bufferInhibit);
	_kernelActivateNeurons.setArg(4, _S_ADDR_MAX);
	_kernelActivateNeurons.setArg(5, _nThresh);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_kernelActivateNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Area::learnSynapses(ComputeSystem&cs, Stimulae stimulae, Forest forest)
{
	_kernelLearnSynapses.setArg(0, stimulae.bufferSStates);
	_kernelLearnSynapses.setArg(1, stimulae.numS);
	_kernelLearnSynapses.setArg(2, forest.bufferSAddrs);
	_kernelLearnSynapses.setArg(3, forest.bufferSPerms);
	_kernelLearnSynapses.setArg(4, forest.numSpD);
	_kernelLearnSynapses.setArg(5, _bufferNStates);
	_kernelLearnSynapses.setArg(6, _S_PERM_MAX);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_kernelLearnSynapses, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Area::predictNeurons(ComputeSystem& cs)
{
	_kernelPredictNeurons.setArg(0, _bufferNStates);
	_kernelPredictNeurons.setArg(1, _bufferNOverlaps);
	_kernelPredictNeurons.setArg(2, _nThresh);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_kernelPredictNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Area::decodeNeurons(ComputeSystem& cs, Stimulae stimulae, Forest forest)
{
	_kernelDecodeNeurons.setArg(0, stimulae.bufferSStates);
	_kernelDecodeNeurons.setArg(1, _bufferNStates);
	_kernelDecodeNeurons.setArg(2, forest.bufferSAddrs);
	_kernelDecodeNeurons.setArg(3, forest.bufferSPerms);
	_kernelDecodeNeurons.setArg(4, forest.numSpD);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_kernelDecodeNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}
