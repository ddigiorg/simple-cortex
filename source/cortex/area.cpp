// ========
// area.cpp
// ========

#include "area.h"

void Area::init(ComputeSystem &cs, ComputeProgram &cp, unsigned int numNpA, std::vector<unsigned int> numSpDs)
{
	_numNpA = static_cast<cl_uint>(numNpA);
	_numDpN = static_cast<cl_uint>(numSpDs.size());
	_numAN  = static_cast<cl_uint>(1);

	_sizeBoosts = sizeof(cl_ushort) * _numNpA;
	_sizeStates = sizeof(cl_char) * _numNpA;
	_sizeOverlaps = sizeof(cl_char) * _numNpA;
	_sizeInhibitFlag = sizeof(cl_char);

	_nBoosts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _sizeBoosts);
	_nStates = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _sizeStates);
	_nOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _sizeOverlaps);
	_inhibitFlag = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _sizeInhibitFlag);

	cs.getQueue().enqueueFillBuffer(_nBoosts, _zeroBoosts, 0, _sizeBoosts);
	cs.getQueue().enqueueFillBuffer(_nStates, _zeroStates, 0, _sizeStates);
	cs.getQueue().enqueueFillBuffer(_nOverlaps, _zeroOverlaps, 0, _sizeOverlaps);
	cs.getQueue().enqueueFillBuffer(_inhibitFlag, _zeroInhibitFlag, 0, _sizeInhibitFlag);

	_forests.resize(_numDpN);

	for (unsigned int d = 0; d < _numDpN; d++)
	{
		_forests[d].numSpD = static_cast<cl_uint>(numSpDs[d]);
		_forests[d].numSpF = static_cast<cl_uint>(numSpDs[d] * _numNpA);

		float percent = 1.0; // 0.5
		unsigned int threshold = _forests[d].numSpD * percent - 1;

		if (threshold == 0)
			threshold = 1;

		_forests[d].dThresh = static_cast<cl_uint>(threshold);

		_forests[d].sizeAddrs = sizeof(cl_ushort) * _forests[d].numSpF;
		_forests[d].sizePerms = sizeof(cl_char) * _forests[d].numSpF;

		_forests[d].sAddrs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _forests[d].sizeAddrs);
		_forests[d].sPerms = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _forests[d].sizePerms);

		cs.getQueue().enqueueFillBuffer(_forests[d].sAddrs, _zeroAddrs, 0, _forests[d].sizeAddrs);
		cs.getQueue().enqueueFillBuffer(_forests[d].sPerms, _zeroPerms, 0, _forests[d].sizePerms);
	}

	_overlapDendrites = cl::Kernel(cp.getProgram(), "overlapDendrites");
	_learnSynapses    = cl::Kernel(cp.getProgram(), "learnSynapses");
	_activateNeurons  = cl::Kernel(cp.getProgram(), "activateNeurons");
	_predictNeurons   = cl::Kernel(cp.getProgram(), "predictNeurons");
	_decodeNeurons    = cl::Kernel(cp.getProgram(), "decodeNeurons");
}

void Area::encode(ComputeSystem &cs, std::vector<Pattern> patterns)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, _zeroOverlaps, 0, _sizeOverlaps);

	// Overlap Dendrites
	for (unsigned int d = 0; d < _numDpN; d++)
	{
		_overlapDendrites.setArg(0, patterns[d].nStates);
		_overlapDendrites.setArg(1, _nOverlaps);
		_overlapDendrites.setArg(2, _forests[d].sAddrs);
		_overlapDendrites.setArg(3, _forests[d].sPerms);
		_overlapDendrites.setArg(4, _forests[d].numSpD);
		_overlapDendrites.setArg(5, _forests[d].dThresh);

		_range = cl::NDRange(_numNpA);
		cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}

	// Neuron Activation
	cs.getQueue().enqueueFillBuffer(_nStates, _zeroStates, 0, _sizeStates);
	cs.getQueue().enqueueFillBuffer(_inhibitFlag, _zeroInhibitFlag, 0, _sizeInhibitFlag);

	cl_uint nThresh = static_cast<cl_uint>(_numDpN);

	_activateNeurons.setArg(0, _nBoosts);
	_activateNeurons.setArg(1, _nStates);
	_activateNeurons.setArg(2, _nOverlaps);
	_activateNeurons.setArg(3, _inhibitFlag);
	_activateNeurons.setArg(4, _MAX_ADDR);
	_activateNeurons.setArg(5, nThresh);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_activateNeurons, cl::NullRange, _range);
	cs.getQueue().finish();

	cl_char inhibitFlagArr[1];

	cs.getQueue().enqueueReadBuffer(_inhibitFlag, CL_TRUE, 0, _sizeInhibitFlag, &inhibitFlagArr, NULL);

	// Neuron Inhibition
	if (inhibitFlagArr[0] == 0)
	{
		cl_ushort nBoostsArr[_numNpA];
		cl_char nStatesArr[_numNpA];

		cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, _sizeBoosts, &nBoostsArr, NULL);
		cs.getQueue().enqueueReadBuffer(_nStates, CL_TRUE, 0, _sizeStates, &nStatesArr, NULL);

		for (unsigned int an = 0; an < _numAN; an++)
		{
			unsigned int maxValue = 0;
			unsigned int maxIndex = 0;

			// get max value
			for (unsigned int n = 0; n < _numNpA; n++)
			{
				if (nBoostsArr[n] > maxValue)
				{
					maxValue = nBoostsArr[n];
					maxIndex = n;
				}
			}

			nStatesArr[maxIndex] = 1;
			nBoostsArr[maxIndex] = 0;
		}

		cs.getQueue().enqueueWriteBuffer(_nBoosts, CL_TRUE, 0, _sizeBoosts, nBoostsArr);
		cs.getQueue().enqueueWriteBuffer(_nStates, CL_TRUE, 0, _sizeStates, nStatesArr);
	}
}

void Area::learn(ComputeSystem& cs, std::vector<Pattern> patterns)
{
	for (unsigned int d = 0; d < _numDpN; d++)
	{
		_learnSynapses.setArg(0, patterns[d].nStates);
		_learnSynapses.setArg(1, patterns[d].numN);
		_learnSynapses.setArg(2, _forests[d].sAddrs);
		_learnSynapses.setArg(3, _forests[d].sPerms);
		_learnSynapses.setArg(4, _forests[d].numSpD);
		_learnSynapses.setArg(5, _nStates);
		_learnSynapses.setArg(6, _MAX_PERM);

		_range = cl::NDRange(_numNpA);
		cs.getQueue().enqueueNDRangeKernel(_learnSynapses, cl::NullRange, _range);
		cs.getQueue().finish();
	}
}

void Area::predict(ComputeSystem& cs, std::vector<Pattern> patterns, std::vector<unsigned int> forests)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, _zeroOverlaps, 0, _sizeOverlaps);
	cs.getQueue().enqueueFillBuffer(_nStates, _zeroStates, 0, _sizeStates);

	// Overlap Dendrites
	for (unsigned int i = 0; i < forests.size(); i++)
	{
		_overlapDendrites.setArg(0, patterns[i].nStates);
		_overlapDendrites.setArg(1, _nOverlaps);
		_overlapDendrites.setArg(2, _forests[forests[i]].sAddrs);
		_overlapDendrites.setArg(3, _forests[forests[i]].sPerms);
		_overlapDendrites.setArg(4, _forests[forests[i]].numSpD);
		_overlapDendrites.setArg(5, _forests[forests[i]].dThresh);

		_range = cl::NDRange(_numNpA);
		cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}

	// Predict Neurons
	cl_uint nThresh = static_cast<cl_uint>(forests.size());

	_predictNeurons.setArg(0, _nStates);
	_predictNeurons.setArg(1, _nOverlaps);
	_predictNeurons.setArg(2, nThresh);

	_range = cl::NDRange(_numNpA);
	cs.getQueue().enqueueNDRangeKernel(_predictNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Area::decode(ComputeSystem& cs, std::vector<Pattern> patterns, std::vector<unsigned int> forests)
{
	for (unsigned int i = 0; i < forests.size(); i++)
	{
		cs.getQueue().enqueueFillBuffer(patterns[i].nStates, static_cast<cl_char>(0), 0, sizeof(cl_char) * patterns[i].numN);

		_decodeNeurons.setArg(0, patterns[i].nStates);
		_decodeNeurons.setArg(1, _nStates);
		_decodeNeurons.setArg(2, _forests[forests[i]].sAddrs);
		_decodeNeurons.setArg(3, _forests[forests[i]].sPerms);
		_decodeNeurons.setArg(4, _forests[forests[i]].numSpD);

		_range = cl::NDRange(_numNpA);
		cs.getQueue().enqueueNDRangeKernel(_decodeNeurons, cl::NullRange, _range);
		cs.getQueue().finish();
	}
}

std::vector<unsigned char> Area::getStates(ComputeSystem &cs)
{
	std::vector<unsigned char> vec(_numNpA);
	cs.getQueue().enqueueReadBuffer(_nStates, CL_TRUE, 0, _sizeStates, vec.data(), NULL);
	return vec;
}

std::vector<unsigned short> Area::getBoosts(ComputeSystem &cs)
{
	std::vector<unsigned short> vec(_numNpA);
	cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, _sizeBoosts, vec.data(), NULL);
	return vec;
}

std::vector<unsigned short> Area::getAddrs(ComputeSystem &cs, unsigned int f)
{
	std::vector<unsigned short> vec(_forests[f].numSpF);
	cs.getQueue().enqueueReadBuffer(_forests[f].sAddrs, CL_TRUE, 0, _forests[f].sizeAddrs, vec.data(), NULL);
	return vec;
}

std::vector<unsigned char> Area::getPerms(ComputeSystem &cs, unsigned int f)
{
	std::vector<unsigned char> vec(_forests[f].numSpF);
	cs.getQueue().enqueueReadBuffer(_forests[f].sPerms, CL_TRUE, 0, _forests[f].sizePerms, vec.data(), NULL);
	return vec;
}

void Area::printStates(ComputeSystem& cs)
{
	std::vector<unsigned char> vec(_numNpA);
	cs.getQueue().enqueueReadBuffer(_nStates, CL_TRUE, 0, _sizeStates, vec.data(), NULL);

	printf("\nStates: ");

	for (unsigned int i = 0; i < _numNpA; i++)
		printf("%i ", vec[i]);
}

void Area::printBoosts(ComputeSystem& cs)
{
	std::vector<unsigned short> vec(_numNpA);
	cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, _sizeBoosts, vec.data(), NULL);

	printf("\nBoosts: ");

	for (unsigned int i = 0; i < _numNpA; i++)
		printf("%i ", vec[i]);
}

void Area::printAddrs(ComputeSystem& cs, unsigned int f)
{
	std::vector<unsigned short> vec(_forests[f].numSpF);
	cs.getQueue().enqueueReadBuffer(_forests[f].sAddrs, CL_TRUE, 0, _forests[f].sizeAddrs, vec.data(), NULL);

	printf("\nAddrs%i: ", f);

	for (unsigned int i = 0; i < _forests[f].numSpF; i++)
	{
		if (vec[i] > 999)
			printf("XXX ");
		else
			printf("%i ", vec[i]);
	}
}

void Area::printPerms(ComputeSystem& cs, unsigned int f)
{
	std::vector<unsigned char> vec(_forests[f].numSpF);
	cs.getQueue().enqueueReadBuffer(_forests[f].sPerms, CL_TRUE, 0, _forests[f].sizePerms, vec.data(), NULL);

	printf("\nPerms%i: ", f);

	for (unsigned int i = 0; i < _forests[f].numSpF; i++)
		printf("%i ", vec[i]);
}
