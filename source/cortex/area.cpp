// ========
// area.cpp
// ========

#include "area.h"

Area::Area(
	ComputeSystem &cs,
	ComputeProgram &cp,
	unsigned int numN,
	std::vector<unsigned int> numVperP,
	std::vector<unsigned int> numSperD)
{
	// Initialize Variables
	_numN = static_cast<cl_uint>(numN);
	_numAN = static_cast<cl_uint>(1);
	_sPermMax = static_cast<cl_uint>(99);
	_sAddrMax = static_cast<cl_uint>(65535);

	_patterns.resize(numVperP.size());

	for (unsigned int p = 0; p < numVperP.size(); p++)
	{
		_patterns[p].numV = static_cast<cl_uint>(numVperP[p]);

		_patterns[p].values = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _patterns[p].numV);

		cs.getQueue().enqueueFillBuffer(_patterns[p].values, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[p].numV);
	}

	_dendrites.resize(numSperD.size());

	for (unsigned int d = 0; d < numSperD.size(); d++)
	{
		_dendrites[d].numSperD = static_cast<cl_uint>(numSperD[d]);
		_dendrites[d].numS     = static_cast<cl_uint>(numSperD[d] * _numN);
		_dendrites[d].dThresh  = static_cast<cl_uint>(1);

		_dendrites[d].sAddrs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _dendrites[d].numS);
		_dendrites[d].sPerms = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char)   * _dendrites[d].numS);

		cs.getQueue().enqueueFillBuffer(_dendrites[d].sAddrs, static_cast<cl_ushort>(_sAddrMax), 0, sizeof(cl_ushort) * _dendrites[d].numS);
		cs.getQueue().enqueueFillBuffer(_dendrites[d].sPerms, static_cast<cl_char>(0),           0, sizeof(cl_char)   * _dendrites[d].numS);
	}

	_nBoosts   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numN);
	_nPredicts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nActives  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_inhibitFlag = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char));

	cs.getQueue().enqueueFillBuffer(_nBoosts,   static_cast<cl_ushort>(0), 0, sizeof(cl_ushort) * _numN);
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives,  static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_inhibitFlag, static_cast<cl_char>(0), 0, sizeof(cl_char));

	// Initialize Kernels
	_overlapDendrites = cl::Kernel(cp.getProgram(), "overlapDendrites");
	_learnSynapses    = cl::Kernel(cp.getProgram(), "learnSynapses");
	_activateNeurons  = cl::Kernel(cp.getProgram(), "activateNeurons");
	_predictNeurons   = cl::Kernel(cp.getProgram(), "predictNeurons");
	_decodeNeurons    = cl::Kernel(cp.getProgram(), "decodeNeurons");
}

void Area::encode(ComputeSystem &cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Overlap Dendrites
	for (unsigned int i = 0; i < dNums.size(); i++)
	{
		_overlapDendrites.setArg(0, _nOverlaps);
		_overlapDendrites.setArg(1, _patterns[pNums[i]].values);
		_overlapDendrites.setArg(2, _dendrites[dNums[i]].sAddrs);
		_overlapDendrites.setArg(3, _dendrites[dNums[i]].sPerms);
		_overlapDendrites.setArg(4, _dendrites[dNums[i]].numSperD);
		_overlapDendrites.setArg(5, _dendrites[dNums[i]].dThresh);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}

	// Neuron Activation
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_inhibitFlag, static_cast<cl_char>(0), 0, sizeof(cl_char));

	cl_uint nThresh = static_cast<cl_uint>(dNums.size()); // consider making a function parameter

	_activateNeurons.setArg(0, _nBoosts);
	_activateNeurons.setArg(1, _nActives);
	_activateNeurons.setArg(2, _nOverlaps);
	_activateNeurons.setArg(3, _inhibitFlag);
	_activateNeurons.setArg(4, _sAddrMax);
	_activateNeurons.setArg(5, nThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_activateNeurons, cl::NullRange, _range);
	cs.getQueue().finish();

	cl_char inhibitFlagArr[1];

	cs.getQueue().enqueueReadBuffer(_inhibitFlag, CL_TRUE, 0, sizeof(cl_char), &inhibitFlagArr, NULL);

	// Neuron Inhibition
	if (inhibitFlagArr[0] == 0)
	{
		cl_ushort nBoostsArr[_numN];
		cl_char nActivesArr[_numN];

		cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, &nBoostsArr, NULL);
		cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, &nActivesArr, NULL);

		for (unsigned int an = 0; an < _numAN; an++)
		{
			unsigned int maxValue = 0;
			unsigned int maxIndex = 0;

			// get max value
			for (unsigned int n = 0; n < _numN; n++)
			{
				if (nBoostsArr[n] > maxValue)
				{
					maxValue = nBoostsArr[n];
					maxIndex = n;
				}
			}

			nActivesArr[maxIndex] = 1;
			nBoostsArr[maxIndex] = 0;
		}

		cs.getQueue().enqueueWriteBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, nBoostsArr);
		cs.getQueue().enqueueWriteBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, nActivesArr);
	}
}

void Area::learn(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums)
{
	for (unsigned int i = 0; i < dNums.size(); i++)
	{
		_learnSynapses.setArg(0, _patterns[pNums[i]].values);
		_learnSynapses.setArg(1, _patterns[pNums[i]].numV);
		_learnSynapses.setArg(2, _dendrites[dNums[i]].sAddrs);
		_learnSynapses.setArg(3, _dendrites[dNums[i]].sPerms);
		_learnSynapses.setArg(4, _dendrites[dNums[i]].numSperD);
		_learnSynapses.setArg(5, _nActives);
		_learnSynapses.setArg(6, _sPermMax);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_learnSynapses, cl::NullRange, _range);
		cs.getQueue().finish();
	}
}

void Area::predict(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Overlap Dendrites
	for (unsigned int i = 0; i < dNums.size(); i++)
	{
		_overlapDendrites.setArg(0, _nOverlaps);
		_overlapDendrites.setArg(1, _patterns[pNums[i]].values);
		_overlapDendrites.setArg(2, _dendrites[dNums[i]].sAddrs);
		_overlapDendrites.setArg(3, _dendrites[dNums[i]].sPerms);
		_overlapDendrites.setArg(4, _dendrites[dNums[i]].numSperD);
		_overlapDendrites.setArg(5, _dendrites[dNums[i]].dThresh);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}

	// Predict Neurons
	cl_uint nThresh = static_cast<cl_uint>(dNums.size()); // consider making a function parameter

	_predictNeurons.setArg(0, _nPredicts);
	_predictNeurons.setArg(1, _nOverlaps);
	_predictNeurons.setArg(2, nThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_predictNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Area::decode(ComputeSystem& cs, std::vector<unsigned int> pNums, std::vector<unsigned int> dNums)
{
	for (unsigned int i = 0; i < dNums.size(); i++)
	{
		cs.getQueue().enqueueFillBuffer(_patterns[pNums[i]].values, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[pNums[i]].numV);

		_decodeNeurons.setArg(0, _patterns[pNums[i]].values);
		_decodeNeurons.setArg(1, _nPredicts);
		_decodeNeurons.setArg(2, _dendrites[dNums[i]].sAddrs);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_decodeNeurons, cl::NullRange, _range);
		cs.getQueue().finish();
	}
}

void Area::setPattern(ComputeSystem& cs, unsigned int p, std::vector<char> vec)
{
	cs.getQueue().enqueueWriteBuffer(_patterns[p].values, CL_TRUE, 0, sizeof(cl_char) * _patterns[p].numV, vec.data());
}

void Area::setPatternFromActiveNeurons(ComputeSystem& cs, unsigned int p)
{
	cs.getQueue().enqueueCopyBuffer(_nActives, _patterns[p].values, 0, 0, sizeof(cl_char) * _numN);
}

void Area::setPatternFromPredictNeurons(ComputeSystem& cs, unsigned int p)
{
	cs.getQueue().enqueueCopyBuffer(_nPredicts, _patterns[p].values, 0, 0, sizeof(cl_char) * _numN);
}

std::vector<char> Area::getPattern(ComputeSystem &cs, unsigned int p)
{
	std::vector<char> vec(_patterns[p].numV);
	cs.getQueue().enqueueReadBuffer(_patterns[p].values, CL_TRUE, 0, sizeof(cl_char) * _patterns[p].numV, vec.data(), NULL);
	return vec;
}

std::vector<unsigned short> Area::getSynapseAddrs(ComputeSystem &cs, unsigned int d)
{
	std::vector<unsigned short> vec(_dendrites[d].numS);
	cs.getQueue().enqueueReadBuffer(_dendrites[d].sAddrs, CL_TRUE, 0, sizeof(cl_ushort) * _dendrites[d].numS, vec.data(), NULL);
	return vec;
}

std::vector<char> Area::getSynapsePerms(ComputeSystem &cs, unsigned int d)
{
	std::vector<char> vec(_dendrites[d].numS);
	cs.getQueue().enqueueReadBuffer(_dendrites[d].sPerms, CL_TRUE, 0, sizeof(cl_char) * _dendrites[d].numS, vec.data(), NULL);
	return vec;
}

std::vector<char> Area::getNeuronActives(ComputeSystem &cs)
{
	std::vector<char> vec(_numN);
	cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, vec.data(), NULL);
	return vec;
}

std::vector<char> Area::getNeuronPredicts(ComputeSystem &cs)
{
	std::vector<char> vec(_numN);
	cs.getQueue().enqueueReadBuffer(_nPredicts, CL_TRUE, 0, sizeof(cl_char) * _numN, vec.data(), NULL);
	return vec;
}

std::vector<unsigned short> Area::getNeuronBoosts(ComputeSystem &cs)
{
	std::vector<unsigned short> vec(_numN);
	cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, vec.data(), NULL);
	return vec;
}
