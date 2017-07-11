// ==========
// region.cpp
// ==========

#include "region.h"

#include <iostream>
#include <cstdlib>

Region::Region(
	ComputeSystem &cs,
	ComputeProgram &cp,
	std::mt19937 rng,
	unsigned int numN,
	std::vector<unsigned int> numIn,
	std::vector<unsigned int> numSpD
)
{
	_rng = rng;

	// Initialize Variables
	_numN   = static_cast<cl_uint>(numN);
	_numAN  = static_cast<cl_uint>(_numN * 0.02);
	_numDpN = static_cast<cl_uint>(numSpD.size());
	_nActThresh = static_cast<cl_uint>(1); // !!!
	_nPreThresh = static_cast<cl_uint>(1); // !!!
	_sPermMax = static_cast<cl_uint>(99);
	_sAddrMax = static_cast<cl_uint>(65535);

	if (_numAN == 0)
		_numAN = 1;

	_dendrites.resize(_numDpN);

	for (unsigned int d = 0; d < _numDpN; d++)
	{
		_dendrites[d].numIn   = static_cast<cl_uint>(numIn[d]);
		_dendrites[d].numSpD  = static_cast<cl_uint>(numSpD[d]);
		_dendrites[d].numS    = static_cast<cl_uint>(numSpD[d] * numN);
		_dendrites[d].dThresh = static_cast<cl_uint>(1);

		_dendrites[d].inputs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char)   * _dendrites[d].numIn);
		_dendrites[d].addrs  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _dendrites[d].numS);
		_dendrites[d].perms  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char)   * _dendrites[d].numS);

		cs.getQueue().enqueueFillBuffer(_dendrites[d].inputs, static_cast<cl_char>(0),    0, sizeof(cl_char)   * _dendrites[d].numIn);
		cs.getQueue().enqueueFillBuffer(_dendrites[d].addrs,  static_cast<cl_ushort>(_sAddrMax), 0, sizeof(cl_ushort) * _dendrites[d].numS);
		cs.getQueue().enqueueFillBuffer(_dendrites[d].perms,  static_cast<cl_char>(0),    0, sizeof(cl_char)   * _dendrites[d].numS);
	}

	_outputs   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _dendrites[0].numIn); // !!!
	_nPredicts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nLearns   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);  // CHANGE TO NWINNERS
	_nActives  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nBoosts   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numN);

	cs.getQueue().enqueueFillBuffer(_outputs,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _dendrites[0].numIn); // !!!
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nLearns,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives,  static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nBoosts,   static_cast<cl_ushort>(0), 0, sizeof(cl_ushort) * _numN);

	// Initialize Kernels
	_overlapDendrites = cl::Kernel(cp.getProgram(), "overlapDendrites");
	_learnSynapses    = cl::Kernel(cp.getProgram(), "learnSynapses");
	_predictNeurons   = cl::Kernel(cp.getProgram(), "predictNeurons");
	_decodeNeurons    = cl::Kernel(cp.getProgram(), "decodeNeurons");
}

void Region::encode(ComputeSystem &cs, bool learn)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nLearns, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	for (unsigned int d = 0; d < _numDpN; d++)
	{
		_overlapDendrites.setArg(0, _nOverlaps);
		_overlapDendrites.setArg(1, _dendrites[d].addrs);
		_overlapDendrites.setArg(2, _dendrites[d].perms);
		_overlapDendrites.setArg(3, _dendrites[d].inputs);
		_overlapDendrites.setArg(4, _dendrites[d].numSpD);
		_overlapDendrites.setArg(5, _dendrites[d].dThresh);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}

	// Neuron Activation and Inhibition
	char nLearnsArr[_numN];
	char nActivesArr[_numN];
	char nOverlapsArr[_numN];
	unsigned short nBoostsArr[_numN];

	cs.getQueue().enqueueReadBuffer(_nLearns, CL_TRUE, 0, sizeof(cl_char) * _numN, &nLearnsArr, NULL);
	cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, &nActivesArr, NULL);
	cs.getQueue().enqueueReadBuffer(_nOverlaps, CL_TRUE, 0, sizeof(cl_char) * _numN, &nOverlapsArr, NULL);
	cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, &nBoostsArr, NULL);

	unsigned int an = 0;

	for (unsigned int n = 0; n < _numN; n++)
	{
		nBoostsArr[n]++;

		if (nOverlapsArr[n] >= _nActThresh)
		{
			nActivesArr[n] = 1;
		}

		if (nOverlapsArr[n] >= 2 && an <= _numAN) // !!!
		{
				nLearnsArr[n] = 1;
				nBoostsArr[n] = 0;
				an++;
		}
	}

	for (an; an < _numAN; an++)
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

		nLearnsArr[maxIndex] = 1;
		nBoostsArr[maxIndex] = 0;
	}

	cs.getQueue().enqueueWriteBuffer(_nLearns, CL_TRUE, 0, sizeof(cl_char) * _numN, nLearnsArr);
	cs.getQueue().enqueueWriteBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, nActivesArr);
	cs.getQueue().enqueueWriteBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, nBoostsArr);

	// Learning
	if (learn)
	{
		for (unsigned int d = 0; d < _numDpN; d++)
		{
			_learnSynapses.setArg(0, _dendrites[d].addrs);
			_learnSynapses.setArg(1, _dendrites[d].perms);
			_learnSynapses.setArg(2, _nLearns);
			_learnSynapses.setArg(3, _dendrites[d].inputs);
			_learnSynapses.setArg(4, _dendrites[d].numSpD);
			_learnSynapses.setArg(5, _dendrites[d].numIn);
			_learnSynapses.setArg(6, _sPermMax);

			_range = cl::NDRange(_numN);
			cs.getQueue().enqueueNDRangeKernel(_learnSynapses, cl::NullRange, _range);
			cs.getQueue().finish();
		}
	}
}

void Region::predict(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Overlap Dendrite 0
	_overlapDendrites.setArg(0, _nOverlaps);
	_overlapDendrites.setArg(1, _dendrites[1].addrs);
	_overlapDendrites.setArg(2, _dendrites[1].perms);
	_overlapDendrites.setArg(3, _nActives);
	_overlapDendrites.setArg(4, _dendrites[1].numSpD);
	_overlapDendrites.setArg(5, _dendrites[1].dThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
	cs.getQueue().finish();

	// Predict Neurons
	_predictNeurons.setArg(0, _nPredicts);
	_predictNeurons.setArg(1, _nOverlaps);
	_predictNeurons.setArg(2, _nPreThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_predictNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::decode(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_outputs, static_cast<cl_char>(0), 0, sizeof(cl_char) * _dendrites[0].numIn);

	_decodeNeurons.setArg(0, _outputs);
	_decodeNeurons.setArg(1, _nPredicts);
	_decodeNeurons.setArg(2, _dendrites[0].addrs);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_decodeNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::print(ComputeSystem& cs)
{
	std::vector<char> nPredictsVec(_numN);
	std::vector<char> nLearnsVec(_numN);
	std::vector<char> nActivesVec(_numN);
	std::vector<unsigned short> nBoostsVec(_numN);
	std::vector<char> nOverlapsVec(_numN);
	std::vector<unsigned short> sAddrs0Vec(_dendrites[0].numS);
	std::vector<char> sPerms0Vec(_dendrites[0].numS);
	std::vector<unsigned short> sAddrs1Vec(_dendrites[1].numS);
	std::vector<char> sPerms1Vec(_dendrites[1].numS);
	std::vector<char> inputs0Vec(_dendrites[0].numIn);
	std::vector<char> inputs1Vec(_dendrites[1].numIn);
	std::vector<char> outputsVec(_dendrites[0].numIn);

	cs.getQueue().enqueueReadBuffer(_nPredicts, CL_TRUE, 0, sizeof(char) * _numN, nPredictsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nLearns,   CL_TRUE, 0, sizeof(char) * _numN, nLearnsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nActives,  CL_TRUE, 0, sizeof(char) * _numN, nActivesVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nBoosts,   CL_TRUE, 0, sizeof(unsigned short) * _numN, nBoostsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nOverlaps, CL_TRUE, 0, sizeof(char) * _numN, nOverlapsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dendrites[0].addrs, CL_TRUE, 0, sizeof(unsigned short) * _dendrites[0].numS, sAddrs0Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dendrites[0].perms, CL_TRUE, 0, sizeof(char)           * _dendrites[0].numS, sPerms0Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dendrites[1].addrs, CL_TRUE, 0, sizeof(unsigned short) * _dendrites[1].numS, sAddrs1Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dendrites[1].perms, CL_TRUE, 0, sizeof(char)           * _dendrites[1].numS, sPerms1Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dendrites[0].inputs, CL_TRUE, 0, sizeof(char)          * _dendrites[0].numIn, inputs0Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dendrites[1].inputs, CL_TRUE, 0, sizeof(char)          * _dendrites[1].numIn, inputs1Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_outputs, CL_TRUE, 0, sizeof(char) * _dendrites[0].numIn, outputsVec.data(), NULL);

	printf("\nNPRED ");
	for(int i = 0; i < _numN; i++){if (nPredictsVec[i] < 10){printf("0%i ", nPredictsVec[i]);}else{printf("%i ", nPredictsVec[i]);}}

	printf("\nNLEAR ");
	for(int i = 0; i < _numN; i++){if (nLearnsVec[i] < 10){printf("0%i ", nLearnsVec[i]);}else{printf("%i ", nLearnsVec[i]);}}

	printf("\nNACTI ");
	for(int i = 0; i < _numN; i++){if (nActivesVec[i] < 10){printf("0%i ", nActivesVec[i]);}else{printf("%i ", nActivesVec[i]);}}

	printf("\nNBOOS ");
	for(int i = 0; i < _numN; i++){if (nBoostsVec[i] < 10){printf("0%i ", nBoostsVec[i]);}else{printf("%i ", nBoostsVec[i]);}}

	printf("\nNOVER ");
	for(int i = 0; i < _numN; i++){if (nOverlapsVec[i] < 10){printf("0%i ", nOverlapsVec[i]);}else{printf("%i ", nOverlapsVec[i]);}}

	printf("\nSADR0 ");
	for(int i = 0; i < _dendrites[0].numS; i++){if (sAddrs0Vec[i] < 10){printf("0%i ", sAddrs0Vec[i]);}else{printf("%i ", sAddrs0Vec[i]);}}

	printf("\nSPER0 ");
	for(int i = 0; i < _dendrites[0].numS; i++){if (sPerms0Vec[i] < 10){printf("0%i ", sPerms0Vec[i]);}else{printf("%i ", sPerms0Vec[i]);}}

	printf("\nSADR1 ");
	for(int i = 0; i < _dendrites[1].numS; i++){if (sAddrs1Vec[i] < 10){printf("0%i ", sAddrs1Vec[i]);}else{printf("%i ", sAddrs1Vec[i]);}}

	printf("\nSPER1 ");
	for(int i = 0; i < _dendrites[1].numS; i++){if (sPerms1Vec[i] < 10){printf("0%i ", sPerms1Vec[i]);}else{printf("%i ", sPerms1Vec[i]);}}

//	printf("\nINPU0 ");
//	for(int i = 0; i < _dendrites[0].numIn; i++){if (inputs0Vec[i] < 10){printf("0%i ", inputs0Vec[i]);}else{printf("%i ", inputs0Vec[i]);}}

//	printf("\nINPU1 ");
//	for(int i = 0; i < _dendrites[1].numIn; i++){if (inputs1Vec[i] < 10){printf("0%i ", inputs1Vec[i]);}else{printf("%i ", inputs1Vec[i]);}}

//	printf("\nOUTPU ");
//	for(int i = 0; i < _dendrites[0].numIn; i++){if (outputsVec[i] < 10){printf("0%i ", outputsVec[i]);}else{printf("%i ", outputsVec[i]);}}

	printf("\n");
}


void Region::setInputs(ComputeSystem& cs, unsigned int d, std::vector<char> vec)
{
	cs.getQueue().enqueueWriteBuffer(_dendrites[d].inputs, CL_TRUE, 0, sizeof(cl_char) * _dendrites[d].numIn, vec.data());
}

void Region::copyInputsToInputs(ComputeSystem& cs, unsigned int dFrom, unsigned int dTo)
{
	cs.getQueue().enqueueCopyBuffer(_dendrites[dFrom].inputs, _dendrites[dTo].inputs, 0, 0, sizeof(cl_char) * _dendrites[dTo].numS);
}

void Region::copyNeuronsToInputs(ComputeSystem& cs, unsigned int d)
{
	cs.getQueue().enqueueCopyBuffer(_nLearns, _dendrites[d].inputs, 0, 0, sizeof(cl_char) * _numN); // !!!
}

std::vector<char> Region::getInputs(ComputeSystem &cs, unsigned int d)
{
	std::vector<char> vec(_dendrites[d].numIn);
	cs.getQueue().enqueueReadBuffer(_dendrites[d].inputs, CL_TRUE, 0, sizeof(cl_char) * _dendrites[d].numIn, vec.data(), NULL);
	return vec;
}

std::vector<char> Region::getOutputs(ComputeSystem &cs)
{
	std::vector<char> vec(_dendrites[0].numIn);
	cs.getQueue().enqueueReadBuffer(_outputs, CL_TRUE, 0, sizeof(cl_char) * _dendrites[0].numIn, vec.data(), NULL);
	return vec;
}
