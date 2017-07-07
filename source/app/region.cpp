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
	unsigned int numIn0,
	unsigned int numIn1,
	unsigned int numN,
	unsigned int numSpD0,
	unsigned int numSpD1
)
{
	_rng = rng;

	// Initialize Variables
	_numIn0   = static_cast<cl_uint>(numIn0);
	_numIn1   = static_cast<cl_uint>(numIn1);
	_numN     = static_cast<cl_uint>(numN);
	_numAN    = static_cast<cl_uint>(_numN * 0.02);
	_numSpD0  = static_cast<cl_uint>(numSpD0);
	_numSpD1  = static_cast<cl_uint>(numSpD1);
	_numSpN0  = static_cast<cl_uint>(numSpD0 * _numN);
	_numSpN1  = static_cast<cl_uint>(numSpD1 * _numN);
	_sPermMax = static_cast<cl_uint>(99);
	_nThresh  = static_cast<cl_uint>(2); // !!!
	_dThresh0 = static_cast<cl_uint>(1); // !!!
	_dThresh1 = static_cast<cl_uint>(1); // !!!

	if (_numAN == 0)
		_numAN = 1;

	// Initialize Buffers
	_inputs0 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numIn0);
	_inputs1 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numIn1);
	_outputs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numIn0); // !!!
	_nActives = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nPredicts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nBoosts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numN);
	_sAddrs0 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numSpN0);
	_sAddrs1 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numSpN1);
	_sPerms0 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numSpN0);
	_sPerms1 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numSpN1);

	// Fill Buffers
	cs.getQueue().enqueueFillBuffer(_inputs0,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _numIn0);
	cs.getQueue().enqueueFillBuffer(_inputs1,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _numIn1);
	cs.getQueue().enqueueFillBuffer(_outputs,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _numIn0); // !!!
	cs.getQueue().enqueueFillBuffer(_nActives,  static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nBoosts,   static_cast<cl_ushort>(0), 0, sizeof(cl_ushort) * _numN);
	cs.getQueue().enqueueFillBuffer(_sAddrs0,   static_cast<cl_ushort>(99), 0, sizeof(cl_ushort) * _numSpN0);
	cs.getQueue().enqueueFillBuffer(_sAddrs1,   static_cast<cl_ushort>(99), 0, sizeof(cl_ushort) * _numSpN1);
	cs.getQueue().enqueueFillBuffer(_sPerms0,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _numSpN0);
	cs.getQueue().enqueueFillBuffer(_sPerms1,   static_cast<cl_char>(0), 0, sizeof(cl_char) * _numSpN1);

	// Initialize Kernels
	_overlapDendrites = cl::Kernel(cp.getProgram(), "overlapDendrites");
	_learnSynapses = cl::Kernel(cp.getProgram(), "learnSynapses");
	_predictNeurons = cl::Kernel(cp.getProgram(), "predictNeurons");
}


void Region::activate(ComputeSystem &cs, bool learn)
{

//	printf("TEST");

	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Overlap Dendrite 0
	_overlapDendrites.setArg(0, _nOverlaps);
	_overlapDendrites.setArg(1, _sAddrs0);
	_overlapDendrites.setArg(2, _sPerms0);
	_overlapDendrites.setArg(3, _inputs0);
	_overlapDendrites.setArg(4, _numSpD0);
	_overlapDendrites.setArg(5, _dThresh0);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
	cs.getQueue().finish();

	// Overlap Dendrite 1
	_overlapDendrites.setArg(0, _nOverlaps);
	_overlapDendrites.setArg(1, _sAddrs1);
	_overlapDendrites.setArg(2, _sPerms1);
	_overlapDendrites.setArg(3, _inputs1);
	_overlapDendrites.setArg(4, _numSpD1);
	_overlapDendrites.setArg(5, _dThresh1);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
	cs.getQueue().finish();

	// Neuron Activation and Inhibition
	char nActivesArr[_numN];
	char nOverlapsArr[_numN];
	unsigned short nBoostsArr[_numN];

	cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, &nActivesArr, NULL);
	cs.getQueue().enqueueReadBuffer(_nOverlaps, CL_TRUE, 0, sizeof(cl_char) * _numN, &nOverlapsArr, NULL);
	cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, &nBoostsArr, NULL);

	unsigned int an = 0;

	for (unsigned int n = 0; n < _numN; n++)
	{
		nBoostsArr[n]++;

		if (nOverlapsArr[n] >= _nThresh && an <= _numAN)
		{
//			printf("TEST");
			nActivesArr[n] = 1;
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

		nActivesArr[maxIndex] = 1;
		nBoostsArr[maxIndex] = 0;
	}

	cs.getQueue().enqueueWriteBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, nActivesArr);
	cs.getQueue().enqueueWriteBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, nBoostsArr);

	// Learning
	if (learn)
	{
		_learnSynapses.setArg(0, _sAddrs0);
		_learnSynapses.setArg(1, _sPerms0);
		_learnSynapses.setArg(2, _nActives);
		_learnSynapses.setArg(3, _inputs0);
		_learnSynapses.setArg(4, _numSpD0);
		_learnSynapses.setArg(5, _numIn0);
		_learnSynapses.setArg(6, _sPermMax);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_learnSynapses, cl::NullRange, _range);
		cs.getQueue().finish();

		_learnSynapses.setArg(0, _sAddrs1);
		_learnSynapses.setArg(1, _sPerms1);
		_learnSynapses.setArg(2, _nActives);
		_learnSynapses.setArg(3, _inputs1);
		_learnSynapses.setArg(4, _numSpD1);
		_learnSynapses.setArg(5, _numIn1);
		_learnSynapses.setArg(6, _sPermMax);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_learnSynapses, cl::NullRange, _range);
		cs.getQueue().finish();
	}
}

void Region::predict(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Overlap Dendrite 0
	_overlapDendrites.setArg(0, _nOverlaps);
	_overlapDendrites.setArg(1, _sAddrs0);
	_overlapDendrites.setArg(2, _sPerms0);
	_overlapDendrites.setArg(3, _inputs0);
	_overlapDendrites.setArg(4, _numSpD0);
	_overlapDendrites.setArg(5, _dThresh0);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
	cs.getQueue().finish();

	// Predict Neurons
	_predictNeurons.setArg(0, _nPredicts);
	_predictNeurons.setArg(1, _nOverlaps);
	_predictNeurons.setArg(2, static_cast<cl_uint>(1));

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_predictNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::print(ComputeSystem& cs)
{
	std::vector<char> nPredictsVec(_numN);
	std::vector<char> nActivesVec(_numN);
	std::vector<unsigned short> nBoostsVec(_numN);
	std::vector<char> nOverlapsVec(_numN);
	std::vector<unsigned short> sAddrs0Vec(_numSpN0);
	std::vector<char> sPerms0Vec(_numSpN0);
	std::vector<unsigned short> sAddrs1Vec(_numSpN1);
	std::vector<char> sPerms1Vec(_numSpN1);
	std::vector<char> inputs0Vec(_numIn0);
	std::vector<char> inputs1Vec(_numIn1);
	std::vector<char> outputsVec(_numIn0);

	cs.getQueue().enqueueReadBuffer(_nPredicts, CL_TRUE, 0, sizeof(char) * _numN, nPredictsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(char) * _numN, nActivesVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, sizeof(unsigned short) * _numN, nBoostsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nOverlaps, CL_TRUE, 0, sizeof(char) * _numN, nOverlapsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_sAddrs0, CL_TRUE, 0, sizeof(unsigned short) * _numSpN0, sAddrs0Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_sPerms0, CL_TRUE, 0, sizeof(char) * _numSpN0, sPerms0Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_sAddrs1, CL_TRUE, 0, sizeof(unsigned short) * _numSpN1, sAddrs1Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_sPerms1, CL_TRUE, 0, sizeof(char) * _numSpN1, sPerms1Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_inputs0, CL_TRUE, 0, sizeof(char) * _numIn0, inputs0Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_inputs1, CL_TRUE, 0, sizeof(char) * _numIn1, inputs1Vec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_outputs, CL_TRUE, 0, sizeof(char) * _numIn0, outputsVec.data(), NULL);

	printf("\nNPRED ");
	for(int i = 0; i < _numN; i++){if (nPredictsVec[i] < 10){printf("0%i ", nPredictsVec[i]);}else{printf("%i ", nPredictsVec[i]);}}

	printf("\nNACTI ");
	for(int i = 0; i < _numN; i++){if (nActivesVec[i] < 10){printf("0%i ", nActivesVec[i]);}else{printf("%i ", nActivesVec[i]);}}

	printf("\nNBOOS ");
	for(int i = 0; i < _numN; i++){if (nBoostsVec[i] < 10){printf("0%i ", nBoostsVec[i]);}else{printf("%i ", nBoostsVec[i]);}}

	printf("\nNOVER ");
	for(int i = 0; i < _numN; i++){if (nOverlapsVec[i] < 10){printf("0%i ", nOverlapsVec[i]);}else{printf("%i ", nOverlapsVec[i]);}}

	printf("\nSADR0 ");
	for(int i = 0; i < _numSpN0; i++){if (sAddrs0Vec[i] < 10){printf("0%i ", sAddrs0Vec[i]);}else{printf("%i ", sAddrs0Vec[i]);}}

	printf("\nSPER0 ");
	for(int i = 0; i < _numSpN0; i++){if (sPerms0Vec[i] < 10){printf("0%i ", sPerms0Vec[i]);}else{printf("%i ", sPerms0Vec[i]);}}

	printf("\nSADR1 ");
	for(int i = 0; i < _numSpN1; i++){if (sAddrs1Vec[i] < 10){printf("0%i ", sAddrs1Vec[i]);}else{printf("%i ", sAddrs1Vec[i]);}}

	printf("\nSPER1 ");
	for(int i = 0; i < _numSpN1; i++){if (sPerms1Vec[i] < 10){printf("0%i ", sPerms1Vec[i]);}else{printf("%i ", sPerms1Vec[i]);}}

	printf("\nINPU0 ");
	for(int i = 0; i < _numIn0; i++){if (inputs0Vec[i] < 10){printf("0%i ", inputs0Vec[i]);}else{printf("%i ", inputs0Vec[i]);}}

	printf("\nINPU1 ");
	for(int i = 0; i < _numIn1; i++){if (inputs1Vec[i] < 10){printf("0%i ", inputs1Vec[i]);}else{printf("%i ", inputs1Vec[i]);}}

	printf("\nOUTPU ");
	for(int i = 0; i < _numIn0; i++){if (outputsVec[i] < 10){printf("0%i ", outputsVec[i]);}else{printf("%i ", outputsVec[i]);}}

	printf("\n");
}

void Region::setInputs0(ComputeSystem& cs, std::vector<char> vec)
{
	cs.getQueue().enqueueWriteBuffer(_inputs0, CL_TRUE, 0, sizeof(cl_char) * _numIn0, vec.data());
}

void Region::setInputs1(ComputeSystem& cs, std::vector<char> vec)
{
	cs.getQueue().enqueueWriteBuffer(_inputs1, CL_TRUE, 0, sizeof(cl_char) * _numIn1, vec.data());
}

std::vector<char> Region::getInputs0(ComputeSystem &cs)
{
	std::vector<char> vec(_numIn0);
	cs.getQueue().enqueueReadBuffer(_inputs0, CL_TRUE, 0, sizeof(cl_char) * _numIn0, vec.data(), NULL);
	return vec;
}

std::vector<char> Region::getInputs1(ComputeSystem &cs)
{
	std::vector<char> vec(_numIn1);
	cs.getQueue().enqueueReadBuffer(_inputs1, CL_TRUE, 0, sizeof(cl_char) * _numIn1, vec.data(), NULL);
	return vec;
}

std::vector<char> Region::getOutputs(ComputeSystem &cs)
{
	std::vector<char> vec(_numIn0);
	cs.getQueue().enqueueReadBuffer(_outputs, CL_TRUE, 0, sizeof(cl_char) * _numIn0, vec.data(), NULL);
	return vec;
}
