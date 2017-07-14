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
	std::vector<unsigned int> numVperP,
	std::vector<unsigned int> numSperD
)
{
	_rng = rng;

	// Initialize Variables
	_numN = static_cast<cl_uint>(numN);
	_numW = static_cast<cl_uint>(_numN * 0.02);
	_numP = static_cast<cl_uint>(numVperP.size());
	_numDperN = static_cast<cl_uint>(numSperD.size());

	_nActThresh = static_cast<cl_uint>(_numDperN);
	_nPreThresh = static_cast<cl_uint>(1); // !!!
	_sPermMax = static_cast<cl_uint>(99);
	_sAddrMax = static_cast<cl_uint>(999); //65535

	if (_numW == 0)
		_numW = 1;

	_patterns.resize(_numP);

	for (unsigned int p = 0; p < _numP; p++)
	{
		_patterns[p].numV = static_cast<cl_uint>(numVperP[p]);

		_patterns[p].values = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _patterns[p].numV);

		cs.getQueue().enqueueFillBuffer(_patterns[p].values, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[p].numV);
	}

	_dendrites.resize(_numDperN);

	for (unsigned int d = 0; d < _numDperN; d++)
	{
		_dendrites[d].numSperD = static_cast<cl_uint>(numSperD[d]);
		_dendrites[d].numS     = static_cast<cl_uint>(numSperD[d] * _numN);
		_dendrites[d].dThresh  = static_cast<cl_uint>(1);

		_dendrites[d].sAddrs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _dendrites[d].numS);
		_dendrites[d].sPerms = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char)   * _dendrites[d].numS);

		cs.getQueue().enqueueFillBuffer(_dendrites[d].sAddrs, static_cast<cl_ushort>(_sAddrMax), 0, sizeof(cl_ushort) * _dendrites[d].numS);  // do I even need _sAddrMax anymore?
		cs.getQueue().enqueueFillBuffer(_dendrites[d].sPerms, static_cast<cl_char>(0),           0, sizeof(cl_char)   * _dendrites[d].numS);
	}

	_inhibitFlag = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char));

	_nBoosts   = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numN);
	_nPredicts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nWinners  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nActives  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);

	_preOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_DOVE0 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_DOVE1 = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);

	_goodPredictions = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _patterns[3].numV);
	_badPredictions  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _patterns[3].numV);


	cs.getQueue().enqueueFillBuffer(_inhibitFlag, static_cast<cl_char>(0), 0, sizeof(cl_char));

	cs.getQueue().enqueueFillBuffer(_nBoosts,   static_cast<cl_ushort>(0), 0, sizeof(cl_ushort) * _numN);
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nWinners,  static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives,  static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	cs.getQueue().enqueueFillBuffer(_preOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	cs.getQueue().enqueueFillBuffer(_goodPredictions, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[3].numV);
	cs.getQueue().enqueueFillBuffer(_badPredictions, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[3].numV);

	// Initialize Kernels
	_overlapDendrites = cl::Kernel(cp.getProgram(), "overlapDendrites");
	_learnSynapses    = cl::Kernel(cp.getProgram(), "learnSynapses");
	_activateNeurons  = cl::Kernel(cp.getProgram(), "activateNeurons");
	_predictNeurons   = cl::Kernel(cp.getProgram(), "predictNeurons");
	_decodeNeurons    = cl::Kernel(cp.getProgram(), "decodeNeurons");
}

void Region::encode(ComputeSystem &cs, std::vector<unsigned int> pEncode)
{
	cs.getQueue().enqueueFillBuffer(_nWinners, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	cs.getQueue().enqueueFillBuffer(_DOVE0, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_DOVE1, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	unsigned int p = 0;

	for (unsigned int d = 0; d < _numDperN; d++)
	{
		_overlapDendrites.setArg(0, _nOverlaps);
		_overlapDendrites.setArg(1, _dendrites[d].sAddrs);
		_overlapDendrites.setArg(2, _dendrites[d].sPerms);
		_overlapDendrites.setArg(3, _patterns[pEncode[p]].values);
		_overlapDendrites.setArg(4, _dendrites[d].numSperD);
		_overlapDendrites.setArg(5, _dendrites[d].dThresh);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
		cs.getQueue().finish();

		/*
		if (p == 0)
		{
			_overlapDendrites.setArg(0, _DOVE0);
			_overlapDendrites.setArg(1, _dendrites[d].sAddrs);
			_overlapDendrites.setArg(2, _dendrites[d].sPerms);
			_overlapDendrites.setArg(3, _patterns[pEncode[p]].values);
			_overlapDendrites.setArg(4, _dendrites[d].numSperD);
			_overlapDendrites.setArg(5, _dendrites[d].dThresh);

			_range = cl::NDRange(_numN);
			cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
			cs.getQueue().finish();
		}
		*/

		/*
		if (p == 1)
		{
			_overlapDendrites.setArg(0, _DOVE1);
			_overlapDendrites.setArg(1, _dendrites[d].sAddrs);
			_overlapDendrites.setArg(2, _dendrites[d].sPerms);
			_overlapDendrites.setArg(3, _patterns[pEncode[p]].values);
			_overlapDendrites.setArg(4, _dendrites[d].numSperD);
			_overlapDendrites.setArg(5, _dendrites[d].dThresh);

			_range = cl::NDRange(_numN);
			cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
			cs.getQueue().finish();
		}
		*/

		p++;
	}

	cs.getQueue().enqueueFillBuffer(_inhibitFlag, static_cast<cl_char>(0), 0, sizeof(cl_char));

	// Neuron Activation
	_activateNeurons.setArg(0, _nBoosts);
	_activateNeurons.setArg(1, _nWinners);
	_activateNeurons.setArg(2, _nActives);
	_activateNeurons.setArg(3, _nOverlaps);
	_activateNeurons.setArg(4, _inhibitFlag);
	_activateNeurons.setArg(5, _nActThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_activateNeurons, cl::NullRange, _range);
	cs.getQueue().finish();

	/*
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Neuron Activation
	_activateNeurons.setArg(0, _nBoosts);
	_activateNeurons.setArg(1, _nWinners);
	_activateNeurons.setArg(2, _nActives);
	_activateNeurons.setArg(3, _DOVE0);
	_activateNeurons.setArg(4, _inhibitFlag);
	_activateNeurons.setArg(5, _nActThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_activateNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
	*/

	char inhibitFlagArr[1];

	cs.getQueue().enqueueReadBuffer(_inhibitFlag, CL_TRUE, 0, sizeof(cl_char), &inhibitFlagArr, NULL);

	// Neuron Inhibition
	if (inhibitFlagArr[0] > 0)
	{
		cs.getQueue().enqueueCopyBuffer(_nWinners, _nActives, 0, 0, sizeof(cl_char) * _numN);
	}
	else
	{
		unsigned short nBoostsArr[_numN];

		char nWinnersArr[_numN];
		char nActivesArr[_numN];

		cs.getQueue().enqueueReadBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, &nBoostsArr, NULL);
		cs.getQueue().enqueueReadBuffer(_nWinners, CL_TRUE, 0, sizeof(cl_char) * _numN, &nWinnersArr, NULL);
		cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, &nActivesArr, NULL);

		unsigned int w = 0;

		for (w; w < _numW; w++)
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
			nWinnersArr[maxIndex] = 1;
			nBoostsArr[maxIndex] = 0;
		}

		cs.getQueue().enqueueWriteBuffer(_nBoosts, CL_TRUE, 0, sizeof(cl_ushort) * _numN, nBoostsArr);
		cs.getQueue().enqueueWriteBuffer(_nWinners, CL_TRUE, 0, sizeof(cl_char) * _numN, nWinnersArr);
		cs.getQueue().enqueueWriteBuffer(_nActives, CL_TRUE, 0, sizeof(cl_char) * _numN, nActivesArr);
	}
}

void Region::learn(ComputeSystem& cs, std::vector<unsigned int> pLearn)
{
	unsigned int p = 0;

	for (unsigned int d = 0; d < _numDperN; d++)
	{
		_learnSynapses.setArg(0, _dendrites[d].sAddrs);
		_learnSynapses.setArg(1, _dendrites[d].sPerms);
		_learnSynapses.setArg(2, _nWinners);
		_learnSynapses.setArg(3, _patterns[pLearn[p]].values);
		_learnSynapses.setArg(4, _dendrites[d].numSperD);
		_learnSynapses.setArg(5, _patterns[pLearn[p]].numV);
		_learnSynapses.setArg(6, _sPermMax);

		_range = cl::NDRange(_numN);
		cs.getQueue().enqueueNDRangeKernel(_learnSynapses, cl::NullRange, _range);
		cs.getQueue().finish();

		p++;
	}
}

// LOOK OVER
void Region::predict(ComputeSystem& cs)
{
	cs.getQueue().enqueueFillBuffer(_preOverlaps, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN); //!!!!
	cs.getQueue().enqueueFillBuffer(_nPredicts, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Overlap Dendrite 0
	_overlapDendrites.setArg(0, _preOverlaps); //!!!!!
	_overlapDendrites.setArg(1, _dendrites[1].sAddrs);
	_overlapDendrites.setArg(2, _dendrites[1].sPerms);
	_overlapDendrites.setArg(3, _nActives);
	_overlapDendrites.setArg(4, _dendrites[1].numSperD);
	_overlapDendrites.setArg(5, _dendrites[1].dThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_overlapDendrites, cl::NullRange, _range);
	cs.getQueue().finish();

	// Predict Neurons
	_predictNeurons.setArg(0, _nPredicts);
	_predictNeurons.setArg(1, _preOverlaps); //!!!!!!!!
	_predictNeurons.setArg(2, _nPreThresh);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_predictNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

// LOOK OVER
void Region::decode(ComputeSystem& cs)
{
	char pOutputArr[_patterns[3].numV];
	char pInputArr[_patterns[0].numV];

	char goodPredictionArr[_patterns[3].numV];
	char badPredictionArr[_patterns[3].numV];

	cs.getQueue().enqueueReadBuffer(_patterns[3].values, CL_TRUE, 0, sizeof(cl_char) * _patterns[3].numV, &pOutputArr, NULL);
	cs.getQueue().enqueueReadBuffer(_patterns[0].values, CL_TRUE, 0, sizeof(cl_char) * _patterns[0].numV, &pInputArr, NULL);

	for (unsigned int i = 0; i < _patterns[3].numV; i++)
	{
		goodPredictionArr[i] = 0;
		badPredictionArr[i] = 0;

		if (pInputArr[i] == 1 && pOutputArr[i] == 1)
			goodPredictionArr[i] = 1;
		if (pInputArr[i] == 0 && pOutputArr[i] == 1)
			badPredictionArr[i] = 1;
		if (pInputArr[i] == 1 && pOutputArr[i] == 0)
			badPredictionArr[i] = 1;
	}

	cs.getQueue().enqueueWriteBuffer(_goodPredictions, CL_TRUE, 0, sizeof(cl_char) * _patterns[3].numV, goodPredictionArr);
	cs.getQueue().enqueueWriteBuffer(_badPredictions, CL_TRUE, 0, sizeof(cl_char) * _patterns[3].numV, badPredictionArr);

	cs.getQueue().enqueueFillBuffer(_patterns[3].values, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[3].numV);

	_decodeNeurons.setArg(0, _patterns[3].values);
	_decodeNeurons.setArg(1, _nPredicts);
	_decodeNeurons.setArg(2, _dendrites[0].sAddrs);

	_range = cl::NDRange(_numN);
	cs.getQueue().enqueueNDRangeKernel(_decodeNeurons, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::print(ComputeSystem& cs)
{
	printf("\nNNUMB ");
	for (int i = 0; i < _numN; i++)
	{
		if (i < 10)
			printf("  %i ", i);
		else
			printf(" %i ", i);
	}


	printf("\nNBOOS ");
	std::vector<unsigned short> nBoostsVec(_numN);
	cs.getQueue().enqueueReadBuffer(_nBoosts,   CL_TRUE, 0, sizeof(unsigned short) * _numN, nBoostsVec.data(), NULL);
	for(int i = 0; i < _numN; i++)
	{
		if (nBoostsVec[i] < 10)
			printf("  %i ", nBoostsVec[i]);
		else if (nBoostsVec[i] < 100)
			printf(" %i ", nBoostsVec[i]);
		else
			printf("%i ", nBoostsVec[i]);
	}

	printf("\nNPRED ");
	std::vector<char> nPredictsVec(_numN);
	cs.getQueue().enqueueReadBuffer(_nPredicts, CL_TRUE, 0, sizeof(char) * _numN, nPredictsVec.data(), NULL);
	for(int i = 0; i < _numN; i++)
	{
		printf("  %i ", nPredictsVec[i]);
	}

	printf("\nPOVER ");
	std::vector<char> preOverlapsVec(_numN);
	cs.getQueue().enqueueReadBuffer(_preOverlaps, CL_TRUE, 0, sizeof(char) * _numN, preOverlapsVec.data(), NULL);
	for(int i = 0; i < _numN; i++)
	{
		printf("  %i ", preOverlapsVec[i]);
	}

	printf("\nNWINN ");
	std::vector<char> nWinnersVec(_numN);
	cs.getQueue().enqueueReadBuffer(_nWinners,  CL_TRUE, 0, sizeof(char) * _numN, nWinnersVec.data(), NULL);
	for(int i = 0; i < _numN; i++)
	{
		printf("  %i ", nWinnersVec[i]);
	}

	printf("\nNACTI ");
	std::vector<char> nActivesVec(_numN);
	cs.getQueue().enqueueReadBuffer(_nActives,  CL_TRUE, 0, sizeof(char) * _numN, nActivesVec.data(), NULL);
	for(int i = 0; i < _numN; i++)
	{
		printf("  %i ", nActivesVec[i]);
	}

	printf("\nNOVER ");
	std::vector<char> nOverlapsVec(_numN);
	cs.getQueue().enqueueReadBuffer(_nOverlaps, CL_TRUE, 0, sizeof(char) * _numN, nOverlapsVec.data(), NULL);
	for(int i = 0; i < _numN; i++)
	{
		printf("  %i ", nOverlapsVec[i]);
	}

//	printf("\nDOVE0 ");
//	std::vector<char> dOverlaps0Vec(_numN);
//	cs.getQueue().enqueueReadBuffer(_DOVE0,     CL_TRUE, 0, sizeof(char) * _numN, dOverlaps0Vec.data(), NULL);
//	for(int i = 0; i < _numN; i++){if (dOverlaps0Vec[i] < 10){printf("0%i ", dOverlaps0Vec[i]);}else{printf("%i ", dOverlaps0Vec[i]);}}

//	printf("\nDOVE1 ");
//	std::vector<char> dOverlaps1Vec(_numN);
//	cs.getQueue().enqueueReadBuffer(_DOVE1,     CL_TRUE, 0, sizeof(char) * _numN, dOverlaps1Vec.data(), NULL);
//	for(int i = 0; i < _numN; i++){if (dOverlaps1Vec[i] < 10){printf("0%i ", dOverlaps1Vec[i]);}else{printf("%i ", dOverlaps1Vec[i]);}}

	printf("\nSADR0 ");
	std::vector<unsigned short> sAddrs0Vec(_dendrites[0].numS);
	cs.getQueue().enqueueReadBuffer(_dendrites[0].sAddrs, CL_TRUE, 0, sizeof(unsigned short) * _dendrites[0].numS, sAddrs0Vec.data(), NULL);
	for(int i = 0; i < _dendrites[0].numS; i++)
	{
		if (sAddrs0Vec[i] < 10)
			printf("  %i ", sAddrs0Vec[i]);
		else if (sAddrs0Vec[i] < 100)
			printf(" %i ", sAddrs0Vec[i]);
		else
			printf("%i ", sAddrs0Vec[i]);
	}

	printf("\nSPER0 ");
	std::vector<char> sPerms0Vec(_dendrites[0].numS);
	cs.getQueue().enqueueReadBuffer(_dendrites[0].sPerms, CL_TRUE, 0, sizeof(char) * _dendrites[0].numS, sPerms0Vec.data(), NULL);
	for(int i = 0; i < _dendrites[0].numS; i++)
	{
		if (sPerms0Vec[i] < 10)
			printf("  %i ", sPerms0Vec[i]);
		else
			printf(" %i ", sPerms0Vec[i]);
	}

	printf("\nINPU0 ");
	std::vector<char> pattern0Vec(_patterns[0].numV);
	cs.getQueue().enqueueReadBuffer(_patterns[0].values, CL_TRUE, 0, sizeof(char) * _patterns[0].numV, pattern0Vec.data(), NULL);
	for(int i = 0; i < _patterns[0].numV; i++)
	{
		if (pattern0Vec[i] > 0)
		{
			if (pattern0Vec[i] < 10)
				printf("  %i ", i);
			else if (pattern0Vec[i] < 100)
				printf(" %i ", i);
			else
				printf("%i ", i);
		}
	}

	printf("\nSADR1 ");
	std::vector<unsigned short> sAddrs1Vec(_dendrites[1].numS);
	cs.getQueue().enqueueReadBuffer(_dendrites[1].sAddrs, CL_TRUE, 0, sizeof(unsigned short) * _dendrites[1].numS, sAddrs1Vec.data(), NULL);
	for(int i = 0; i < _dendrites[1].numS; i++)
	{
		if (sAddrs1Vec[i] < 10)
			printf("  %i ", sAddrs1Vec[i]);
		else if (sAddrs1Vec[i] < 100)
			printf(" %i ", sAddrs1Vec[i]);
		else
			printf("%i ", sAddrs1Vec[i]);
	}

	printf("\nSPER1 ");
	std::vector<char> sPerms1Vec(_dendrites[1].numS);
	cs.getQueue().enqueueReadBuffer(_dendrites[1].sPerms, CL_TRUE, 0, sizeof(char)           * _dendrites[1].numS, sPerms1Vec.data(), NULL);
	for(int i = 0; i < _dendrites[1].numS; i++)
	{
		if (sPerms1Vec[i] < 10)
			printf("  %i ", sPerms1Vec[i]);
		else
			printf(" %i ", sPerms1Vec[i]);
		}

	printf("\nINPU1 ");
	std::vector<char> pattern1Vec(_patterns[1].numV);
	cs.getQueue().enqueueReadBuffer(_patterns[1].values, CL_TRUE, 0, sizeof(char) * _patterns[1].numV, pattern1Vec.data(), NULL);
	for(int i = 0; i < _patterns[1].numV; i++)
	{
		printf("  %i ", pattern1Vec[i]);
	}

	printf("\n");
}


void Region::setPattern(ComputeSystem& cs, unsigned int p, std::vector<char> vec)
{
	cs.getQueue().enqueueWriteBuffer(_patterns[p].values, CL_TRUE, 0, sizeof(cl_char) * _patterns[p].numV, vec.data());
}

void Region::zeroPattern(ComputeSystem& cs, unsigned int p)
{
	cs.getQueue().enqueueFillBuffer(_patterns[p].values, static_cast<cl_char>(0), 0, sizeof(cl_char) * _patterns[p].numV);
}

/*
void Region::copyInputsToInputs(ComputeSystem& cs, unsigned int dFrom, unsigned int dTo)
{
	cs.getQueue().enqueueCopyBuffer(_dendrites[dFrom].inputs, _dendrites[dTo].inputs, 0, 0, sizeof(cl_char) * _dendrites[dTo].numS);
}
*/

void Region::copyActiveNeuronsToPattern(ComputeSystem& cs, unsigned int p)
{
	cs.getQueue().enqueueCopyBuffer(_nActives, _patterns[p].values, 0, 0, sizeof(cl_char) * _numN);
}

void Region::copyWinnerNeuronsToPattern(ComputeSystem& cs, unsigned int p)
{
	cs.getQueue().enqueueCopyBuffer(_nWinners, _patterns[p].values, 0, 0, sizeof(cl_char) * _numN);
}

std::vector<char> Region::getPattern(ComputeSystem &cs, unsigned int p)
{
	std::vector<char> vec(_patterns[p].numV);
	cs.getQueue().enqueueReadBuffer(_patterns[p].values, CL_TRUE, 0, sizeof(cl_char) * _patterns[p].numV, vec.data(), NULL);
	return vec;
}

std::vector<char> Region::getGoodPrediction(ComputeSystem &cs)
{
	std::vector<char> vec(_patterns[3].numV);
	cs.getQueue().enqueueReadBuffer(_goodPredictions, CL_TRUE, 0, sizeof(cl_char) * _patterns[3].numV, vec.data(), NULL);
	return vec;
}

std::vector<char> Region::getBadPrediction(ComputeSystem &cs)
{
	std::vector<char> vec(_patterns[3].numV);
	cs.getQueue().enqueueReadBuffer(_badPredictions, CL_TRUE, 0, sizeof(cl_char) * _patterns[3].numV, vec.data(), NULL);
	return vec;
}

