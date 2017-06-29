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
	unsigned int numI,
	unsigned int numC,
	unsigned int numNpC,
	unsigned int numDDpN,
	unsigned int numDSpDD
)
{
	_rng = rng;

	// Initialize Variables
	_numI     = static_cast<cl_uint>(numI);
	_numC     = static_cast<cl_uint>(numC);
	_numNpC   = static_cast<cl_uint>(numNpC);
	_numDDpN  = static_cast<cl_uint>(numDDpN);
	_numDSpDD = static_cast<cl_uint>(numDSpDD);
	_numAC    = static_cast<cl_uint>(_numC * 0.02);
	_numN     = static_cast<cl_uint>(_numNpC * numC);
	_numPDpC  = static_cast<cl_uint>(1);
	_numPSpPD = static_cast<cl_uint>(_numI / 2);
	_numPD    = static_cast<cl_uint>(_numPDpC * _numC);
	_numPSpC  = static_cast<cl_uint>(_numPSpPD * _numPDpC);
	_numPS    = static_cast<cl_uint>(_numPSpPD * _numPDpC * _numC);
	_numDDpC  = static_cast<cl_uint>(numDDpN * numNpC);
	_numDD    = static_cast<cl_uint>(_numDDpN * _numN);
	_numDSpN  = static_cast<cl_uint>(_numDSpDD * _numDDpN);
	_numDS    = static_cast<cl_uint>(_numDSpDD * _numDDpN * _numN);
	_sMaxAddr = static_cast<cl_uint>(65535);
	_sMaxPerm = static_cast<cl_uint>(99);
	_dThresh  = static_cast<cl_uint>(1);
	_sThresh  = static_cast<cl_uint>(30);
	_sLearn   = static_cast<cl_uint>(5);

	if (_numAC == 0)
		_numAC = 1;

	// Initialize Buffers
	_inputs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numI);
	_nActives = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nWinners = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nActivesPrev = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_nWinnersPrev = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
	_pdActives = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numPD);
	_ddActives = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numDD);
	_ddLearns  = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numDD);
	_psPerms = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numPS);
	_dsPerms = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numDS);
	_psAddrs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numPS);
	_dsAddrs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numDS);
	_pdOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numPD); // !!!
	_ddOverlaps = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_ushort) * _numDD); // !!!

	// Fill Buffers
	cs.getQueue().enqueueFillBuffer(_inputs, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numI);
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nWinners, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActivesPrev, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nWinnersPrev, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_pdActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numPD);
	cs.getQueue().enqueueFillBuffer(_ddActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numDD);
	cs.getQueue().enqueueFillBuffer(_ddLearns,  static_cast<cl_char>(0), 0, sizeof(cl_char) * _numDD);
	cs.getQueue().enqueueFillBuffer(_psPerms, static_cast<cl_char>(_sThresh), 0, sizeof(cl_char) * _numPS);
	cs.getQueue().enqueueFillBuffer(_dsPerms, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numDS);
	cs.getQueue().enqueueFillBuffer(_psAddrs, static_cast<cl_ushort>(_sMaxAddr), 0, sizeof(cl_ushort) * _numPS);
	cs.getQueue().enqueueFillBuffer(_dsAddrs, static_cast<cl_ushort>(_sMaxAddr), 0, sizeof(cl_ushort) * _numDS);
	cs.getQueue().enqueueFillBuffer(_pdOverlaps, static_cast<cl_ushort>(0), 0, sizeof(cl_ushort) * _numPD);
	cs.getQueue().enqueueFillBuffer(_ddOverlaps, static_cast<cl_ushort>(0), 0, sizeof(cl_ushort) * _numDD);

	// Initialize Kernels
//	_randomizeAddresses = cl::Kernel(cp.getProgram(), "randomizeAddresses");
	_activateDendrites = cl::Kernel(cp.getProgram(), "activateDendrites");
	_learnDendrites = cl::Kernel(cp.getProgram(), "learnDendrites");
	_setNeuronStates = cl::Kernel(cp.getProgram(), "setNeuronStates");
}

/*
void Region::initSynapsesTest(ComputeSystem& cs)
{
	unsigned short psAddrsVec[] = {0, 1, 4, 7, 5, 3, 2, 1};
	char psPermsVec[] = {30, 25, 25, 25, 30, 30, 25, 25};

	cs.getQueue().enqueueWriteBuffer(_psAddrs, CL_TRUE, 0, sizeof(cl_ushort) * _numPS, psAddrsVec);
	cs.getQueue().enqueueWriteBuffer(_psPerms, CL_TRUE, 0, sizeof(cl_char) * _numPS, psPermsVec);
}
*/

/*
void Region::initSynapsesRandom(ComputeSystem& cs)
{
	std::uniform_int_distribution<int> seedDist(0, 999);

	cl_uint2 seed = {(cl_uint)seedDist(_rng), (cl_uint)seedDist(_rng)};

	std::vector<unsigned short> addrsVec(_numI);

	cl::Buffer addrs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_short) * _numI);

	for (int i = 0; i < _numI; i++)
		addrsVec[i] = i;

	cs.getQueue().enqueueWriteBuffer(addrs, CL_TRUE, 0, sizeof(cl_ushort) * _numI, addrsVec.data());

	_randomizeAddresses.setArg(0, _inputs);
	_randomizeAddresses.setArg(1, addrs);
	_randomizeAddresses.setArg(2, _psAddrs);
	_randomizeAddresses.setArg(3, _numI);
	_randomizeAddresses.setArg(4, _numPSpPD);
	_randomizeAddresses.setArg(5, seed);

	_range = cl::NDRange(_numPD);
	cs.getQueue().enqueueNDRangeKernel(_randomizeAddresses, cl::NullRange, _range);
	cs.getQueue().finish();

}
*/

void Region::sp(ComputeSystem &cs, bool learn)
{
	// Overlap
	_activateDendrites.setArg(0, _inputs);
	_activateDendrites.setArg(1, _psAddrs);
	_activateDendrites.setArg(2, _psPerms);
	_activateDendrites.setArg(3, _pdOverlaps);
	_activateDendrites.setArg(4, _pdActives);
	_activateDendrites.setArg(5, _numPSpPD);
	_activateDendrites.setArg(6, _sMaxAddr);
	_activateDendrites.setArg(7, _dThresh);
	_activateDendrites.setArg(8, _sThresh);

	_range = cl::NDRange(_numPD);
	cs.getQueue().enqueueNDRangeKernel(_activateDendrites, cl::NullRange, _range);
	cs.getQueue().finish();

	// Inhibition
//	std::vector<unsigned short> pdOverlapsVec[_numPD];
//	std::vector<char> pdActivesVec[_numPD];

//	cs.getQueue().enqueueReadBuffer(_pdOverlaps, CL_TRUE, 0, sizeof(cl_ushort) * _numPD, &pdOverlapsVec, NULL);



//	cs.getQueue().enqueueWriteBuffer(_pdActives, CL_TRUE, 0, sizeof(cl_char) * _numPD, pdActivesVec);

	// Learning
	if (learn)
	{
		cl::Buffer psPotentials = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numI);
		cs.getQueue().enqueueCopyBuffer(_inputs, psPotentials, 0, 0, sizeof(cl_char) * _numI);

		std::vector<char> inputsVec(_numI);
		unsigned int numActiveInputs = 0;

		cs.getQueue().enqueueReadBuffer(_inputs, CL_TRUE, 0, sizeof(cl_char) * _numI, inputsVec.data(), NULL);
	
		for (unsigned int i = 0; i < _numI; i++)
		{
			if (inputsVec[i] == 1)
				numActiveInputs++;
		}

		_learnDendrites.setArg(0, _inputs);
		_learnDendrites.setArg(1, _psAddrs);
		_learnDendrites.setArg(2, _psPerms);
		_learnDendrites.setArg(3, _pdActives);
		_learnDendrites.setArg(4, psPotentials);
		_learnDendrites.setArg(5, numActiveInputs);
		_learnDendrites.setArg(6, _numI);
		_learnDendrites.setArg(7, _numPSpPD);
		_learnDendrites.setArg(8, _sMaxAddr);
		_learnDendrites.setArg(9, _sMaxPerm);
		_learnDendrites.setArg(10, _sThresh);
		_learnDendrites.setArg(11, _sLearn);

		_range = cl::NDRange(_numPD);
		cs.getQueue().enqueueNDRangeKernel(_learnDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}
}

void Region::tm(ComputeSystem& cs, bool learn)
{
	// Store Current Neuron States to Previous Neuron States
	cs.getQueue().enqueueCopyBuffer(_nActives, _nActivesPrev, 0, 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueCopyBuffer(_nWinners, _nWinnersPrev, 0, 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nActives, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);
	cs.getQueue().enqueueFillBuffer(_nWinners, static_cast<cl_char>(0), 0, sizeof(cl_char) * _numN);

	// Set Neuron States
	cs.getQueue().enqueueCopyBuffer(_ddActives, _ddLearns, 0, 0, sizeof(cl_char) * _numDD);

	_setNeuronStates.setArg(0, _nActives);
	_setNeuronStates.setArg(1, _nWinners);
	_setNeuronStates.setArg(2, _pdActives);
	_setNeuronStates.setArg(3, _ddActives);
	_setNeuronStates.setArg(4, _ddLearns);
	_setNeuronStates.setArg(5, _dsAddrs);
	_setNeuronStates.setArg(6, _numDDpC);
	_setNeuronStates.setArg(7, _numNpC);
	_setNeuronStates.setArg(8, _numDDpN);
	_setNeuronStates.setArg(9, _numDSpDD);
	_setNeuronStates.setArg(10, _sMaxAddr);

	_range = cl::NDRange(_numC);
	cs.getQueue().enqueueNDRangeKernel(_setNeuronStates, cl::NullRange, _range);
	cs.getQueue().finish();

	// Learning
	if (learn)
	{
		cl::Buffer dsPotentials = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, sizeof(cl_char) * _numN);
		cs.getQueue().enqueueCopyBuffer(_nWinnersPrev, dsPotentials, 0, 0, sizeof(cl_char) * _numN);

		std::vector<char> nWinnersPrevVec(_numI);
		unsigned int numWinnersPrev = 0;

		cs.getQueue().enqueueReadBuffer(_nWinnersPrev, CL_TRUE, 0, sizeof(cl_char) * _numN, nWinnersPrevVec.data(), NULL);
	
		for (unsigned int n = 0; n < _numN; n++)
		{
			if (nWinnersPrevVec[n] == 1)
				numWinnersPrev++;
		}

		_learnDendrites.setArg(0, _nWinnersPrev);
		_learnDendrites.setArg(1, _dsAddrs);
		_learnDendrites.setArg(2, _dsPerms);
		_learnDendrites.setArg(3, _ddLearns);
		_learnDendrites.setArg(4, dsPotentials);
		_learnDendrites.setArg(5, numWinnersPrev);
		_learnDendrites.setArg(6, _numN);
		_learnDendrites.setArg(7, _numDSpDD);
		_learnDendrites.setArg(8, _sMaxAddr);
		_learnDendrites.setArg(9, _sMaxPerm);
		_learnDendrites.setArg(10, _sThresh);
		_learnDendrites.setArg(11, _sLearn);

		_range = cl::NDRange(_numDD);
		cs.getQueue().enqueueNDRangeKernel(_learnDendrites, cl::NullRange, _range);
		cs.getQueue().finish();
	}

	//Prediction
	_activateDendrites.setArg(0, _nActives);
	_activateDendrites.setArg(1, _dsAddrs);
	_activateDendrites.setArg(2, _dsPerms);
	_activateDendrites.setArg(3, _ddOverlaps);
	_activateDendrites.setArg(4, _ddActives);
	_activateDendrites.setArg(5, _numDSpDD);
	_activateDendrites.setArg(6, _sMaxAddr);
	_activateDendrites.setArg(7, _dThresh);
	_activateDendrites.setArg(8, _sThresh);

	_range = cl::NDRange(_numDD);
	cs.getQueue().enqueueNDRangeKernel(_activateDendrites, cl::NullRange, _range);
	cs.getQueue().finish();
}

void Region::print(ComputeSystem& cs)
{
	std::vector<unsigned short> ddOverlapsVec(_numDD);
	std::vector<char> ddActivesVec(_numDD);
	std::vector<char> ddLearnsVec(_numDD);
	std::vector<unsigned short> dsAddrsVec(_numDS);
	std::vector<char> dsPermsVec(_numDS);
	std::vector<char> nActivesVec(_numN);
	std::vector<char> nWinnersVec(_numN);
	std::vector<char> nActivesPrevVec(_numN);
	std::vector<char> nWinnersPrevVec(_numN);
	std::vector<unsigned short> pdOverlapsVec(_numPD);
	std::vector<char> pdActivesVec(_numPD);
	std::vector<unsigned short> psAddrsVec(_numPS);
	std::vector<char> psPermsVec(_numPS);
	std::vector<char> inputsVec(_numI);

	cs.getQueue().enqueueReadBuffer(_ddOverlaps, CL_TRUE, 0, sizeof(unsigned short) * _numDD, ddOverlapsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_ddActives, CL_TRUE, 0, sizeof(char) * _numDD, ddActivesVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_ddLearns, CL_TRUE, 0, sizeof(char) * _numDD, ddLearnsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dsAddrs, CL_TRUE, 0, sizeof(unsigned short) * _numDS, dsAddrsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_dsPerms, CL_TRUE, 0, sizeof(char) * _numDS, dsPermsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nActives, CL_TRUE, 0, sizeof(char) * _numN, nActivesVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nWinners, CL_TRUE, 0, sizeof(char) * _numN, nWinnersVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nActivesPrev, CL_TRUE, 0, sizeof(char) * _numN, nActivesPrevVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_nWinnersPrev, CL_TRUE, 0, sizeof(char) * _numN, nWinnersPrevVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_pdOverlaps, CL_TRUE, 0, sizeof(unsigned short) * _numPD, pdOverlapsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_pdActives, CL_TRUE, 0, sizeof(char) * _numPD, pdActivesVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_psAddrs, CL_TRUE, 0, sizeof(unsigned short) * _numPS, psAddrsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_psPerms, CL_TRUE, 0, sizeof(char) * _numPS, psPermsVec.data(), NULL);
	cs.getQueue().enqueueReadBuffer(_inputs, CL_TRUE, 0, sizeof(char) * _numI, inputsVec.data(), NULL);

	printf("\nDDOVE ");
	for(int i = 0; i < _numDD; i++){
		if (ddOverlapsVec[i] < 10){printf("0%i ", ddOverlapsVec[i]);}else{printf("%i ", ddOverlapsVec[i]);}}

	printf("\nDDACT ");
	for(int i = 0; i < _numDD; i++){
		if (ddActivesVec[i] < 10){printf("0%i ", ddActivesVec[i]);}else{printf("%i ", ddActivesVec[i]);}}

	printf("\nDDLRN ");
	for(int i = 0; i < _numDD; i++){
		if (ddLearnsVec[i] < 10){printf("0%i ", ddLearnsVec[i]);}else{printf("%i ", ddLearnsVec[i]);}}

	printf("\nDSADR ");
	for(int i = 0; i < _numDS; i++){if (dsAddrsVec[i] < 10){printf("0%i ", dsAddrsVec[i]);}else{printf("%i ", dsAddrsVec[i]);}}

	printf("\nDSPER ");
	for(int i = 0; i < _numDS; i++){if (dsPermsVec[i] < 10){printf("0%i ", dsPermsVec[i]);}else{printf("%i ", dsPermsVec[i]);}}

	printf("\nNACTC ");
	for(int i = 0; i < _numN; i++){if (nActivesVec[i] < 10){printf("0%i ", nActivesVec[i]);}else{printf("%i ", nActivesVec[i]);}}

	printf("\nNWINC ");
	for(int i = 0; i < _numN; i++){if (nWinnersVec[i] < 10){printf("0%i ", nWinnersVec[i]);}else{printf("%i ", nWinnersVec[i]);}}

	printf("\nNACTP ");
	for(int i = 0; i < _numN; i++){
		if (nActivesPrevVec[i] < 10){printf("0%i ", nActivesPrevVec[i]);}else{printf("%i ", nActivesPrevVec[i]);}}

	printf("\nNWINP ");
	for(int i = 0; i < _numN; i++){
		if (nWinnersPrevVec[i] < 10){printf("0%i ", nWinnersPrevVec[i]);}else{printf("%i ", nWinnersPrevVec[i]);}}

	printf("\nPDOVE ");
	for(int i = 0; i < _numPD; i++){
		if (pdOverlapsVec[i] < 10){printf("0%i ", pdOverlapsVec[i]);}else{printf("%i ", pdOverlapsVec[i]);}}

	printf("\nPDACT ");
	for(int i = 0; i < _numPD; i++){
		if (pdActivesVec[i] < 10){printf("0%i ", pdActivesVec[i]);}else{printf("%i ", pdActivesVec[i]);}}

	printf("\nPSADR ");
	for(int i = 0; i < _numPS; i++){if (psAddrsVec[i] < 10){printf("0%i ", psAddrsVec[i]);}else{printf("%i ", psAddrsVec[i]);}}

	printf("\nPSPER ");
	for(int i = 0; i < _numPS; i++){if (psPermsVec[i] < 10){printf("0%i ", psPermsVec[i]);}else{printf("%i ", psPermsVec[i]);}}

	printf("\nINPUT ");
	for(int i = 0; i < _numI; i++){if (inputsVec[i] < 10){printf("0%i ", inputsVec[i]);}else{printf("%i ", inputsVec[i]);}}

	printf("\n");
}

void Region::setInputs(ComputeSystem& cs, std::vector<char> inputsVec)
{
	cs.getQueue().enqueueWriteBuffer(_inputs, CL_TRUE, 0, sizeof(cl_char) * _numI, inputsVec.data());
}


std::vector<char> Region::getInputs(ComputeSystem &cs)
{
	std::vector<char> inputsVec(_numI);
	cs.getQueue().enqueueReadBuffer(_inputs, CL_TRUE, 0, sizeof(cl_char) * _numI, inputsVec.data(), NULL);
	return inputsVec;
}
