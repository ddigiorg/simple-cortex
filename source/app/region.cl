// =========
// region.cl
// =========

/*
// http://stackoverflow.com/questions/16064591/simple-opencl-random-generator
uint randInt(uint2* seed)
{
	uint x = (*seed).x * 17 + (*seed).y * 13123;
	(*seed).x = (x << 13) ^ x;
	(*seed).y ^= (x << 7);

	uint randValue = x * (x * x * 15731 + 74323) + 871483;

	return randValue;
}

kernel void randomizeAddresses(
	global uchar* sInputs,
	global ushort* randomAddrs,
	global ushort* sAddrs,
	uint numI,
	uint numSpD,
	uint2 seed
)
{
	unsigned int d = get_global_id(0);

	uint2 tempSeed = seed + (uint2)(
		get_global_id(0) * 12 + 76 + get_global_size(0) * 3,
		get_global_size(0) * 21 + 42 + get_global_id(0) * 7) * 12;

//	private ushort dRandomAddrs[numI];

//	unsigned int s0 = d * numSpD;

	// Fisher-Yates Shuffle
	int tmp;
	for (int i = numI - 1; i > 0; i--)
	{
		int j = randInt(&seed) % (i + 1);
		tmp = randomAddrs[j];
//		a[j] = a[i];
//		a[i] = tmp;
	}

}
*/
kernel void activateDendrites(
	global uchar* sInputs,
	global ushort* sAddrs,
	global uchar* sPerms,
	global ushort* dOverlaps,
	global uchar* dActives,
	uint numSpD,
	uint sMaxAddr,
	uint dThresh,
	uint sThresh)
{
	uint d = get_global_id(0);

	dOverlaps[d] = 0;
	dActives[d] = 0;

	uint s0 = d * numSpD;
	for (uint s = s0; s < s0 + numSpD; s++)
	{
		if (sAddrs[s] < sMaxAddr)
		{
			if (sPerms[s] >= sThresh && sInputs[sAddrs[s]] > 0)
			{
				dOverlaps[d]++;
			}
		}
	}

	if (dOverlaps[d] >= dThresh)
		dActives[d] = 1;
}

// sPotentials wont be different through multiple work-items
kernel void learnDendrites(
	global uchar* sInputs,
	global ushort* sAddrs,
	global uchar* sPerms,
	global uchar* dActives,
	global uchar* sPotentials,
	uint numActiveInputs,
	uint numInputs,
	uint numSpD,
	uint sMaxAddr,
	uint sMaxPerm,
	uint sThresh,
	uint sLearn)
{
	uint d = get_global_id(0);

	if (dActives[d] > 0)
	{
		int numInsert = numActiveInputs;

		uint s0 = d * numSpD;
		for (uint s = s0; s < s0 + numSpD; s++)
		{
			if (sAddrs[s] < sMaxAddr)
			{
				numInsert--;
				sPotentials[sAddrs[s]] = 0;

				if (sPerms[s] > 0)
				{
					if(sInputs[sAddrs[s]] > 0)
					{
						sPerms[s] += sLearn;

						if (sPerms[s] > sMaxPerm)
							sPerms[s] = sMaxPerm;
					}
					else
						sPerms[s] -= sLearn;
				}
				else
					sAddrs[s] = sMaxAddr;
			}
		}

		uint i = 0;
		if (numInsert > 0)
		{
			for (uint s = s0; s < s0 + numSpD; s++) 
			{
				if (sAddrs[s] == sMaxAddr)
				{
					for (uint p = i; p < numInputs; p++)
					{
						if (sPotentials[p] > 0)
						{
							sAddrs[s] = p;
							sPerms[s] = sThresh;
							i = p + 1;
							break;
						}
					}
				}
			}
		}
	}
}

kernel void setNeuronStates(
	global uchar* nActives,
	global uchar* nWinners,
	global uchar* pdActives,
	global uchar* ddActives,
	global uchar* ddLearns,
	global ushort* dsAddrs,
	uint numDDpC,
	uint numNpC,
	uint numDDpN,
	uint numDSpDD,
	uint sMaxAddr)
{
	uint c = get_global_id(0);

	if (pdActives[c] > 0)
	{
		bool burst = true;

		uint d0 = c * numDDpC;
		for (uint d = d0; d < d0 + numDDpC; d++)
		{
			if (ddActives[d] > 0)
			{
				uint n = d  / numDDpN;
				nActives[n] = 1;
				nWinners[n] = 1;
				ddLearns[d] = 1;
				burst = false;
//				d = ((n + 1) * numDDpN) - 1;
			}
		}

		if (burst)
		{
			bool selectWinnerDendrite = true;

			uint n0 = c * numNpC;
			for (uint n = n0; n < n0 + numNpC; n++)
			{
				nActives[n] = 1;

				if (selectWinnerDendrite)
				{
					uint d0 = n * numDDpN;
					for (uint d = d0; d < d0 + numDDpN; d++)
					{
						uint overlap = 0;

						uint s0 = d * numDSpDD;
						for (uint s = s0; s < s0 + numDSpDD; s++)
						{
							if (dsAddrs[s] == sMaxAddr)
								overlap++;
						}

						if (overlap == numDDpN)
						{
							nWinners[n] = 1;
							ddLearns[d] = 1;
							selectWinnerDendrite = false;
							break;
						}
					}
				}
			}
		}
	}
}
