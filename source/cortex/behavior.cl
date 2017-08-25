// ===========
// behavior.cl
// ===========

kernel void overlapSynapses(
	global uchar* nOverlaps,
	global uchar* sStates,
	global uint* sAddrs,
	global uchar* sPerms,
	const uint numSpD,
	const uint dThresh)
{
	uint n = get_global_id(0);

	uint dOverlap = 0;

	uint s0 = n * numSpD;

	for (uint s = s0; s < s0 + numSpD; s++)
		if (sPerms[s] > 0 && sStates[sAddrs[s]] > 0)
			dOverlap++;

	if (dOverlap >= dThresh)
		nOverlaps[n]++;
}

kernel void activateNeurons(
	global uint* nBoosts,
	global uchar* nStates,
	global uchar* nOverlaps,
	global uchar* inhibit,
	const uint sAddrMax,
	const uint nThresh)
{
	uint n = get_global_id(0);

	if (nBoosts[n] < sAddrMax)
		nBoosts[n]++;

	if (nOverlaps[n] >= nThresh)
	{
		nBoosts[n] = 0;
		nStates[n] = 1;
		inhibit[0] = 1;
	}
}

// Need to develop more optimized learning algorithm
kernel void learnSynapses(
	global uchar* sStates,
	const uint numStimulus,
	global uint* sAddrs,
	global uchar* sPerms,
	const uint numSpD,
	global uchar* nStates,
	const uchar sPermMax)
{
	uint n = get_global_id(0);

	if (nStates[n] > 0)
	{
		uint j = 0;

		uint s0 = n * numSpD;

		// Synaptic Learning - Grow or Shrink
		for (uint s = s0; s < s0 + numSpD; s++)
		{
			if (sPerms[s] > 0)
			{
				if (sStates[sAddrs[s]] > 0)
				{
					if (sPerms[s] < sPermMax)
						sPerms[s]++;
				}
				else
					sPerms[s]--;
			}
		}

		// Synaptic Learning - Move
		for (uint s = s0; s < s0 + numSpD; s++)
		{
			if (sPerms[s] == 0)
			{
				for (uint i = j; i < numStimulus; i++)
				{
					if (sStates[i] > 0)
					{
						bool flag = true;

						for (uint s2 = s0; s2 < s0 + numSpD; s2++)
						{
							if (sAddrs[s2] == i)
								flag = false;
								break;
						}

						if (flag)
						{
							sAddrs[s] = i;
							sPerms[s] = 1;
							j = i + 1;
							break;
						}
					}
				}				
			}
		}
	}
}

kernel void predictNeurons(
	global uchar* nStates,
	global uchar* nOverlaps,
	const uint nThresh)
{
	uint n = get_global_id(0);

	if (nOverlaps[n] >= nThresh)
		nStates[n] = 1;
}

kernel void decodeNeurons(
	global uchar* sStates,
	global uchar* nStates,
	global uint* sAddrs,
	global uchar* sPerms,
	const uint numSpD)
{
	uint n = get_global_id(0);

	if (nStates[n] > 0)
	{
		uint s0 = n * numSpD;

		for (uint s = s0; s < s0 + numSpD; s++)
		{
			if (sPerms[s] > 0)
				sStates[sAddrs[s]] = 1;
		}
	}
}
