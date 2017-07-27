// =======
// area.cl
// =======

kernel void overlapDendrites(
	global uchar* inputs,
	global uchar* nOverlaps,
	global ushort* sAddrs,
	global uchar* sPerms,
	uint numSpD,
	uint dThresh)
{
	uint n = get_global_id(0);

	uint dOverlap = 0;

	uint s0 = n * numSpD;

	for (uint s = s0; s < s0 + numSpD; s++)
		if (sPerms[s] > 0 && inputs[sAddrs[s]] > 0)
			dOverlap++;

	if (dOverlap >= dThresh)
		nOverlaps[n]++;
}

kernel void learnSynapses(
	global uchar* inputs,
	uint numIn,
	global ushort* sAddrs,
	global uchar* sPerms,
	uint numSpD,
	global uchar* nActives,
	uint sPermMax)
{
	uint n = get_global_id(0);

	if (nActives[n] > 0)
	{
		uint j = 0;

		uint s0 = n * numSpD;
		for (uint s = s0; s < s0 + numSpD; s++)
		{
			if (sPerms[s] > 0)
			{
				if (inputs[sAddrs[s]] > 0)
				{
					if (sPerms[s] < sPermMax)
						sPerms[s]++;
				}
				else
					sPerms[s]--;
			}
		}

		for (uint s = s0; s < s0 + numSpD; s++)
		{
			if (sPerms[s] == 0)
			{
				for (uint i = j; i < numIn; i++)
				{
					if (inputs[i] > 0)
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

kernel void activateNeurons(
	global ushort* nBoosts,
	global uchar* nActives,
	global uchar* nOverlaps,
	global uchar* inhibitFlag,
	uint sAddrMax,
	uint nThresh)
{
	uint n = get_global_id(0);

	if (nBoosts[n] < sAddrMax)
		nBoosts[n]++;

	if (nOverlaps[n] >= nThresh)
	{
		nActives[n] = 1;
		nBoosts[n] = 0;
		inhibitFlag[0] = 1;
	}
}

kernel void predictNeurons(
	global uchar* nActives,
	global uchar* nOverlaps,
	uint nThresh)
{
	uint n = get_global_id(0);

	if (nOverlaps[n] >= nThresh)
		nActives[n] = 1;
}

kernel void decodeNeurons(
	global uchar* outputs,
	global uchar* nActives,
	global ushort* sAddrs,
	global ushort* sPerms,
	uint numSpD)
{
	uint n = get_global_id(0);

	if (nActives[n] > 0)
	{
		uint s0 = n * numSpD;
		for (uint s = s0; s < s0 + numSpD; s++)
		{
//			if (sPerms[s] > 0)
				outputs[sAddrs[s]] = 1;
		}
	}
}
