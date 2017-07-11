// =========
// region.cl
// =========

kernel void overlapDendrites(
	global uchar* nOverlaps,
	global ushort* sAddrs,
	global uchar* sPerms,
	global uchar* inputs,
	uint numSpD,
	uint dThresh)
{
	uint n = get_global_id(0);

	uint dOverlap = 0;

	uint s0 = n * numSpD;
	for (uint s = s0; s < s0 + numSpD; s++)
	{
		if (sPerms[s] > 0 && inputs[sAddrs[s]] > 0)
			dOverlap++;
	}

	if (dOverlap >= dThresh)
	{
		nOverlaps[n]++;
	}
}

// Need to figure out how to optimize this algorithm
kernel void learnSynapses(
	global ushort* sAddrs,
	global uchar* sPerms,
	global uchar* nActives,
	global uchar* inputs,
	uint numSpD,
	uint numIn,
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

kernel void predictNeurons(
	global uchar* nPredicts,
	global uchar* nOverlaps,
	uint nPredThresh)
{
	uint n = get_global_id(0);

	if (nOverlaps[n] >= nPredThresh)
		nPredicts[n] = 1;
}

kernel void decodeNeurons(
	global uchar* outputs,
	global uchar* nPredicts,
	global ushort* sAddrs)
{
	uint n = get_global_id(0);

	if (nPredicts[n] > 0)
		outputs[sAddrs[n]] = 1;
}
