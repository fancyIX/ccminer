extern "C" {
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include <stdio.h>
#include <memory.h>

#define NBN 2

// GOST CPU Hash
extern "C" void gostd_hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[64];

	sph_gost512(hash, (const void*)input, 80);
	sph_gost256(hash, (const void*)hash, 64);

	memcpy(output, hash, 32);
}

extern "C" void gostd(void *output, const void *input, size_t len)
{
	unsigned char _ALIGN(64) hash[64];

	sph_gost512(hash, (const void*)input, len);
	sph_gost256(hash, (const void*)hash, 64);

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "gost"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };
extern void gostd_init(int thr_id);
extern void gostd_free(int thr_id);
extern void gostd_setBlock_80(uint32_t *pdata, uint32_t *ptarget);
extern void gostd_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces);

extern "C" int scanhash_gostd(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	//ptarget[7] = 0x000000FF;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 25);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x03;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		gostd_init(thr_id);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	gostd_setBlock_80(endiandata, ptarget);

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		gostd_hash_80(thr_id, throughput, pdata[19], work->nonces);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			uint32_t _ALIGN(64) vhash_le[8];

			endiandata[19] = swab32 (work->nonces[0]);
			gostd_hash(vhash, endiandata);
			if (swab32(vhash[0]) <= ptarget[7] /*&& fulltest(vhash, ptarget)*/) 
			{
				work->valid_nonces = 1;

				for (int i = 0; i < 8; i++)
					vhash_le[i] = swab32(vhash[7-i]);

				work_set_target_ratio(work, vhash_le);
				if (work->nonces[1] != UINT32_MAX) 
				{
					endiandata[19] = swab32 (work->nonces[1]);
					gostd_hash(vhash, endiandata);
					if (swab32(vhash[0]) <= ptarget[7] /*&& fulltest(vhash, ptarget)*/)
					{
						work->valid_nonces++;
						bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]);
				} 
				else 
					pdata[19] = work->nonces[0];
				return work->valid_nonces;
			}
			else if (swab32(vhash[0]) > ptarget[7]) 
			{
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_gostd(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	gostd_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}