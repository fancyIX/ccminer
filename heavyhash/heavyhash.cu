extern "C" {
#include "keccak_tiny.h"
#include "heavyhash-gate.h"
}

#include <miner.h>
#include <cuda_helper.h>

extern void heavyhash_cpu_setBlock_80(uint32_t *pdata);
extern void heavyhash_cpu_setTarget(const void *pTargetIn);
extern uint32_t heavyhash_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, int order);
extern uint32_t heavyhash_getSecNonce(int thr_id, int num);
extern void heavyhash_cpu_free(int thr_id);
extern void heavyhash_init(int thr_id);


extern "C" void heavyhash_hash(void *ohash, const void *input)
{
    const uint32_t *data = (const uint32_t *) input;
    uint32_t seed[8];

    uint32_t matrix[64][64];
    struct xoshiro_state state;

    kt_sha3_256((uint8_t *)seed, 32, (const uint8_t *)(data+1), 32);

    for (int i = 0; i < 4; ++i) {
        state.s[i] = le64dec(seed + 2*i);
    }

    generate_matrix(matrix, &state);

    heavyhash(matrix, (uint8_t *)data, 80, (uint8_t *)ohash);
}

static bool init[MAX_GPUS] = { 0 };
static __thread uint32_t throughput = 0;

extern "C" int scanhash_heavyhash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x0400;

	if (!init[thr_id])
	{
        int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		CUDA_LOG_ERROR();

		int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 17 : 16;
		if (device_sm[device_map[thr_id]] == 500) intensity = 15;
		throughput = cuda_default_throughput(thr_id, 1U << intensity); // 18=256*256*4;
		if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);

        gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

        heavyhash_init(thr_id);

        init[thr_id] = true;
    }

    uint32_t _ALIGN(128) endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

    heavyhash_cpu_setBlock_80(pdata);
    heavyhash_cpu_setTarget(ptarget);

    do {
		int order = 0;
        work->nonces[0] = heavyhash_cpu_hash(thr_id, throughput, pdata[19], order++);
        *hashes_done = pdata[19] - first_nonce + throughput;

        if (work->nonces[0] != UINT32_MAX)
		{
            const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
            heavyhash_hash(vhash, endiandata);

            if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = heavyhash_getSecNonce(thr_id, 1);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					heavyhash_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				}
				else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;
    } while (!work_restart[thr_id].restart);

    *hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_heavyhash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	heavyhash_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
