/**
 * BMW512
 */
extern "C" {
#include "sph/sph_bmw.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include <unistd.h>

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_setBlock_80(void *pdata);
void quark_bmw512_cpu_hash_80_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNonce, const uint64_t target);


extern "C" void bmw512_hash(void *state, const void *input) {
	sph_bmw512_context ctx_bmw;
	unsigned char hash[64];

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, input, 80);
	sph_bmw512_close(&ctx_bmw, hash);
	memcpy(state, hash, 32);
}


static bool init[MAX_GPUS] = { 0 };


extern "C" int scanhash_bmw512(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) {
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];

	if (opt_benchmark) ptarget[7] = 0x00ff;

	for (int k=0; k < 20; k++) be32enc(&endiandata[k], pdata[k]);

	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << 28);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		quark_bmw512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	quark_bmw512_cpu_setBlock_80((void*)endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	*hashes_done = 0;

	do {
		quark_bmw512_cpu_hash_80_final(thr_id, throughput, pdata[19], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		*hashes_done += throughput;

		if (h_resNonce[thr_id][0] != UINT32_MAX) {
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			bmw512_hash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[0] = startNounce + h_resNonce[thr_id][0];
				work_set_target_ratio(work, vhash);
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					uint32_t secNonce = work->nonces[1] = startNounce + h_resNonce[thr_id][1];
					be32enc(&endiandata[19], secNonce);
					bmw512_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				cudaMemset(d_resNonce[thr_id], 0xff, 2*sizeof(uint32_t));
				pdata[19] = startNounce + h_resNonce[thr_id][0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);
	return 0;
}


extern "C" void free_bmw512(int thr_id) {
	if (!init[thr_id]) return;

	cudaSetDevice(device_map[thr_id]);

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}