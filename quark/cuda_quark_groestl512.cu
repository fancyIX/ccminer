// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include <cuda_helper.h>
#include "cuda_vector.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#endif

#define TPB 512
#define THF 4

#include "groestl_functions_quad.cu"
#include "bitslice_transformations_quad.cu"

#define WANT_GROESTL80
#ifdef WANT_GROESTL80
__constant__ static uint32_t c_Message80[20];
#endif

#include "cuda_quark_groestl512_sm2.cuh"
static cudaStream_t	gpustream[MAX_GPUS];

__global__ __launch_bounds__(TPB, 2)
void quark_groestl512_gpu_hash_64_quad(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	uint32_t __align__(16) msgBitsliced[8];
	uint32_t __align__(16) state[8];
	uint32_t __align__(16) hash[16];
	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
    if (thread < threads)
    {
        // GROESTL
        const uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
		const uint32_t hashPosition = nounce - startNounce;
        uint32_t *const inpHash = &g_hash[hashPosition * 16];

        const uint32_t thr = threadIdx.x & (THF-1);

		uint32_t message[8] =
		{
			inpHash[thr], inpHash[(THF)+thr], inpHash[(2 * THF) + thr], inpHash[(3 * THF) + thr],0, 0, 0, 
		};
		if (thr == 0) message[4] = 0x80UL;
		if (thr == 3) message[7] = 0x01000000UL;

		to_bitslice_quad(message, msgBitsliced);

        groestl512_progressMessage_quad(state, msgBitsliced);

		from_bitslice_quad(state, hash);

		if (thr == 0)
		{
			uint28 *phash = (uint28*)hash;
			uint28 *outpt = (uint28*)inpHash; /* var kept for hash align */
			outpt[0] = phash[0];
			outpt[1] = phash[1];
//			outpt[2] = phash[2];
//			outpt[3] = phash[3];
		}
    }
}

__host__ void quark_groestl512_cpu_init(int thr_id, uint32_t threads)
{
//    cudaGetDeviceProperties(&props[thr_id], device_map[thr_id]);
}

__host__
void quark_groestl512_cpu_free(int thr_id)
{
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
		quark_groestl512_sm20_free(thr_id);
}

__host__ void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{

    // berechne wie viele Thread Blocks wir brauchen
	dim3 grid(THF*((threads + TPB - 1) / TPB));
	dim3 block(TPB);

    quark_groestl512_gpu_hash_64_quad<<<grid, block, 0, gpustream[thr_id]>>>(threads, startNounce, d_hash, d_nonceVector);
	CUDA_SAFE_CALL(cudaGetLastError());
}

// --------------------------------------------------------------------------------------------------------------------------------------------

#ifdef WANT_GROESTL80

__host__
void groestl512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_Message80, endiandata, sizeof(c_Message80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(TPB, THF)
void groestl512_gpu_hash_80_quad(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
#if __CUDA_ARCH__ >= 300
	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread < threads)
	{
		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 M4 | M5 M6 M7 | (input)
		--|----------------|----------|
		T0|  0  4  8 12 16 | 80       |
		T1|  1  5       17 |          |
		T2|  2  6       18 |          |
		T3|  3  7       Nc |       01 |
		--|----------------|----------| TPR */

		uint32_t message[8];

		#pragma unroll 5
		for(int k=0; k<5; k++) message[k] = c_Message80[thr + (k * THF)];

		#pragma unroll 3
		for(int k=5; k<8; k++) message[k] = 0;

		if (thr == 0) message[5] = 0x80U;
		if (thr == 3) {
			message[4] = cuda_swab32(startNounce + thread);
			message[7] = 0x01000000U;
		}

		uint32_t msgBitsliced[8];
		to_bitslice_quad(message, msgBitsliced);

		uint32_t state[8];
		groestl512_progressMessage_quad(state, msgBitsliced);

		uint32_t hash[16];
		from_bitslice_quad(state, hash);

		if (thr == 0) { /* 4 threads were done */
			const off_t hashPosition = thread;
			//if (!thread) hash[15] = 0xFFFFFFFF;
			uint4 *outpt = (uint4*) &g_outhash[hashPosition << 4];
			uint4 *phash = (uint4*) hash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
	}
#endif
}

__host__
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	int dev_id = device_map[thr_id];

	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300) {
		const uint32_t threadsperblock = TPB;
		const uint32_t factor = THF;

		dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
		dim3 block(threadsperblock);

		groestl512_gpu_hash_80_quad <<<grid, block>>> (threads, startNounce, d_hash);

	} else {

		const uint32_t threadsperblock = 256;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);

		groestl512_gpu_hash_80_sm2 <<<grid, block>>> (threads, startNounce, d_hash);
	}
}

#endif
