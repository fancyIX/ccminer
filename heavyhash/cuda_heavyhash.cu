extern "C" {
#include "keccak_tiny.h"
#include "heavyhash-gate.h"
}

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "miner.h"

#include <memory.h>

__constant__ static uint32_t c_data[20];
__constant__ static uint32_t c_matrix[64][64];
__constant__ uint32_t pTarget[8];

static uint32_t *h_GNonces[MAX_GPUS];
static uint32_t *d_GNonces[MAX_GPUS];

typedef union {
    uint32_t h4[8];
    uint64_t h8[4];
    uint4 h16[2];
    ulong2 hl16[2];
    ulong4 h32;
} hash_t;

__constant__ uint2 keccak_round_constants35[24] = {
	{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
	{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
	{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
	{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
	{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
	{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
	{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
	{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
	{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
	{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};

static void __forceinline__ __device__ keccak_block(uint2 *s)
{
	uint2 bc[5], tmpxor[5], u, v;
	//	uint2 s[25];

	#pragma unroll 1
	for (int i = 0; i < 24; i++)
	{
		#pragma unroll
		for (uint32_t x = 0; x < 5; x++)
			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		u = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(s[22] ^ bc[1], 61);
		s[22] = ROL2(s[14] ^ bc[3], 39);
		s[14] = ROL2(s[20] ^ bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(s[12] ^ bc[1], 43);
		s[12] = ROL2(s[13] ^ bc[2], 25);
		s[13] = ROL8(s[19] ^ bc[3]);
		s[19] = ROR8(s[23] ^ bc[2]);
		s[23] = ROL2(s[15] ^ bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(s[24] ^ bc[3], 14);
		s[24] = ROL2(s[21] ^ bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(s[18] ^ bc[2], 21);
		s[18] = ROL2(s[17] ^ bc[1], 15);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(u, 1);

		u = s[0]; v = s[1]; s[0] ^= (~v) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & u; s[4] ^= (~u) & v;
		u = s[5]; v = s[6]; s[5] ^= (~v) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & u; s[9] ^= (~u) & v;
		u = s[10]; v = s[11]; s[10] ^= (~v) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & u; s[14] ^= (~u) & v;
		u = s[15]; v = s[16]; s[15] ^= (~v) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & u; s[19] ^= (~u) & v;
		u = s[20]; v = s[21]; s[20] ^= (~v) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & u; s[24] ^= (~u) & v;
		s[0] ^= keccak_round_constants35[i];
	}
}

__global__
void heavyhash_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces)
{
	__shared__ uint64_t matrix[1024 * 2];

    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    uint32_t nonce = startNonce + thread;
    if (thread < threads)
	{
		uint32_t tid = threadIdx.x;
		uint64_t *cp = (uint64_t *)(c_matrix);
		for (int i = 0; i < 8; i++) {
			matrix[tid + i * 256] = cp[tid + i * 256];
		}

        hash_t hash;

        uint32_t pdata[50] = {0};
        for (int i = 0; i < 19; i++) {
            pdata[i] = c_data[i];
        }
        pdata[19] = cuda_swab32(nonce);

        uint8_t hash_first[32];
        uint8_t hash_second[32];
        uint8_t hash_xored[32];

        uint32_t vector[64];
        uint32_t product[64];

        ((uint8_t *) pdata)[80] = 0x06;
        ((uint8_t *) pdata)[135] = 0x80;

        keccak_block((uint2 *) pdata);

        for (int i = 0; i < 4; i++) {
            ((uint64_t *)hash_first)[i] = ((uint64_t *) pdata)[i];
        }

        for (int i = 0; i < 32; ++i) {
            vector[2*i] = (hash_first[i] >> 4);
            vector[2*i+1] = hash_first[i] & 0xF;
        }

        for (int i = 0; i < 64; ++i) {
            uint32_t sum = 0;
			for (int k = 0; k < 8; k++) {
				uint64_t buf0 = matrix[i * 32 + k * 4 + 0];
				uint64_t buf1 = matrix[i * 32 + k * 4 + 1];
				uint64_t buf2 = matrix[i * 32 + k * 4 + 2];
				uint64_t buf3 = matrix[i * 32 + k * 4 + 3];
				uint32_t *m0 = (uint32_t *)&buf0;
				for (int j = 0; j < 2; j++) {
					sum += m0[j] * vector[(k * 4 + 0) * 2 + j];
				}
				uint32_t *m1 = (uint32_t *)&buf1;
				for (int j = 0; j < 2; j++) {
					sum += m1[j] * vector[(k * 4 + 1) * 2 + j];
				}
				uint32_t *m2 = (uint32_t *)&buf2;
				for (int j = 0; j < 2; j++) {
					sum += m2[j] * vector[(k * 4 + 2) * 2 + j];
				}
				uint32_t *m3 = (uint32_t *)&buf3;
				for (int j = 0; j < 2; j++) {
					sum += m3[j] * vector[(k * 4 + 3) * 2 + j];
				}
			}
            product[i] = (sum >> 10);
        }

        for (int i = 0; i < 32; ++i) {
            hash_second[i] = (product[2*i] << 4) | (product[2*i+1]);
        }

        for (int i = 0; i < 32; ++i) {
            hash_xored[i] = hash_first[i] ^ hash_second[i];
        }

        uint32_t tmp[50] = {0};
        for (int i = 0; i < 32; i++) {
            ((uint8_t *) tmp)[i] = hash_xored[i];
        }

        ((uint8_t *)tmp)[32] = 0x06;
        ((uint8_t *)tmp)[135] = 0x80;

        keccak_block((uint2 *) tmp);

        for (int i = 0; i < 4; i++) {
            hash.h8[i] = ((uint64_t *) tmp)[i];
        }

		if ( hash.h8[3] <= ((uint64_t *) pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
    }
}

__host__
void heavyhash_cpu_setBlock_80(uint32_t *pdata)
{
	uint32_t data[20];
	for (int k = 0; k < 20; k++)
		be32enc(&data[k], pdata[k]);

	cudaMemcpyToSymbol(c_data, &data[0], sizeof(c_data), 0, cudaMemcpyHostToDevice);

    uint32_t seed[8];
    uint32_t matrix[64][64];
    struct xoshiro_state state;

    kt_sha3_256((uint8_t *)seed, 32, (const uint8_t *)(data+1), 32);

    for (int i = 0; i < 4; ++i) {
        state.s[i] = le64dec(seed + 2*i);
    }

    generate_matrix(matrix, &state);

    cudaMemcpyToSymbol(c_matrix, &matrix[0][0], sizeof(c_matrix), 0, cudaMemcpyHostToDevice);
}

__host__
void heavyhash_cpu_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}

__host__
void heavyhash_init(int thr_id)
{
    cudaMalloc(&d_GNonces[thr_id], 2*sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2*sizeof(uint32_t));
}

__host__
uint32_t heavyhash_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, int order)
{
	uint32_t result = UINT32_MAX;
	cudaMemset(d_GNonces[thr_id], 0xff, 2*sizeof(uint32_t));
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
    size_t shared_size = 8192 * 2;

	heavyhash_gpu_hash<<<grid, block, shared_size>>>(threads, startNounce, d_GNonces[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	return result;
}

__host__
uint32_t heavyhash_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void heavyhash_cpu_free(int thr_id)
{
	cudaFree(d_GNonces[thr_id]);
	cudaFreeHost(h_GNonces[thr_id]);
}
