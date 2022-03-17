/**
 * BuBoNiK666
 * Lyra2 (v1) cuda implementation based on djm34 work
 * tpruvot@github 2015, Nanashi 08/2016 (from 1.8-r2)
 */

 #include <stdio.h>
 #include <memory.h>
 
 #define TPB52 32
 
 #include "cuda_lyra2_sm2.cuh"
 #include "cuda_lyra2_sm5.cuh"
 
 #ifdef __INTELLISENSE__
 /* just for vstudio code colors */
 #define __CUDA_ARCH__ 520
 #endif
 
 #if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ > 500
 
 #include "cuda_lyra2_vectors.h"
 
 #ifdef __INTELLISENSE__
 /* just for vstudio code colors */
 __device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
 #endif
 
 #define Nrow 8
 #define Ncol 8
 #define memshift 3
 
 #define BUF_COUNT 0
 
 __device__ uint2 *DMatrix;
 
 __device__ __forceinline__ void LD4SSB(uint2 res[3], const int row, const int col, const int thread, const int threads)
 {
     extern __shared__ uint2 shared_mem[];
     const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;

	res[0] = shared_mem[((s0 + 0) * 8 + threadIdx.y) * 4 + threadIdx.x];
	res[1] = shared_mem[((s0 + 1) * 8 + threadIdx.y) * 4 + threadIdx.x];
	res[2] = shared_mem[((s0 + 2) * 8 + threadIdx.y) * 4 + threadIdx.x];
 }
 
 __device__ __forceinline__ void ST4SSB(const int row, const int col, const uint2 data[3], const int thread, const int threads)
 {
     extern __shared__ uint2 shared_mem[];
     const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
 
     shared_mem[((s0 + 0) * 8 + threadIdx.y) * 4 + threadIdx.x] = data[0];
	 shared_mem[((s0 + 1) * 8 + threadIdx.y) * 4 + threadIdx.x] = data[1];
	 shared_mem[((s0 + 2) * 8 + threadIdx.y) * 4 + threadIdx.x] = data[2];
 }

 __device__ __forceinline__ void LD4SS(uint2 res[3], const int row, const int col, const int thread, const int threads)
 {
     extern __shared__ uint2 shared_mem[];
     const int s0 = (Ncol / 2 * (row - BUF_COUNT) + col / 2) * memshift;

	res[0] = shared_mem[((s0 + 0) * 16 + threadIdx.y) * 4 + threadIdx.x];
	res[1] = shared_mem[((s0 + 1) * 16 + threadIdx.y) * 4 + threadIdx.x];
	res[2] = shared_mem[((s0 + 2) * 16 + threadIdx.y) * 4 + threadIdx.x];
 }
 
 __device__ __forceinline__ void ST4SS(const int row, const int col, const uint2 data[3], const int thread, const int threads)
 {
     extern __shared__ uint2 shared_mem[];
     const int s0 = (Ncol / 2 * (row - BUF_COUNT) + col / 2) * memshift;
 
     shared_mem[((s0 + 0) * 16 + threadIdx.y) * 4 + threadIdx.x] = data[0];
	 shared_mem[((s0 + 1) * 16 + threadIdx.y) * 4 + threadIdx.x] = data[1];
	 shared_mem[((s0 + 2) * 16 + threadIdx.y) * 4 + threadIdx.x] = data[2];
 }

 __device__ __forceinline__ void LD4SL(uint2 res[3], const int row, const int col, uint2 pad[Ncol / 2][Nrow][3])
 {
         res[0] = pad[col / 2][row][0];
         res[1] = pad[col / 2][row][1];
         res[2] = pad[col / 2][row][2];
 }

 __device__ __forceinline__ void ST4SL(const int row, const int col, const uint2 data[3], uint2 pad[Ncol / 2][Nrow][3])
 {
         pad[col / 2][row][0] = data[0];
         pad[col / 2][row][1] = data[1];
         pad[col / 2][row][2] = data[2];
 }

 __device__ __forceinline__ void LD4SG(uint2 res[3], const int row, const int col, const int thread, const int threads, const int offset, uint64_t *GPad, uint64_t *GPadO)
 {
    uint64_t *p = (uint64_t *) ((GPad + offset + 2 * (threadIdx.x + 4 * threadIdx.y + (((col /2) * 64 + row * 64 * (Ncol / 2))))));
    uint64_t *q = (uint64_t *) (GPadO + offset + (threadIdx.x + 4 * threadIdx.y + ((col /2) * 64 + row * 64 * (Ncol / 2))));
     /*uint4 tmp = __ldg((uint4 *)p);
     res[0] = make_uint2(tmp.x, tmp.y);
     res[1] = make_uint2(tmp.z, tmp.w);
     res[2] = __ldg((uint2 *)q);*/
     uint64_t * r = (uint64_t *)(&res[0]);
     asm ("ld.global.ca.v2.u64 {%0, %1}, [%3];\n\tld.global.ca.u64 %2, [%4];"
     : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]) : "l"(p), "l"(q) : "memory");
 }

 __device__ __forceinline__ void ST4SG(const int row, const int col, const uint2 data[3], const int thread, const int threads, const int offset, uint64_t *GPad, uint64_t *GPadO)
 {
     uint64_t a = MAKE_ULONGLONG(data[0].x, data[0].y);
     uint64_t b = MAKE_ULONGLONG(data[1].x, data[1].y);
     uint64_t c = MAKE_ULONGLONG(data[2].x, data[2].y);
     uint64_t *p = (uint64_t *) ((GPad + offset + 2 * (threadIdx.x + 4 * threadIdx.y + (((col /2) * 64 + row * 64 * (Ncol / 2))))));
     uint64_t *q = (uint64_t *) (GPadO + offset + (threadIdx.x + 4 * threadIdx.y + ((col /2) * 64 + row * 64 * (Ncol / 2))));
     asm ("st.global.v2.u64 [%0], {%2, %3};\n\t st.global.u64 [%1], %4;\n\t" :: "l"(p), "l"(q), "l"(a), "l"(b), "l"(c): "memory");
    //((uint4*)GPad)[offset + ((threadIdx.x + 4 * threadIdx.y + (col /2) * 64 + row * 64 * (Ncol / 2)))] = tmp;
    //GPad[offset + ((threadIdx.x + 4 * threadIdx.y + (col /2) * 64 + row * 64 * (Ncol / 2))) + 64 * Nrow * (Ncol / 2) * 2] = data[2];
 }
 
 #if __CUDA_ARCH__ >= 300
 __device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
 {
     return __shfl(a, b, c);
 }
 
 __device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
 {
     return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
 }
 
 __device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
 {
     a1 = WarpShuffle(a1, b1, c);
     a2 = WarpShuffle(a2, b2, c);
     a3 = WarpShuffle(a3, b3, c);
 }
 
 #else
 __device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
 {
     extern __shared__ uint2 shared_mem[];
 
     const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
     uint32_t *_ptr = (uint32_t*)shared_mem;
 
     __threadfence_block();
     uint32_t buf = _ptr[thread];
 
     _ptr[thread] = a;
     __threadfence_block();
     uint32_t result = _ptr[(thread&~(c - 1)) + (b&(c - 1))];
 
     __threadfence_block();
     _ptr[thread] = buf;
 
     __threadfence_block();
     return result;
 }
 
 __device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
 {
     extern __shared__ uint2 shared_mem[];
 
     const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
 
     __threadfence_block();
     uint2 buf = shared_mem[thread];
 
     shared_mem[thread] = a;
     __threadfence_block();
     uint2 result = shared_mem[(thread&~(c - 1)) + (b&(c - 1))];
 
     __threadfence_block();
     shared_mem[thread] = buf;
 
     __threadfence_block();
     return result;
 }
 
 __device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
 {
     extern __shared__ uint2 shared_mem[];
 
     const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
 
     __threadfence_block();
     uint2 buf = shared_mem[thread];
 
     shared_mem[thread] = a1;
     __threadfence_block();
     a1 = shared_mem[(thread&~(c - 1)) + (b1&(c - 1))];
     __threadfence_block();
     shared_mem[thread] = a2;
     __threadfence_block();
     a2 = shared_mem[(thread&~(c - 1)) + (b2&(c - 1))];
     __threadfence_block();
     shared_mem[thread] = a3;
     __threadfence_block();
     a3 = shared_mem[(thread&~(c - 1)) + (b3&(c - 1))];
 
     __threadfence_block();
     shared_mem[thread] = buf;
     __threadfence_block();
 }
 
 #endif
 
 #if __CUDA_ARCH__ > 500 || !defined(__CUDA_ARCH)
 static __device__ __forceinline__
 void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
 {
     a += b; uint2 tmp = d; d.y = a.x ^ tmp.x; d.x = a.y ^ tmp.y;
     c += d; b ^= c; b = ROR24(b);
     a += b; d ^= a; d = ROR16(d);
     c += d; b ^= c; b = ROR2(b, 63);
 }
 #endif
 
 __device__ __forceinline__ void round_lyra(uint2 s[4])
 {
     Gfunc(s[0], s[1], s[2], s[3]);
     WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
     Gfunc(s[0], s[1], s[2], s[3]);
     WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
 }
 
 static __device__ __forceinline__
 void round_lyra(uint2x4* s)
 {
     Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
     Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
     Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
     Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
     Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
     Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
     Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
     Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
 }
 
 static __device__ __forceinline__
 void reduceDuplex(uint2 state[4], uint32_t thread, const uint32_t threads, uint2 pad[Ncol / 2][Nrow][3])
 {
     uint2 state1[3];
	 uint2 state2[3];
 

     for (int i = 0; i < Nrow; i++)
     {
         if ((i & 1) == 1)
         ST4SS(0, Ncol - i - 1, state, thread, threads);
         else
         ST4SL(0, Ncol - i - 1, state, pad);
 
         round_lyra(state);
     }
 
     for (int i = 0; i < Nrow; i+=2)
     {
         LD4SS(state1, 0, i, thread, threads);
		 LD4SL(state2, 0, i + 1, pad);
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j];
 
         round_lyra(state);
 
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state1[j] ^= state[j];
		
			 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state2[j];
 
         round_lyra(state);
 
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state2[j] ^= state[j];
         ST4SL(1, Ncol - i - 1, state1, pad);
		 ST4SS(1, Ncol - (i + 1) - 1, state2, thread, threads);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads, uint2 pad[Ncol / 2][Nrow][3])
 {
     uint2 state1[3], state2[3], state3[3], state4[3];

     for (int i = 0; i < Nrow; i+=2)
     {
         LD4SS(state1, rowIn, i, thread, threads);
		 LD4SS(state2, rowInOut, i, thread, threads);
		 LD4SL(state3, rowIn, i + 1, pad);
		 LD4SL(state4, rowInOut, i + 1, pad);
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         round_lyra(state);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state1[j] ^= state[j];
 
         ST4SL(rowOut, Ncol - i - 1, state1, pad);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data0 = state[0];
         uint2 Data1 = state[1];
         uint2 Data2 = state[2];
         WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state2[0] ^= Data2;
             state2[1] ^= Data0;
             state2[2] ^= Data1;
         } else {
             state2[0] ^= Data0;
             state2[1] ^= Data1;
             state2[2] ^= Data2;
         }
 
         ST4SS(rowInOut, i, state2, thread, threads);

		//=====================================
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
         round_lyra(state);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state3[j] ^= state[j];
 
         ST4SS(rowOut, Ncol - (i + 1) - 1, state3, thread, threads);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data01 = state[0];
         uint2 Data11 = state[1];
         uint2 Data21 = state[2];
         WarpShuffle3(Data01, Data11, Data21, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state4[0] ^= Data21;
             state4[1] ^= Data01;
             state4[2] ^= Data11;
         } else {
             state4[0] ^= Data01;
             state4[1] ^= Data11;
             state4[2] ^= Data21;
         }
 
         ST4SL(rowInOut, (i + 1), state4, pad);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads, uint2 pad[Ncol / 2][Nrow][3])
 {
     for (int i = 0; i < Nrow; i+=2)
     {
         uint2 state1[3], state2[3], state3[3], state4[3];
 
         LD4SS(state1, rowIn, i, thread, threads);
         LD4SS(state2, rowInOut, i, thread, threads);
		 LD4SL(state3, rowIn, i + 1, pad);
         LD4SL(state4, rowInOut, i + 1, pad);
 
 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         LD4SS(state1, rowOut, i, thread, threads);

         round_lyra(state);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data0 = state[0];
         uint2 Data1 = state[1];
         uint2 Data2 = state[2];
         WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state2[0] ^= Data2;
             state2[1] ^= Data0;
             state2[2] ^= Data1;
         }
         else
         {
             state2[0] ^= Data0;
             state2[1] ^= Data1;
             state2[2] ^= Data2;
         }

        if (rowInOut != rowOut) {
             ST4SS(rowInOut, i, state2, thread, threads);
                 #pragma unroll
            for (int j = 0; j < 3; j++)
                state2[j] = state1[j];
         }

#pragma unroll
        for (int j = 0; j < 3; j++)
            state2[j] ^= state[j];

        ST4SS(rowOut, i, state2, thread, threads);

		 //======================================
 
 
 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
        LD4SL(state3, rowOut, i + 1, pad);

         round_lyra(state);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data01 = state[0];
         uint2 Data11 = state[1];
         uint2 Data21 = state[2];
         WarpShuffle3(Data01, Data11, Data21, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state4[0] ^= Data21;
             state4[1] ^= Data01;
             state4[2] ^= Data11;
         }
         else
         {
             state4[0] ^= Data01;
             state4[1] ^= Data11;
             state4[2] ^= Data21;
         }

         if (rowInOut != rowOut) {
             ST4SL(rowInOut, i + 1, state4, pad);
                 #pragma unroll
            for (int j = 0; j < 3; j++)
                state4[j] = state3[j];
         }

#pragma unroll
        for (int j = 0; j < 3; j++)
            state4[j] ^= state[j];

        ST4SL(rowOut, i + 1, state4, pad);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowt_8(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads, uint2 pad[Ncol / 2][Nrow][3])
 {
     uint2 state1[3], state2[3], state3[3], state4[3], last[3];
 
     LD4SS(state1, 2, 0, thread, threads);
     LD4SS(last, rowInOut, 0, thread, threads);
 
     #pragma unroll
     for (int j = 0; j < 3; j++)
         state[j] ^= state1[j] + last[j];
 
     round_lyra(state);
 
     // simultaneously receive data from preceding thread and send data to following thread
     uint2 Data0 = state[0];
     uint2 Data1 = state[1];
     uint2 Data2 = state[2];
     WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
     if (threadIdx.x == 0)
     {
         last[0] ^= Data2;
         last[1] ^= Data0;
         last[2] ^= Data1;
     } else {
         last[0] ^= Data0;
         last[1] ^= Data1;
         last[2] ^= Data2;
     }
 
     if (rowInOut == 5)
     {
         #pragma unroll
         for (int j = 0; j < 3; j++)
             last[j] ^= state[j];
     }
 
	 LD4SL(state1, 2, 1, pad);
	 LD4SL(state2, rowInOut, 1, pad);

	 #pragma unroll
	 for (int j = 0; j < 3; j++)
		 state[j] ^= state1[j] + state2[j];

	 round_lyra(state);

     for (int i = 2; i < Nrow; i+=2)
     {
         LD4SS(state1, 2, i, thread, threads);
         LD4SS(state2, rowInOut, i, thread, threads);
		 LD4SL(state3, 2, i + 1, pad);
         LD4SL(state4, rowInOut, i + 1, pad);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         round_lyra(state);

		 //============================
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
         round_lyra(state);
     }
 
     #pragma unroll
     for (int j = 0; j < 3; j++)
         state[j] ^= last[j];
 }

 // ================================= big local mem ==========================
 static __device__ __forceinline__
 void reduceDuplex_biglocal(uint2 state[4], uint32_t thread, const uint32_t threads)
 {
     uint2 state1[3];
	 uint2 state2[3];
 

     for (int i = 0; i < Nrow; i++)
     {
         ST4SSB(0, Ncol - i - 1, state, thread, threads);
 
         round_lyra(state);
     }
 
     for (int i = 0; i < Nrow; i+=2)
     {
         LD4SSB(state1, 0, i, thread, threads);
		 LD4SSB(state2, 0, i + 1, thread, threads);
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j];
 
         round_lyra(state);
 
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state1[j] ^= state[j];
		
			 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state2[j];
 
         round_lyra(state);
 
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state2[j] ^= state[j];
         ST4SSB(1, Ncol - i - 1, state1, thread, threads);
		 ST4SSB(1, Ncol - (i + 1) - 1, state2, thread, threads);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowSetup_biglocal(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads)
 {
     uint2 state1[3], state2[3], state3[3], state4[3];

     for (int i = 0; i < Nrow; i+=2)
     {
         LD4SSB(state1, rowIn, i, thread, threads);
		 LD4SSB(state2, rowInOut, i, thread, threads);
		 LD4SSB(state3, rowIn, i + 1, thread, threads);
		 LD4SSB(state4, rowInOut, i + 1, thread, threads);
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         round_lyra(state);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state1[j] ^= state[j];
 
         ST4SSB(rowOut, Ncol - i - 1, state1, thread, threads);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data0 = state[0];
         uint2 Data1 = state[1];
         uint2 Data2 = state[2];
         WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state2[0] ^= Data2;
             state2[1] ^= Data0;
             state2[2] ^= Data1;
         } else {
             state2[0] ^= Data0;
             state2[1] ^= Data1;
             state2[2] ^= Data2;
         }
 
         ST4SSB(rowInOut, i, state2, thread, threads);

		//=====================================
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
         round_lyra(state);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state3[j] ^= state[j];
 
         ST4SSB(rowOut, Ncol - (i + 1) - 1, state3, thread, threads);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data01 = state[0];
         uint2 Data11 = state[1];
         uint2 Data21 = state[2];
         WarpShuffle3(Data01, Data11, Data21, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state4[0] ^= Data21;
             state4[1] ^= Data01;
             state4[2] ^= Data11;
         } else {
             state4[0] ^= Data01;
             state4[1] ^= Data11;
             state4[2] ^= Data21;
         }
 
         ST4SSB(rowInOut, (i + 1), state4, thread, threads);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowt_biglocal(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
 {
     for (int i = 0; i < Nrow; i+=2)
     {
         uint2 state1[3], state2[3], state3[3], state4[3];
 
         LD4SSB(state1, rowIn, i, thread, threads);
         LD4SSB(state2, rowInOut, i, thread, threads);
		 LD4SSB(state3, rowIn, i + 1, thread, threads);
         LD4SSB(state4, rowInOut, i + 1, thread, threads);
 
 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         LD4SSB(state1, rowOut, i, thread, threads);

         round_lyra(state);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data0 = state[0];
         uint2 Data1 = state[1];
         uint2 Data2 = state[2];
         WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state2[0] ^= Data2;
             state2[1] ^= Data0;
             state2[2] ^= Data1;
         }
         else
         {
             state2[0] ^= Data0;
             state2[1] ^= Data1;
             state2[2] ^= Data2;
         }

        if (rowInOut != rowOut) {
             ST4SSB(rowInOut, i, state2, thread, threads);
                 #pragma unroll
            for (int j = 0; j < 3; j++)
                state2[j] = state1[j];
         }

#pragma unroll
        for (int j = 0; j < 3; j++)
            state2[j] ^= state[j];

        ST4SSB(rowOut, i, state2, thread, threads);

		 //======================================
 
 
 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
        LD4SSB(state3, rowOut, i + 1, thread, threads);

         round_lyra(state);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data01 = state[0];
         uint2 Data11 = state[1];
         uint2 Data21 = state[2];
         WarpShuffle3(Data01, Data11, Data21, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state4[0] ^= Data21;
             state4[1] ^= Data01;
             state4[2] ^= Data11;
         }
         else
         {
             state4[0] ^= Data01;
             state4[1] ^= Data11;
             state4[2] ^= Data21;
         }

         if (rowInOut != rowOut) {
             ST4SSB(rowInOut, i + 1, state4, thread, threads);
                 #pragma unroll
            for (int j = 0; j < 3; j++)
                state4[j] = state3[j];
         }

#pragma unroll
        for (int j = 0; j < 3; j++)
            state4[j] ^= state[j];

        ST4SSB(rowOut, i + 1, state4, thread, threads);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowt_8_biglocal(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
 {
     uint2 state1[3], state2[3], state3[3], state4[3], last[3];
 
     LD4SSB(state1, 2, 0, thread, threads);
     LD4SSB(last, rowInOut, 0, thread, threads);
 
     #pragma unroll
     for (int j = 0; j < 3; j++)
         state[j] ^= state1[j] + last[j];
 
     round_lyra(state);
 
     // simultaneously receive data from preceding thread and send data to following thread
     uint2 Data0 = state[0];
     uint2 Data1 = state[1];
     uint2 Data2 = state[2];
     WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
     if (threadIdx.x == 0)
     {
         last[0] ^= Data2;
         last[1] ^= Data0;
         last[2] ^= Data1;
     } else {
         last[0] ^= Data0;
         last[1] ^= Data1;
         last[2] ^= Data2;
     }
 
     if (rowInOut == 5)
     {
         #pragma unroll
         for (int j = 0; j < 3; j++)
             last[j] ^= state[j];
     }
 
	 LD4SSB(state1, 2, 1, thread, threads);
	 LD4SSB(state2, rowInOut, 1, thread, threads);

	 #pragma unroll
	 for (int j = 0; j < 3; j++)
		 state[j] ^= state1[j] + state2[j];

	 round_lyra(state);

     for (int i = 2; i < Nrow; i+=2)
     {
         LD4SSB(state1, 2, i, thread, threads);
         LD4SSB(state2, rowInOut, i, thread, threads);
		 LD4SSB(state3, 2, i + 1, thread, threads);
         LD4SSB(state4, rowInOut, i + 1, thread, threads);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         round_lyra(state);

		 //============================
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
         round_lyra(state);
     }
 
     #pragma unroll
     for (int j = 0; j < 3; j++)
         state[j] ^= last[j];
 }
 // ================================= local mem big ==========================

 // ================================= high end ==========================
 static __device__ __forceinline__
 void reduceDuplex_high_end(uint2 state[4], uint32_t thread, const uint32_t threads, uint64_t *g_pad)
 {

    const int offset = blockIdx.x * 64 * Nrow * (Ncol / 2) * 3;
    uint64_t *g_pado = g_pad + 64 * Nrow * (Ncol / 2) * 2;

     uint2 state1[3];
	 uint2 state2[3];
 

     for (int i = 0; i < Nrow; i++)
     {
         if ((i & 1) == 1)
         ST4SS(0, Ncol - i - 1, state, thread, threads);
         else
         ST4SG(0, Ncol - i - 1, state, thread, threads, offset, g_pad, g_pado);
 
         round_lyra(state);
     }
 
     for (int i = 0; i < Nrow; i+=2)
     {
         LD4SS(state1, 0, i, thread, threads);
		 LD4SG(state2, 0, i + 1, thread, threads, offset, g_pad, g_pado);
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j];
 
         round_lyra(state);
 
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state1[j] ^= state[j];
		
			 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state2[j];
 
         round_lyra(state);
 
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state2[j] ^= state[j];
         ST4SG(1, Ncol - i - 1, state1, thread, threads, offset, g_pad, g_pado);
		 ST4SS(1, Ncol - (i + 1) - 1, state2, thread, threads);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowSetup_high_end(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads, uint64_t *g_pad)
 {
     const int offset = blockIdx.x * 64 * Nrow * (Ncol / 2) * 3;
     uint64_t *g_pado = g_pad + 64 * Nrow * (Ncol / 2) * 2;

     uint2 state1[3], state2[3], state3[3], state4[3];

     for (int i = 0; i < Nrow; i+=2)
     {
         LD4SS(state1, rowIn, i, thread, threads);
		 LD4SS(state2, rowInOut, i, thread, threads);
		 LD4SG(state3, rowIn, i + 1, thread, threads, offset, g_pad, g_pado);
		 LD4SG(state4, rowInOut, i + 1, thread, threads, offset, g_pad, g_pado);
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         round_lyra(state);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state1[j] ^= state[j];
 
         ST4SG(rowOut, Ncol - i - 1, state1, thread, threads, offset, g_pad, g_pado);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data0 = state[0];
         uint2 Data1 = state[1];
         uint2 Data2 = state[2];
         WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state2[0] ^= Data2;
             state2[1] ^= Data0;
             state2[2] ^= Data1;
         } else {
             state2[0] ^= Data0;
             state2[1] ^= Data1;
             state2[2] ^= Data2;
         }
 
         ST4SS(rowInOut, i, state2, thread, threads);

		//=====================================
		 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
         round_lyra(state);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state3[j] ^= state[j];
 
         ST4SS(rowOut, Ncol - (i + 1) - 1, state3, thread, threads);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data01 = state[0];
         uint2 Data11 = state[1];
         uint2 Data21 = state[2];
         WarpShuffle3(Data01, Data11, Data21, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state4[0] ^= Data21;
             state4[1] ^= Data01;
             state4[2] ^= Data11;
         } else {
             state4[0] ^= Data01;
             state4[1] ^= Data11;
             state4[2] ^= Data21;
         }
 
         ST4SG(rowInOut, (i + 1), state4, thread, threads, offset, g_pad, g_pado);
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowt_high_end(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads, uint64_t *g_pad)
 {
     const int offset = blockIdx.x * 64 * Nrow * (Ncol / 2) * 3;
     uint64_t *g_pado = g_pad + 64 * Nrow * (Ncol / 2) * 2;

     uint2 state3[3], state4[3], state5[3];

     LD4SG(state3, rowIn, 0 + 1, thread, threads, offset, g_pad, g_pado);
     LD4SG(state4, rowInOut, 0 + 1, thread, threads, offset, g_pad, g_pado);
     LD4SG(state5, rowOut, 0 + 1, thread, threads, offset, g_pad, g_pado);

     for (int i = 0; i < Nrow; i+=2)
     {
         uint2 state1[3], state2[3], state7[3], state8[3], state9[3];
 
         LD4SS(state1, rowIn, i, thread, threads);
         LD4SS(state2, rowInOut, i, thread, threads);
		 
 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         LD4SS(state1, rowOut, i, thread, threads);

         round_lyra(state);
         LD4SG(state7, rowIn, i + 3, thread, threads, offset, g_pad, g_pado);
         LD4SG(state8, rowInOut, i + 3, thread, threads, offset, g_pad, g_pado);
         LD4SG(state9, rowOut, i + 3, thread, threads, offset, g_pad, g_pado);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data0 = state[0];
         uint2 Data1 = state[1];
         uint2 Data2 = state[2];
         WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state2[0] ^= Data2;
             state2[1] ^= Data0;
             state2[2] ^= Data1;
         }
         else
         {
             state2[0] ^= Data0;
             state2[1] ^= Data1;
             state2[2] ^= Data2;
         }

        if (rowInOut != rowOut) {
             ST4SS(rowInOut, i, state2, thread, threads);
                 #pragma unroll
            for (int j = 0; j < 3; j++)
                state2[j] = state1[j];
         }

#pragma unroll
        for (int j = 0; j < 3; j++)
            state2[j] ^= state[j];

        ST4SS(rowOut, i, state2, thread, threads);

		 //======================================
 
 
 #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];

         round_lyra(state);
 
         // simultaneously receive data from preceding thread and send data to following thread
         uint2 Data01 = state[0];
         uint2 Data11 = state[1];
         uint2 Data21 = state[2];
         WarpShuffle3(Data01, Data11, Data21, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
         if (threadIdx.x == 0)
         {
             state4[0] ^= Data21;
             state4[1] ^= Data01;
             state4[2] ^= Data11;
         }
         else
         {
             state4[0] ^= Data01;
             state4[1] ^= Data11;
             state4[2] ^= Data21;
         }

         if (rowInOut != rowOut) {
             ST4SG(rowInOut, i + 1, state4, thread, threads, offset, g_pad, g_pado);
                 #pragma unroll
            for (int j = 0; j < 3; j++)
                state4[j] = state5[j];
         }

#pragma unroll
        for (int j = 0; j < 3; j++)
            state4[j] ^= state[j];

        ST4SG(rowOut, i + 1, state4, thread, threads, offset, g_pad, g_pado);

        #pragma unroll
        for (int j = 0; j < 3; j++)
            state3[j] = state7[j];
        #pragma unroll
        for (int j = 0; j < 3; j++)
            state4[j] = state8[j];
        #pragma unroll
        for (int j = 0; j < 3; j++)
            state5[j] = state9[j];
     }
 }
 
 static __device__ __forceinline__
 void reduceDuplexRowt_8_high_end(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads, uint64_t *g_pad)
 {
     const int offset = blockIdx.x * 64 * Nrow * (Ncol / 2) * 3;
     uint64_t *g_pado = g_pad + 64 * Nrow * (Ncol / 2) * 2;

     uint2 state1[3], state2[3], state3[3], state4[3], last[3];
 
     LD4SS(state1, 2, 0, thread, threads);
     LD4SS(last, rowInOut, 0, thread, threads);
 
     #pragma unroll
     for (int j = 0; j < 3; j++)
         state[j] ^= state1[j] + last[j];
 
     round_lyra(state);
 
     // simultaneously receive data from preceding thread and send data to following thread
     uint2 Data0 = state[0];
     uint2 Data1 = state[1];
     uint2 Data2 = state[2];
     WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
 
     if (threadIdx.x == 0)
     {
         last[0] ^= Data2;
         last[1] ^= Data0;
         last[2] ^= Data1;
     } else {
         last[0] ^= Data0;
         last[1] ^= Data1;
         last[2] ^= Data2;
     }
 
     if (rowInOut == 5)
     {
         #pragma unroll
         for (int j = 0; j < 3; j++)
             last[j] ^= state[j];
     }
 
	 LD4SG(state1, 2, 1, thread, threads, offset, g_pad, g_pado);
	 LD4SG(state2, rowInOut, 1, thread, threads, offset, g_pad, g_pado);

	 #pragma unroll
	 for (int j = 0; j < 3; j++)
		 state[j] ^= state1[j] + state2[j];

	 round_lyra(state);

     for (int i = 2; i < Nrow; i+=2)
     {
         LD4SS(state1, 2, i, thread, threads);
         LD4SS(state2, rowInOut, i, thread, threads);
		 LD4SG(state3, 2, i + 1, thread, threads, offset, g_pad, g_pado);
         LD4SG(state4, rowInOut, i + 1, thread, threads, offset, g_pad, g_pado);
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state1[j] + state2[j];
 
         round_lyra(state);

		 //============================
 
         #pragma unroll
         for (int j = 0; j < 3; j++)
             state[j] ^= state3[j] + state4[j];
 
         round_lyra(state);
     }
 
     #pragma unroll
     for (int j = 0; j < 3; j++)
         state[j] ^= last[j];
 }
 // ================================= end high ==========================
 
 __constant__ uint2x4 blake2b_IV[2] = {
     0xf3bcc908lu, 0x6a09e667lu,
     0x84caa73blu, 0xbb67ae85lu,
     0xfe94f82blu, 0x3c6ef372lu,
     0x5f1d36f1lu, 0xa54ff53alu,
     0xade682d1lu, 0x510e527flu,
     0x2b3e6c1flu, 0x9b05688clu,
     0xfb41bd6blu, 0x1f83d9ablu,
     0x137e2179lu, 0x5be0cd19lu
 };
 
 __global__ __launch_bounds__(64, 1)
 void lyra2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
 {
     const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
 
     if (thread < threads)
     {
         uint2x4 state[4];
 
         state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
         state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
         state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
         state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
         state[2] = blake2b_IV[0];
         state[3] = blake2b_IV[1];
 
         for (int i = 0; i<24; i++)
             round_lyra(state); //because 12 is not enough
 
         ((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
         ((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
         ((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
         ((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
     }
 }
 
 __global__
 __launch_bounds__(64, 1)
 void lyra2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
 {
     const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;
 
     if (thread < threads)
     {
         uint2 pad[Ncol / 2][Nrow][3];
         
         uint2 state[4];
         state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
         state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
         state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
         state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);
 
         reduceDuplex(state, thread, threads, pad);
         reduceDuplexRowSetup(1, 0, 2, state, thread, threads, pad);
         reduceDuplexRowSetup(2, 1, 3, state, thread, threads, pad);
         reduceDuplexRowSetup(3, 0, 4, state, thread, threads, pad);
         reduceDuplexRowSetup(4, 3, 5, state, thread, threads, pad);
         reduceDuplexRowSetup(5, 2, 6, state, thread, threads, pad);
         reduceDuplexRowSetup(6, 1, 7, state, thread, threads, pad);

         uint32_t rowa;
         uint32_t row = 0;
         uint32_t pre = 7;
         for (int i = 0; i < 7; i++) {
            rowa = WarpShuffle(state[0].x, 0, 4) & 7;
            reduceDuplexRowt(pre, rowa, row, state, thread, threads, pad);
            pre = row;
            row = (row + 3) % 8;
         }
         rowa = WarpShuffle(state[0].x, 0, 4) & 7;
         reduceDuplexRowt_8(rowa, state, thread, threads, pad);
 
         DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
         DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
         DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
         DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
     }
 }
 
 __global__
 __launch_bounds__(64, 1)
 void lyra2_gpu_hash_32_2_biglocal(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
 {
     const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;
 
     if (thread < threads)
     {
         uint2 state[4];
         state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
         state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
         state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
         state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);
 
         reduceDuplex_biglocal(state, thread, threads);
         reduceDuplexRowSetup_biglocal(1, 0, 2, state, thread, threads);
         reduceDuplexRowSetup_biglocal(2, 1, 3, state, thread, threads);
         reduceDuplexRowSetup_biglocal(3, 0, 4, state, thread, threads);
         reduceDuplexRowSetup_biglocal(4, 3, 5, state, thread, threads);
         reduceDuplexRowSetup_biglocal(5, 2, 6, state, thread, threads);
         reduceDuplexRowSetup_biglocal(6, 1, 7, state, thread, threads);

         uint32_t rowa;
         uint32_t row = 0;
         uint32_t pre = 7;
         for (int i = 0; i < 7; i++) {
            rowa = WarpShuffle(state[0].x, 0, 4) & 7;
            reduceDuplexRowt_biglocal(pre, rowa, row, state, thread, threads);
            pre = row;
            row = (row + 3) % 8;
         }
         rowa = WarpShuffle(state[0].x, 0, 4) & 7;
         reduceDuplexRowt_8_biglocal(rowa, state, thread, threads);
 
         DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
         DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
         DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
         DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
     }
 }

__global__
 __launch_bounds__(64, 1)
 void lyra2_gpu_hash_32_2_high_end(uint32_t threads, uint32_t startNounce, uint64_t *g_pad)
 {
     const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;
 
     if (thread < threads)
     {
         uint2 state[4];
         state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
         state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
         state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
         state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);
 
         reduceDuplex_high_end(state, thread, threads, g_pad);
         reduceDuplexRowSetup_high_end(1, 0, 2, state, thread, threads, g_pad);
         reduceDuplexRowSetup_high_end(2, 1, 3, state, thread, threads, g_pad);
         reduceDuplexRowSetup_high_end(3, 0, 4, state, thread, threads, g_pad);
         reduceDuplexRowSetup_high_end(4, 3, 5, state, thread, threads, g_pad);
         reduceDuplexRowSetup_high_end(5, 2, 6, state, thread, threads, g_pad);
         reduceDuplexRowSetup_high_end(6, 1, 7, state, thread, threads, g_pad);

         uint32_t rowa;
         uint32_t row = 0;
         uint32_t pre = 7;
         for (int i = 0; i < 7; i++) {
            rowa = WarpShuffle(state[0].x, 0, 4) & 7;
            reduceDuplexRowt_high_end(pre, rowa, row, state, thread, threads, g_pad);
            pre = row;
            row = (row + 3) % 8;
         }
         rowa = WarpShuffle(state[0].x, 0, 4) & 7;
         reduceDuplexRowt_8_high_end(rowa, state, thread, threads, g_pad);
 
         DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
         DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
         DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
         DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
     }
 }

 __global__ __launch_bounds__(64, 1)
 void lyra2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
 {
     const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
 
     uint28 state[4];
 
     if (thread < threads)
     {
         state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
         state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
         state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
         state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);
 
         for (int i = 0; i < 12; i++)
             round_lyra(state);
 
         g_hash[thread + threads * 0] = state[0].x;
         g_hash[thread + threads * 1] = state[0].y;
         g_hash[thread + threads * 2] = state[0].z;
         g_hash[thread + threads * 3] = state[0].w;
 
     } //thread
 }
 #else
 #if __CUDA_ARCH__ < 500
 
 /* for unsupported SM arch */
 __device__ void* DMatrix;
 #endif
 __global__ void lyra2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
 __global__ void lyra2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash) {}
 __global__ void lyra2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
 #endif
__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}
__host__
void lyra2_cpu_init_high_end(int thr_id, uint32_t threads, uint64_t *g_pad)
{
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, bool gtx750ti, uint32_t high_end)
{
}

__host__
void lyra2_cpu_hash_32_BuBoNiK666(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, uint64_t *g_pad, bool gtx750ti, uint32_t high_end)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = TPB52;

	if (cuda_arch[dev_id] >= 520) tpb = TPB52;
	else if (cuda_arch[dev_id] >= 500) tpb = TPB50;
	else if (cuda_arch[dev_id] >= 200) tpb = TPB20;

    dim3 grid0((threads * 4 + 32 - 1) / 32);
	dim3 block0(4, 32 >> 2);

	dim3 grid1((threads * 4 + 64 - 1) / 64);
	dim3 block1(4, 64 >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	dim3 grid3((threads + tpb - 1) / tpb);
	dim3 block3(tpb);

	if (cuda_arch[dev_id] >= 520)
	{
		lyra2_gpu_hash_32_1 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

        if (high_end == 1)
            lyra2_gpu_hash_32_2_high_end <<< grid1, block1, 12 * (8 - 0) * sizeof(uint2) * 64 >>> (threads, startNounce, g_pad);
        else if (high_end == 2)
            lyra2_gpu_hash_32_2_biglocal <<< grid0, block0, 24 * (8 - 0) * sizeof(uint2) * 32 >>> (threads, startNounce, g_pad);
        else
		    lyra2_gpu_hash_32_2 <<< grid1, block1, 12 * (8 - 0) * sizeof(uint2) * 64 >>> (threads, startNounce, d_hash);

		lyra2_gpu_hash_32_3 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);
	}
	else if (cuda_arch[dev_id] >= 500)
	{
		size_t shared_mem = 0;

		if (gtx750ti)
			// suitable amount to adjust for 8warp
			shared_mem = 8192;
		else
			// suitable amount to adjust for 10warp
			shared_mem = 6144;

		lyra2_gpu_hash_32_1_sm5 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

		lyra2_gpu_hash_32_2_sm5 <<< grid1, block1, shared_mem >>> (threads, startNounce, (uint2*)d_hash);

		lyra2_gpu_hash_32_3_sm5 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);
	}
	else
		lyra2_gpu_hash_32_sm2 <<< grid3, block3 >>> (threads, startNounce, d_hash);
}
