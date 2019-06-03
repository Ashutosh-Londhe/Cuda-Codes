/*
                    column
    A[][] = ---------------------threadIdx.y
            |
            |
            |
            |
   row      |
            |
            |
            |
            |
        threadIdx.x
*/


#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TILE_WIDTH 16

#define ar 311
#define ac_br 312
#define bc 115

using namespace std;

void check_gpu_error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        cerr<<"\n Error in: "<<msg<<"  cuda error string: "<<cudaGetErrorString(err);
        exit(-1);
    }
}// End of check_gpu_error function

__global__ void mat_mul(int *d_A, int *d_B, int *d_C, int rowA, int colA, int rowB, int colB, int rowC, int colC)
{
    int row, col;
    row = threadIdx.x + blockIdx.x*blockDim.x;      // 0 to rowA/rowC
    col = threadIdx.y + blockIdx.y*blockDim.y;      // 0 to colB/colC

    if(row < rowC && col < colC)
    {
        for(int i = 0; i < colA; i++)               // colA = rowB
            d_C[row*colC + col] += d_A[row*colA + i]*d_B[i*colB + col];
    }
}// End of mat_mul function

__global__ void mat_mul_shared(int *d_A, int *d_B, int *d_C, int rowA, int colA, int rowB, int colB, int rowC, int colC)
{
    int bx = blockIdx.x,     by = blockIdx.y;
    int tx = threadIdx.x,    ty = threadIdx.y;
    int row = tx + bx*TILE_WIDTH;      // 0 to rowA/rowC
    int col = ty + by*TILE_WIDTH;      // 0 to colB/colC

    __shared__ int s_A[TILE_WIDTH][TILE_WIDTH], s_B[TILE_WIDTH][TILE_WIDTH];
    int cvalue = 0;

    for(int i = 0; i < (colA+TILE_WIDTH-1)/TILE_WIDTH; i++)
    {
        if(row < rowA && i*TILE_WIDTH+ty < colA)
            s_A[tx][ty] = d_A[row*colA + i*TILE_WIDTH+ty];
        else
            s_A[tx][ty] = 0;

        if(i*TILE_WIDTH+tx < rowB && col < colB)
            s_B[tx][ty] = d_B[(i*TILE_WIDTH+tx)*colB + col];
        else
            s_B[tx][ty] = 0;

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++)
            cvalue += s_A[tx][k]*s_B[k][ty];

        __syncthreads();
    }

    if(row < rowC && col < colC)
        d_C[row*colC + col] = cvalue;

}// End of mat_mul_shared function

int main()
{
    int *A, *B, *C1, *C2, rowA, colA, rowB, colB, rowC, colC;
    int *d_A, *d_B, *d_C;
    dim3 dimg, dimb;
    cudaEvent_t start, stop;
    float elapsed_time;

    rowA = ar;      rowC = ar;
    colA = ac_br;   rowB = ac_br;
    colB = bc;      colC = bc;

    A = new int[rowA*colA];     B  = new int[rowB*colB];
    C1 = new int[rowC*colC];    C2 = new int[rowC*colC];

    cudaMalloc((void**)&d_A, rowA*colA*sizeof(int));
    cudaMalloc((void**)&d_B, rowB*colB*sizeof(int));
    cudaMalloc((void**)&d_C, rowC*colC*sizeof(int));

    srand(time(NULL));

    for(int i = 0; i < rowA*colA; i++)
        A[i] = rand()%5;

    for(int i = 0; i < rowB*colB; i++)
        B[i] = rand()%5;

    dimg = dim3((rowC+TILE_WIDTH-1)/TILE_WIDTH, (colC+TILE_WIDTH-1)/TILE_WIDTH);
    dimb = dim3(TILE_WIDTH, TILE_WIDTH);

    cudaEventCreate(&start);        cudaEventCreate(&stop);

    cudaMemcpy(d_A, A, rowA*colA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rowB*colB*sizeof(int), cudaMemcpyHostToDevice);
    check_gpu_error("cuda memcpy host to device");
	
	// Without shared memory
    cudaMemset(d_C, 0, rowC*colC*sizeof(int));
    cudaEventRecord(start, 0);
    mat_mul<<<dimg, dimb>>>(d_A, d_B, d_C, rowA, colA, rowB, colB, rowC, colC);

    cudaEventRecord(stop, 0);           cudaEventSynchronize(stop);

    cudaMemcpy(C1, d_C, rowC*colC*sizeof(int), cudaMemcpyDeviceToHost);
    check_gpu_error("cuda memcpy device to host");

    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout<<"\n Matrix mulitplication(without shared memory): "<<elapsed_time<<"  mili-seconds";

    // With Shared memory
    cudaMemset(d_C, 0, rowC*colC*sizeof(int));
    cudaEventRecord(start, 0);
    mat_mul_shared<<<dimg, dimb>>>(d_A, d_B, d_C, rowA, colA, rowB, colB, rowC, colC);

    cudaEventRecord(stop, 0);           cudaEventSynchronize(stop);

    cudaMemcpy(C2, d_C, rowC*colC*sizeof(int), cudaMemcpyDeviceToHost);
    check_gpu_error("cuda memcpy device to host");

    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout<<"\n Matrix mulitplication(with shared memory)   : "<<elapsed_time<<"  mili-seconds";

    for(int i = 0; i < rowC*colC; i++)
    {
        if(C1[i] != C2[i])
        {
            cerr<<"\n Error!!! wrong Matrix calculation is done....";
            exit(-2);
        }
    }
    cout<<"\n Matrix mulitplication done...\n";

    return 0;
}// End of main