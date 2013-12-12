/////////////////////////////////////////////////////////////////////////////
//****************************************************************************
//
//    FILE NAME: kernel.cu
//
//    DECSRIPTION: This is the source file containing the kernel 
//                 for the HEVC encoding  
//
//    OPERATING SYSTEM: Linux UNIX only
//    TESTED ON:
//
//    CHANGE ACTIVITY:
//    Date        Who      Description
//    ==========  =======  ===============
//    12-11-2013	   Initial creation
//
//****************************************************************************
//////////////////////////////////////////////////////////////////////////////

#include<stdio.h>

#define ZERO 0 
#define ONE 1
#define TWO 2
#define MINUS -1

__global__ void hevcPredictionKernel(uint8_t *y, uint8_t *cr, uint8_t *cb, int32_t *res_y, int32_t *res_cr, int32_t *res_cb, uint8_t *y_modes, uint8_t *cr_modes, uint8_t *cb_modes, int width)
{

    int bsize = blockDim.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Thread Index to Data Index Mapping
    int col = tx + blockDim.x * bx; // Column
    int row = ty + blockDim.y * by; // Row

    // Shared neighbour memory
    int neighbourArraySize = (bsize * TWO) + ONE;

    // y is vertical array that has the extra element that is [-1][-1]
    // x is horizontal component

    // Neigbour Array for luma component
    __device__ __shared__ int p_yy[neighbourArraySize];
    __device__ __shared__ int p_xy[neighboutArraySize - ONE];

    // Neighbour array for chroma component 
    __device__ __shared__ int p_ycr[neighbourArraySize];
    __device__ __shared__ int p_ycb[neighbourArraySize];
    __device__ __shared__ int p_xcr[neighbourArraySize - ONE];
    __device__ __shared__ int p_xcb[neighbourArraySize - ONE];

    // Pointer to neighbour elements in shared memory
    int *pyy = p_yy[ONE];
    int *pxy = p_xy[ZERO];
    int *pycr = p_ycr[ONE];
    int *pxcr = p_xcr[ZERO];
    int *pycb = p_ycb[ONE];
    int *pxcb = p_xcb[ZERO];


    unsigned int fallOutside = 0;

    // This is to take care of the four corner blocks in the grid
    // OPTIMIZATION
    if ( (0 == bx && 0 == by) )
         fallOutside = 1;

    //////////////////////////////////
    // Step 1: LOAD NEIGHBOUR ELEMENTS
    //////////////////////////////////

    // Load into the shared memory from global memory
    // The loading is done based on a row basis

    // Load luma elements
    if ( ZERO == row )
    {
        // TO DO : what is bitDepthC and bitDepthY ?
        pxy[tx] = (fallOutside == 1) ? (1 << (bitDepthY -1)) : y[(row*width)+col]; // TO DO
        pxcr[tx] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : cr[(row*width)+col];
        pxcb[tx] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : cb[(row*width)+col];
    }
    else if ( ONE == row )
    {
        // TO DO : what is bitDepthC and bitDepthY ?
    	pxy[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthY)) : y[(row*width)+col];
        pxcr[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cr[(row*width)+col]);
        pxcb[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cb[(row*width)+col]);
    }
    else if ( TWO == row )
    {
        // TO DO : what is bitDepthC and bitDepthY ?
        pyy[ty] = (fallOutside == 1) ? (1 << (bitDepthY)) : y[row*width + col];
        pycr[ty] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cr[row*width + col]);
        pycb[ty] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cb[row*width + col]);
    }
    else if ( THREE == row )
    {
        pyy[ty + bsize] = (fallOutside == 1) ? (1 << (bitDepthY)) : y[row*width + col];
        pycr[ty + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cr[row*width + col]);
        pycb[ty + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cb[row*width + col]);
    }
    else
    {

    }

    // This is to load the extra guy in the neighbour element array
    // who is not filled by the threads in the current block
    if ( 0 == tx && 0 == ty ) 
    {
        if ( ! ((0 == bx) || (0 == by)) )
        {
            // this should have been pyy[MINUS]
            pyy[MINUS] = y[(row-1)*width + (col-1)];
            pycr[MINUS] = y[(row-1)*width + (col-1)];
            pycb[MINUS] = y[(row-1)*width + (col-1)];
            //pyy[ZERO] =  y[(row-1)*width + (col-1)];
            //pycr[ZERO] = cr[(row-1)*width +(col-1)];
            //pycb[ZERO] = cb[(row-1)*width +(col-1)];
        }
    }

    // Barrier Synchronization
    __syncthreads();

    //////////////////////////
    // Step 2: First Filtering
    //////////////////////////
    
    if ( ZERO == tx && ZERO == ty )
    {
        // 
        if (by==(gridDim.y-1)){
              if(bx==ZERO){
                  for(int i=0;i<neighbourArraySize;i++){
                         pyy[i]=pxy[ZERO];
                  }
                   pyy[MINUS] = pxy[ZERO];
              }
              else{
                  for(int i=bsize;i<=(2*bsize-1);i++){
                     pyy[i]=pyy[bsize-ONE];
                  }
                   pyy[MINUS] = pyy[bsize-ONE];
              }
         }
         if(0==by){
              pyy[MINUS]=pyy[ZERO];
              for(int i=0;i<2*bsize;i++){
                  pxy[i]=pyy[MINUS];
              }
         }      
         if((bx == (gridDim.x - 1)) && (0 != by))
         {
             for ( int i = bsize; i < (2 * bsize - 1); i++ )
             {
                 pxy[i] = pxy[bsize - 1];
             }
         }
    }

    
    


    

} // End of kernel function hevcPredictionKernel()
