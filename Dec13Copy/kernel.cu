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
#define THREE 3
#define MINUS -1
#define DC_MODE 0 //TO DO: check if dc mode is zero
#define PLANAR_MODE 1 //TO DO:

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
    extern __device__ __shared__ uint8_t p_yy[];
    extern __device__ __shared__ uint8_t p_xy[];

    // Neighbour array for chroma component 
    extern __device__ __shared__ uint8_t p_ycr[];
    extern __device__ __shared__ uint8_t p_ycb[];
    extern __device__ __shared__ uint8_t p_xcr[];
    extern __device__ __shared__ uint8_t p_xcb[];

    // Pointer to neighbour elements in shared memory
    uint8_t *pyy = &p_yy[ONE];
    uint8_t *pxy = &p_xy[ZERO];
    uint8_t *pycr = &p_ycr[ONE];
    uint8_t *pxcr = &p_xcr[ZERO];
    uint8_t *pycb = &p_ycb[ONE];
    uint8_t *pxcb = &p_xcb[ZERO];


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
        // TO DO : what is bitDepthC and bitDepthY ? 8 for all sizes ?
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
                         pycr[i] = pxcr[ZERO];
                         pycb[i] = pxcb[ZERO];
                  }
                   pyy[MINUS] = pxy[ZERO];
                   pycr[MINUS] = pxcr[ZERO];
                   pycb[MINUS] = pxcb[ZERO];
              }
              else{
                  for(int i=bsize;i<=(2*bsize-1);i++){
                     pyy[i]=pyy[bsize-ONE];
                     pycr[i] = pycr[bsize-ONE];
                     pycb[i] = pycb[bsize-ONE];
                  }
                   pyy[MINUS] = pyy[bsize-ONE];
                   pycr[MINUS] = pyy[bsize-ONE];
                   pycb[MINUS] = pyy[bsize-ONE];
              }
         }
         if(0==by){
              pyy[MINUS]=pyy[ZERO];
              pycr[MINUS] = pycr[ZERO];
              pycb[MINUS] = pycb[ZERO];
              for(int i=0;i<2*bsize;i++){
                  pxy[i]=pyy[MINUS];
                  pxcr[i]=pycr[MINUS];
                  pxcb[i]=pycb[MINUS];
              }
         }      
         if((bx == (gridDim.x - 1)) && (0 != by))
         {
             for ( int i = bsize; i < (2 * bsize - 1); i++ )
             {
                 pxy[i] = pxy[bsize - 1];
                 pxcr[i] = pxcr[bsize - 1];
                 pxcb[i] = pxcb[bsize - 1];
             }
         }
    }
  
  __syncthreads();
  //////////////////////////////////////////////////////////////////////////////////////////
  // done with first filteing of neighboring elements. second filtering happens inline with 
  // mode computation. 
  ////////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////////////
  // STEP 3 STARTS FROM HERE : MODE COMPUTATION AND 2ND FILTERING INLINE
  ///////////////////////////////////////////////////////////////////////////////////////////

  extern __device__ __shared__ uint8_t pf_yy[];
  extern __device__ __shared__ uint8_t pf_xy[];   
  
  __device__ __shared__ uint8_t predSamplesY[BLOCK_SIZE][BLOCK_SIZE];//TO DO: Derive this size.Ask matt
  __device__ __shared__ uint8_t predSamplesCr[BLOCK_SIZE][BLOCK_SIZE];//TO DO: Derive this size.Ask matt
  __device__ __shared__ uint8_t predSamplesCb[BLOCK_SIZE][BLOCK_SIZE];//TO DO: Derive this size.Ask matt

  uint8_t *pfyy = &pf_yy[ONE];
  uint8_t *pfxy = &pf_xy[ZERO];
   
  for(int mode =0;mode <35;i++){
  // if the  computed value of  filterFlag==1, use the filtered array pF instead of p for intra prediction.
           int filterFlag=0;
           int biIntFlag= 0;
   
           if(ty==0 && tx==0){ 
                      if(mode==DC_MODE || bsize==4){
                                    filterFlag=1;
                      }
                      else{
  	          	 //TO_DO Check if abs can be called in GPU Kernel
                 	 int minDistVerHor=min(abs(mode-26),abs(mode-10));
		  	 int intraHorVerDistThres;
   			 if(bsize==8){
        			intraHorVerDistThres=7;
         		 }
      			 else if(bsize==16){
       		        	intraHorVerDistThres=1;
   		         }
  		         else if(bsize==32){
       			 	intraHorVerDistThres=0;
  		         }
		         if(minDistVerHor>intraHorVerDistThres){
        	                filterFlag=1;
   		         }
                      }
           
   		     if(filterFlag==1){
                               if(bsize==32 && ( abs ( pyy[-1] + pxy[bsize*2-1] - (2*pxy[bsize-1]) ) < (1<<(BITDEPTHy-5) ) ) && ( abs ( pyy[-1] + pyy[bsize*2-1] - (2*pyy[bsize-1]) ) < (1<<(BITDEPTHy-5) ) )){
                                      biIntFlag=1;
                               }
                     }
     
                     if(biIntFlag==1){
                              pfyy[MINUS]=pyy[MINUS];
                              for(int i=0;i<(bsize*2-2);i++){
                                       pfyy[i]=((63-i)*pyy[MINUS]+(i+1)*pyy[63]+32)>>6; 
                              }
                              pfyy[63]=pyy[63];
                              for(int i=0;i<(bsize*2-2);i++){
                                       pfxy[i]=((63-i)*pyy[MINUS]+(i+1)*pxy[63]+32)>>6;
                              }
                              pfxy[63]=pxy[63];
                     }

                     else{
                              pfyy[MINUS]=(pyy[ZERO]+2*pyy[MINUS]+pxy[ZERO]+2)>>2;
                              for(int i=0;i<(bsize*2-2);i++){
                                      pfyy[i]=(pyy[i+1]+2*pyy[i]+pyy[i-1]+2)>>2;
                              }
                              pfyy[bsize*2-1]=pyy[bsize*2-1];
                              for(int i=0;i<(bsize*2-2);i++){
                                      pfxy[i]=(pxy[i-1]+2*pxy[i]+pxy[i+1]+2)>>2;
                              }
                              pfxy[bsize*2-1]=pxy[bsize*2-1];
                    }
           }
  }  

  __syncthreads();

    if(mode==PLANAR_MODE){
         predSamplesY[tx][ty]=((bsize-1-tx)*Pyy[ty]+(tx+1)*Pxy[bsize]+(bsize-1-ty)*Pxy[tx]+(ty+1)*Pyy[bsize]+bsize)>>log2(bsize+1); //TO_DO: Replace logarithmic with appropriate C function    
         predSamplesCr[tx][ty]=((bsize-1-tx)*Pycr[ty]+(tx+1)*Pxcr[bsize]+(bsize-1-ty)*Pxcr[tx]+(ty+1)*Pycr[bsize]+bsize)>>log2(bsize+1); //TO_DO: Replace logarithmic with appropriate C function    
         predSamplesCb[tx][ty]=((bsize-1-tx)*Pycb[ty]+(tx+1)*Pxcb[bsize]+(bsize-1-ty)*Pxcb[tx]+(ty+1)*Pycb[bsize]+bsize)>>log2(bsize+1); //TO_DO: Replace logarithmic with appropriate C function    
    
    }

    

} // End of kernel function hevcPredictionKernel()
