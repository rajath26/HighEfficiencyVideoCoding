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

#include <stdio.h>
#include <math.h>

#define ZERO 0 
#define ONE 1
#define TWO 2
#define THREE 3
#define MINUS -1
#define DC_MODE 1 
#define PLANAR_MODE 0 
#define BITDEPTHY 8
#define BITDEPTHC 8
#define ANGULAR_18 18
#define ANGULAR_26 26
#define  ANGULAR_10 10
#define TOTAL_MODES 35
#define MAX_BLOCK_SIZE 32
#define IA_MODES 16
#define BITS_PER_SUM (8 * sizeof(sum_t))

#define HADAMARD4(d0, d1, d2, d3, s0, s1, s2, s3) { \
         sum2_t t0 = s0 + s1; \
         sum2_t t1 = s0 - s1; \
         sum2_t t2 = s2 + s3; \
         sum2_t t3 = s2 - s3; \
         d0 = t0 + t2; \
         d2 = t0 - t2; \
         d1 = t1 + t3; \
         d3 = t1 - t3; \
}

#define abs(x) ( ( (x) < 0 ) ? -(x) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

//////////////////
// CONSTANT MEMORY
//////////////////
__device__ __constant__ int ipa[TOTAL_MODES] = {0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32};
__device__ __constant__ int ia[IA_MODES] = {-4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096};

__device__ int sumArray(uint8_t *array, uint8_t start, uint8_t end)
{

    int result;

    for ( int counter = start; counter < end; counter++ )
        result += array[counter];

    return result;

} // End of sumArray()

__device__ uint8_t clip3(uint8_t x, uint8_t y, uint8_t z)
{

    if ( z < x )
        return x;
    else if ( z > y )
        return y;
    else
        return z;

} // End of clip3()

__device__ uint8_t clip1Y(uint8_t x)
{

    uint8_t ret = clip3(0, ( 1 << BITDEPTHY ) - 1, x);

    return ret;

} // End of clip1Y()

__device__ sum2_t abs2(sum2_t a)
{
    sum2_t s = ((a >> (BITS_PER_SUM - 1)) & (((sum2_t)1 << BITS_PER_SUM) + 1)) * ((sum_t)-1);
    return (a + s) ^ s;
}

__device__ void sort(int32_t*  input_values)
{
        for(int i =0;i<TOTAL_MODES;i++)
        {
            int j=i;
            while(j>0 && input_values[j] < input_values[j-1])
            {
                int32_t temp=input_values[j];
                input_values[j]=input_values[j-1];
                input_values[j-1]=temp;
                j--;
            }
        }    
} // End of sort()

__device__ void extract(int32_t *sorted_values, int32_t *res, uint8_t *modes)
{
   for ( int counter = 0; counter < TOTAL_MODES; counter++)
   {
       uint8_t mode = sorted_values[counter] >> 8 & 0XFF;
       int32_t value = sorted_values[counter] >> 8;
       res[counter] = value;
       modes[counter] = mode;
   }
} // End of extract()

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////  KERNEL FUNCTION  /////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
/*
__global__ void hevcPredictionKernel(uint8_t *y, uint8_t *cr, uint8_t *cb, int32_t *res_y, int32_t *res_cr, int32_t *res_cb, uint8_t *y_modes, uint8_t *cr_modes, uint8_t *cb_modes, int height, int width)
{
 printf("\nYUP I AM HERE\n");

}
*/

__global__ void hevcPredictionKernel(uint8_t *y, uint8_t *cr, uint8_t *cb, int32_t *res_y, int32_t *res_cr, int32_t *res_cb, uint8_t *y_modes, uint8_t *cr_modes, uint8_t *cb_modes, int height, int width)
{

    // Thread indices, Block Indices and Dimensions
    uint8_t bsize = blockDim.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Thread Index to Data Index Mapping
    int col = tx + blockDim.x * bx; 
    int row = ty + blockDim.y * by;

    if ( 0 == tx && 0 == ty  && row == 0 && col == 0)
        printf("\n YUP I AM HERE \n");

    // Shared neighbour memory
    int neighbourArraySize = (bsize * TWO) + ONE;

    int bitDepthY=BITDEPTHY;
    int bitDepthC=BITDEPTHC;

    int rowToBeLoaded=0;
    int colToBeLoaded=0;
    int var = 3;
    int var1 =  3;
    
    /////////
    // Neighbour Array
    ////////
    // y is vertical array that has the extra element that is [-1][-1]
    // x is horizontal component

    // Neigbour Array for luma component
    __device__ __shared__ uint8_t p_yy[MAX_BLOCK_SIZE*2+1];
    __device__ __shared__ uint8_t p_xy[MAX_BLOCK_SIZE*2+1];

    // Neighbour array for chroma component 
    __device__ __shared__ uint8_t p_ycr[MAX_BLOCK_SIZE*2+1];
    __device__ __shared__ uint8_t p_ycb[MAX_BLOCK_SIZE*2+1];
    __device__ __shared__ uint8_t p_xcr[MAX_BLOCK_SIZE*2+1];
    __device__ __shared__ uint8_t p_xcb[MAX_BLOCK_SIZE*2+1];
   
    // Pointer to neighbour elements in shared memory
    uint8_t *pyy = &p_yy[ONE];
    uint8_t *pxy = &p_xy[ZERO];
    uint8_t *pycr = &p_ycr[ONE];
    uint8_t *pxcr = &p_xcr[ZERO];
    uint8_t *pycb = &p_ycb[ONE];
    uint8_t *pxcb = &p_xcb[ZERO];

    // Points to the righ top most block for which all
    // the neighbour elements fall outside the image boundaries
    unsigned int fallOutside = 0;
    
    // This is to take care of the top right corner blocks in the grid
    // OPTIMIZATION
    if ( (0 == bx && 0 == by) )
         fallOutside = 1;

    /// DEBUG
    //if ( fallOutside )
        //printf("\nI AM FALLING OUTSIDE\n");

    /// DEBUG
    /*
    if ( blockIdx.x == 0  && by == 0 && tx == 0 && ty == 0 )
    {
    printf("\nINPUT MATRIX WIDTH: %d HEIGHT: %d\n", width, height);
    for ( int i = 0 ; i < width; i++)
    {
        for (int j = 0; j < height; j++ )
        {
             printf("\t%u", y[i*width+j]);
        }
        printf("\n");
    }
    }

    __syncthreads();
    */
    
    //////////////////////////////////
    //////////////////////////////////
    // Step 1: LOAD NEIGHBOUR ELEMENTS
    //////////////////////////////////
    //////////////////////////////////

    // Load into the shared memory from global memory
    // The loading is done based on a row basis

    // Load luma elements
    if ( ZERO == ty )
    {
        rowToBeLoaded=row-1;
        colToBeLoaded=col;

        /// DEBUG
        /*
        if ( var == bx && var1 == by )
           printf("\nRow: %d col: %d rowTO: %d colTO: %d\n", row, col, rowToBeLoaded, colToBeLoaded);
        */

        if((rowToBeLoaded>=0 && rowToBeLoaded<height && colToBeLoaded>=0 && colToBeLoaded<width) || fallOutside)
        { 
            pxy[tx] = (fallOutside == 1) ? (1 << (bitDepthY -1)) : y[(rowToBeLoaded*width)+colToBeLoaded];  
            pxcr[tx] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : cr[(rowToBeLoaded*width)+colToBeLoaded];
            pxcb[tx] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : cb[(rowToBeLoaded*width)+colToBeLoaded];
        }
    }
    else if ( ONE == ty )
    {
        rowToBeLoaded=row-2;
        colToBeLoaded=col+blockDim.x;

        /// DEBUG
        /*
        if ( var == bx && var1 == by )
           printf("\nRow: %d col: %d rowTO: %d colTO: %d\n", row, col, rowToBeLoaded, colToBeLoaded);
        */

        if((rowToBeLoaded>=0 && rowToBeLoaded<height && colToBeLoaded>=0 && colToBeLoaded<width) || fallOutside)
        { 
    	    pxy[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthY - 1)) : y[(rowToBeLoaded*width)+colToBeLoaded];
            pxcr[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cr[(rowToBeLoaded*width)+colToBeLoaded]);
            pxcb[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cb[(rowToBeLoaded*width)+colToBeLoaded]);
        }
    }
    else if ( TWO == ty )
    {
        rowToBeLoaded=(row-2)+tx;
        colToBeLoaded=blockDim.x*blockIdx.x-1;

        /// DEBUG
        /*
        if ( var == bx && var1 == by )
           printf("\nRow: %d col: %d rowTO: %d colTO: %d\n", row, col, rowToBeLoaded, colToBeLoaded);
        */

        if((rowToBeLoaded>=0 && rowToBeLoaded<height && colToBeLoaded>=0 && colToBeLoaded<width) || fallOutside)
        { 
            pyy[tx] = (fallOutside == 1) ? (1 << (bitDepthY - 1)) : y[rowToBeLoaded*width + colToBeLoaded];
            pycr[tx] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cr[rowToBeLoaded*width + colToBeLoaded]);
            pycb[tx] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cb[rowToBeLoaded*width + colToBeLoaded]);
        }
    }
    else if ( THREE == ty )
    {
        rowToBeLoaded=(row+1)+tx;
        colToBeLoaded=blockIdx.x*blockDim.x-1;

        /// DEBUG
        /*
        if ( var == bx && var1 == by )
           printf("\nRow: %d col: %d rowTO: %d colTO: %d\n", row, col, rowToBeLoaded, colToBeLoaded);
        */

        if((rowToBeLoaded>=0 && rowToBeLoaded<height && colToBeLoaded>=0 && colToBeLoaded<width) || fallOutside)
        { 
            pyy[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthY - 1)) : y[rowToBeLoaded*width + colToBeLoaded];
            pycr[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cr[rowToBeLoaded*width + colToBeLoaded]);
            pycb[tx + bsize] = (fallOutside == 1) ? (1 << (bitDepthC - 1)) : (cb[rowToBeLoaded *width + colToBeLoaded]);
        }
    }
    else
    {
        // Nothing to do here
    }
    
    // This is to load the extra guy in the neighbour element array
    // who is not filled by the threads in the current block
    // i.e. the extra element in the pyy, pycr, pycb array
    if ( 0 == tx && 0 == ty ) 
    {
        if ( ! ((0 == bx) || (0 == by)) )
        {
            // this should have been pyy[MINUS]
            rowToBeLoaded=row-1;
            colToBeLoaded=col-1;
            
            if(rowToBeLoaded>=0 && rowToBeLoaded<height && colToBeLoaded>=0 && colToBeLoaded<width)
            { 
                pyy[MINUS] = y[(rowToBeLoaded-1)*width + (colToBeLoaded-1)];
                pycr[MINUS] = y[(rowToBeLoaded-1)*width + (colToBeLoaded-1)];
                pycb[MINUS] = y[(rowToBeLoaded-1)*width + (colToBeLoaded-1)];
            }
        } // End of if ( ! ((0 == bx) || (0 == by)) )
        if ( fallOutside) 
        {
            pyy[MINUS] = 1 << (bitDepthY - 1);
            pycr[MINUS] = 1 << (bitDepthC - 1);
            pycb[MINUS] = 1 << (bitDepthC - 1);
        }
    } // End of if ( 0 == tx && 0 == ty )

    __syncthreads();

    /// DEBUG
    /*
    if ( blockIdx.x == var && blockIdx.y == var1 && tx == 0 && ty == 0 )
    {
    printf("\nPREDICTED MATRIX - PYY\n");
    for ( int i = 0 ; i < 2*bsize+1; i++)
    {
             printf("\t%u", p_yy[i]);
    }
    printf("\nPREDICTED MATRIX - PXY\n");
    for ( int i = 0 ; i < 2*bsize; i++)
    {
             printf("\t%u", p_xy[i]);
    }
    }
    */
 

 
    //////////////////////////
    //////////////////////////
    // Step 2: First Filtering
    //////////////////////////
    //////////////////////////
    
    if ( ZERO == tx && ZERO == ty )
    {

        if (by==(gridDim.y-1))
        {
            if(bx==ZERO)
            {
                for(int i=0;i<neighbourArraySize-1;i++)
                {
                    pyy[i]=pxy[ZERO];
                    pycr[i] = pxcr[ZERO];
                    pycb[i] = pxcb[ZERO];
                }
                pyy[MINUS] = pxy[ZERO];
                pycr[MINUS] = pxcr[ZERO];
                pycb[MINUS] = pxcb[ZERO];
            }
            else
            {
                for(int i=bsize;i<(2*bsize);i++)
                {
                    pyy[i]=pyy[bsize-ONE];
                    pycr[i] = pycr[bsize-ONE];
                    pycb[i] = pycb[bsize-ONE];
                }
            }
         } // End of if (by==(gridDim.y-1))
         if(0==by && !fallOutside)
         {
             pyy[MINUS]=pyy[ZERO];
             pycr[MINUS] = pycr[ZERO];
             pycb[MINUS] = pycb[ZERO];
             for(int i=0;i<2*bsize;i++)
             {
                 pxy[i]=pyy[MINUS];
                 pxcr[i]=pycr[MINUS];
                 pxcb[i]=pycb[MINUS];
             }
         } // End of if ( 0 == by )
         if((bx == (gridDim.x - 1)) && (0 != by))
         {
             for ( int i = bsize; i < (2 * bsize); i++ )
             {
                 pxy[i] = pxy[bsize - 1];
                 pxcr[i] = pxcr[bsize - 1];
                 pxcb[i] = pxcb[bsize - 1];
             }
         }
    } // End of if ( ZERO == tx && ZERO == ty )
  
    __syncthreads();

    /// DEBUG
    /*
    if ( blockIdx.x == var && blockIdx.y == var1 && tx == 0 && ty == 0 )
    {
    printf("\nPREDICTED MATRIX - PYY\n");
    for ( int i = 0 ; i < 2*bsize+1; i++)
    {
             printf("\t%u", p_yy[i]);
    }
    printf("\nPREDICTED MATRIX - PXY\n");
    for ( int i = 0 ; i < 2*bsize; i++)
    {
             printf("\t%u", p_xy[i]);
    }
    }
    */


    /////////////////////////////////////////////////
    /////////////////////////////////////////////////
    // STEP 3 : MODE COMPUTATION AND SECOND FILTERING
    /////////////////////////////////////////////////
    /////////////////////////////////////////////////

    // TO DO
    /////////
    // Second Filtered neighbour array
    /////////
    __device__ __shared__ uint8_t pf_yy[MAX_BLOCK_SIZE*2+1];
    __device__ __shared__ uint8_t pf_xy[MAX_BLOCK_SIZE*2+1];   
  
    ////////
    // Predicted pixels
    ///////
    __device__ __shared__ uint8_t predSamplesY[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
    __device__ __shared__ uint8_t predSamplesCr[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
    __device__ __shared__ uint8_t predSamplesCb[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

    // Pointer to predicted pixels
    uint8_t *pfyy = &pf_yy[ONE];
    uint8_t *pfxy = &pf_xy[ZERO];

    //////
    // Hadamard shared memory
    //////
    __device__ __shared__ uint8_t ay[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
    __device__ __shared__ uint8_t acr[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
    __device__ __shared__ uint8_t acb[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
    __device__ __shared__ uint8_t hby[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE/2];
    __device__ __shared__ uint8_t bcr[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE/2];
    __device__ __shared__ uint8_t bcb[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE/2];
    __device__ __shared__ int32_t y_satd_shared[TOTAL_MODES];
    __device__ __shared__ int32_t cr_satd_shared[TOTAL_MODES];
    __device__ __shared__ int32_t cb_satd_shared[TOTAL_MODES];
    __device__ __shared__ int32_t y_modes_shared[TOTAL_MODES];
    __device__ __shared__ int32_t cr_modes_shared[TOTAL_MODES];
    __device__ __shared__ int32_t cb_modes_shared[TOTAL_MODES];
   
    
    // Loop through all modes
    for(int mode =0;mode <35;mode++)
    {
        // if the  computed value of  filterFlag==1, use the filtered array pF instead of p for intra prediction.
        int filterFlag=0;
        int biIntFlag= 0;
   
        if(ty==0 && tx==0)
        { 
            //////////////
            // FILTER FLAG
            //////////////
            if(mode==DC_MODE || bsize==4)
            {
                filterFlag=0;
            }
            else
            {
                int minDistVerHor = min(abs(mode-26),abs(mode-10));
                int intraHorVerDistThres;
   			 
                if(bsize==8)
                {
                    intraHorVerDistThres=7;
                }
                else if(bsize==16)
                {
       		    intraHorVerDistThres=1;
                }
  		else if(bsize==32)
                {
       	            intraHorVerDistThres=0;
                }
                else
                {
                    // Nothing to do`
                }
		if(minDistVerHor>intraHorVerDistThres)
                {
        	    filterFlag=1;
                }
                else
                {
                    filterFlag = 0;
                }
            } // End of else of if ( mode == DC_MODE || bsize == 4 )
           
            if(filterFlag==1)
            {
                /////////////
                // B INT FLAG
                /////////////
                if(bsize==32 && ( abs ( pyy[-1] + pxy[bsize*2-1] - (2*pxy[bsize-1]) ) < (1<<(bitDepthY-5) ) ) && ( abs ( pyy[-1] + pyy[bsize*2-1] - (2*pyy[bsize-1]) ) < (1<<(bitDepthY-5) ) ))
                {
                    biIntFlag=1;
                }
                else
                {
                    biIntFlag = 0;
                }
            } // End of if ( 1 == filterFlag )

     
            ///////////////////
            // SECOND FILTERING
            ///////////////////
            if(biIntFlag==1)
            {
                pfyy[MINUS]=pyy[MINUS];
                for(int i=0;i<(bsize*2-1);i++)
                {
                    pfyy[i]=((63-i)*pyy[MINUS]+(i+1)*pyy[63]+32)>>6; 
                }
                pfyy[63]=pyy[63];
                for(int i=0;i<(bsize*2-1);i++)
                {
                    pfxy[i]=((63-i)*pyy[MINUS]+(i+1)*pxy[63]+32)>>6;
                }
                pfxy[63]=pxy[63];
            } // End of if ( 1 == biIntFlag )
            else
            {
                pfyy[MINUS]=(pyy[ZERO]+2*pyy[MINUS]+pxy[ZERO]+2)>>2;
                for(int i=0;i<(bsize*2-1);i++)
                {
                    pfyy[i]=(pyy[i+1]+2*pyy[i]+pyy[i-1]+2)>>2;
                }
                pfyy[bsize*2-1]=pyy[bsize*2-1];
                pfxy[0] = (pyy[MINUS] + 2 * pxy[ZERO] + pxy[ONE] + 2) >> 2;
                for(int i=1;i<(bsize*2-1);i++)
                {
                    pfxy[i]=(pxy[i-1]+2*pxy[i]+pxy[i+1]+2)>>2;
                }
                pfxy[bsize*2-1]=pxy[bsize*2-1];
           } // End of else of if ( 1 -- biIntFlag )

       } // End of if(ty==0 && tx==0)
    
        __syncthreads();

/*
        //////////////
        // Switch pointer to pfyy or p_yy
        // Switch pointer to pfxy or p_xy
        /////////////
        uint8_t *selyy, *selxy; 
        if(filterFlag==1)
        {
            selyy=&pf_yy[ONE];
            selxy=&pf_xy[ZERO];
        }
        else
        {
            selyy=pyy;
            selxy=pxy;
        }   

        __device__ __shared__ uint8_t ref_Y[3*MAX_BLOCK_SIZE+1];
        __device__ __shared__ uint8_t ref_Cr[3*MAX_BLOCK_SIZE+1];
        __device__ __shared__ uint8_t ref_Cb[3*MAX_BLOCK_SIZE+1];

        // Pointer to ref arrays
        uint8_t *refY = &ref_Y[4];
        uint8_t *refCr = &ref_Cr[4];
        uint8_t *refCb = &ref_Cb[4];

        // OPTIMIZATION making iIdx and IFact as matrices
        __device__ __shared__ int iIdx[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
        __device__ __shared__ int iFact[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];


        ////////////////////
        // MODE: PLANAR MODE
        ////////////////////
        // TO DO : is this ty tx
        if(mode==PLANAR_MODE)
        {

            float logValue = log2f(bsize+1.0);
            int intLog = (int) logValue;

            predSamplesY[tx][ty]=((bsize-1-tx)*selyy[ty]+(tx+1)*selxy[bsize]+(bsize-1-ty)*selxy[tx]+(ty+1)*selyy[bsize]+bsize)>>intLog; //TO_DO: Replace logarithmic with appropriate C function    
            predSamplesCr[tx][ty]=((bsize-1-tx)*pycr[ty]+(tx+1)*pxcr[bsize]+(bsize-1-ty)*pxcr[tx]+(ty+1)*pycr[bsize]+bsize)>>intLog; //TO_DO: Replace logarithmic with appropriate C function    
            predSamplesCb[tx][ty]=((bsize-1-tx)*pycb[ty]+(tx+1)*pxcb[bsize]+(bsize-1-ty)*pxcb[tx]+(ty+1)*pycb[bsize]+bsize)>>intLog; //TO_DO: Replace logarithmic with appropriate C function    
    
        }

        ////////////////
        // MODE: DC MODE
        ////////////////
        else if ( DC_MODE == mode )
        {

            uint8_t dcValY = 0;
            uint8_t dcValCr = 0;
            uint8_t dcValCb = 0;
 
            uint8_t firstSumY = 0;
            uint8_t secondSumY = 0;
            uint8_t firstSumCr = 0;
            uint8_t secondSumCr = 0;
            uint8_t firstSumCb = 0;
            uint8_t secondSumCb = 0;

            if ( 0 == tx && 0 == ty )
            {
                firstSumY = sumArray(selxy, 0, bsize - 1);
            }
            else if ( 1 == tx && 0 == ty )
            {
                secondSumY = sumArray(selyy, 0, bsize - 1);
            }
            else if ( 2 == tx && 0 == ty ) 
            {
                firstSumCr = sumArray(pxcr, 0, bsize - 1);
            } 
            else if ( 3 == tx && 0 == ty )
            {
                secondSumCr = sumArray(pycr, 0, bsize - 1);
            }
            else if ( 4 == tx && 0 == ty )
            {
                firstSumCr = sumArray(pxcb, 0, bsize - 1);
            } 
            else if ( 5 == tx && 0 == ty )
            {
                secondSumCr = sumArray(pycb, 0, bsize - 1);
            }

            __syncthreads(); 
 
            if ( 0 == tx && 0 == ty )
            {
                dcValY = (firstSumY + secondSumY + bsize) >> ((int)log2f((float)bsize)+1);
            }

            else if ( 1 == tx && 0 == ty )
            {
                dcValCr = (firstSumCr + secondSumCr + bsize) >> ((int)log2f((float)bsize)+1);
            }

            else if ( 2 == tx && 0 == ty )
            {
                dcValCb = (firstSumCb + secondSumCb + bsize) >> ((int)log2f((float)bsize)+1);
            }

            __syncthreads();

            if ( bsize < 32 )
            {
                //Apply following changes to predSamples only for luma channel
                predSamplesY[0][0]=(selyy[ZERO]+2*dcValY+selxy[0]+2)>>2;
           
                for(int i=1;i<bsize;i++)
                {
            	    predSamplesY[i][0]+=(selyy[i]+3*dcValY+2)>>2;
                }  
            
                for(int i=1;i<bsize;i++)
                {
            	    predSamplesY[0][i]+=(selxy[i]+3*dcValY+2)>>2;
                } 

                for(int i=1;i<bsize;i++)
                {
                    for(int j=1;j<bsize;j++)
                    {
                        predSamplesY[i][j]=dcValY;
                    }
                }
            } // End of if ( bsize < 32 )
            else 
            {
                //For cr and cb, set dcValue as all value for predSamples of cr and cb  
                for(int i=0;i<bsize;i++)
                {
                   for(int j=0;j<bsize;j++)
                   {
                      if ( bsize == 32 )
                          predSamplesY[i][j] = dcValY;
                      predSamplesCr[i][j]=dcValCr;
                      predSamplesCb[i][j]=dcValCb;
                   }
                }  
            } // End of else of if ( bsize < 32 )

        } // End of else if ( DC_MODE == mode )

        ///////////////
        // ANGULAR MODE
        ///////////////

        else if ( mode >= ANGULAR_18 )
        {

            // OPTIMIZATION 
            if ( bsize == 4 )
            {
                if ( 0 == ty ) 
                {
                    refY[tx] = selxy[-1 + tx];
                    refCr[tx] = pxcr[-1 + tx]; 
                    refCb[tx] = pxcb[-1 + tx];
                    if ( 0 == tx )
                    {
                        refY[bsize+tx] = selxy[-1 + (tx + bsize)];
                        refCr[bsize+tx] = pxcr[-1 + (tx + bsize)];
                        refCb[bsize+tx] = pxcb[-1 + (tx + bsize)];
                    }
                }

                if (ipa[mode] < 0) 
                {
                    if ( ((bsize * ipa[mode]) >> 5) < -1 )
                    {
                        if ( 1 == ty )
                        {
                            refY[-(tx + 1)] = selyy[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                        if ( 2 == ty ) 
                        {
                            refCr[-(tx + 1)] = pycr[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                            refCb[-(tx + 1)] = pycb[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        } 
                    } // End of if ( ((bsize * ipa[mode]) >> 5) < -1 )
                } // End of if (ipa[mode] < 0)
                else 
                {
                    if ( 3 == ty )
                    {
                        refY[tx + bsize + 1] = selxy[-1 + tx + bsize + 1];
                        refCr[tx + bsize + 1] = pxcr[-1 + tx + bsize + 1];
                        refCb[tx + bsize + 1] = pxcb[-1 + tx + bsize + 1];
                    }
                } // End of else of if (ipa[mode] < 0)

            } // End of if ( bsize == 4 )
            else 
            {
                if ( 0 == ty ) 
                {
                    refY[tx] = selxy[-1 + tx];
                    if ( 0 == tx )
                        refY[bsize + tx] = selxy[-1 + (tx + bsize)];
                }
                if ( 1 == ty ) 
                {
                    refCr[tx] = pxcr[-1 + tx];
                    if ( 0 == tx )
                        refCr[bsize+tx] = pxcr[-1 + (tx + bsize)];
                } 
                if ( 2 == ty ) 
                {
                    refCb[tx] = pxcb[-1 + tx];
                    if ( 0 == tx )
                        refCb[bsize+tx] = pxcb[-1 + (tx + bsize)];
                } 
                if (ipa[mode] < 0)
                {
                    if ( ((bsize * ipa[mode]) >> 5) < -1 )
                    {
                        if ( 3 == ty )
                        {
                            refY[-(tx + 1)] = selyy[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                        if ( 4 == ty )
                        {
                            refCr[-(tx + 1)] = pycr[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                        if ( 5 == ty )
                        {
                            refCb[-(tx + 1)] = pycb[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                    } // End of if ( ((bsize * ipa[mode]) >> 5) < -1 )
                } // End of if (ipa[mode] < 0)
                else
                {
                    if ( 6 == ty )
                        refY[tx + bsize + 1] = selxy[-1 + tx + bsize + 1];
                    if ( 7 == ty )
                    {
                        refCr[tx + bsize + 1] = pxcr[-1 + tx + bsize + 1];
                        if ( bsize == 8 )
                            refCb[tx + bsize + 1] = pxcb[-1 + tx + bsize + 1];
                    }
                    if ( 8 == ty && bsize != 8 )
                    {
                        refCb[tx + bsize + 1] = pxcb[-1 + tx + bsize + 1];
                    }
                } // End of else of if (ipa[mode] < 0)
                
            } // End of else of if ( bsize == 4 )

            // Load iIdx and iFact
            iIdx[ty][tx] = ((ty+1) * ipa[mode]) >> 5;
            iFact[ty][tx] = ((ty+1) * ipa[mode]) & 31;

            if ( iFact[ty][tx] != 0 )
            {
                predSamplesY[ty][tx] = ((32 - iFact[ty][tx]) * refY[tx + iIdx[ty][tx] + 1] + iFact[ty][tx] * refY[tx + iIdx[ty][tx] + 2] + 16) >> 5;
                predSamplesCr[ty][tx] = ((32 - iFact[ty][tx]) * refCr[tx + iIdx[ty][tx] + 1] + iFact[ty][tx] * refCr[tx + iIdx[ty][tx] + 2] + 16) >> 5;
                predSamplesCb[ty][tx] = ((32 - iFact[ty][tx]) * refCb[tx + iIdx[ty][tx] + 1] + iFact[ty][tx] * refCb[tx + iIdx[ty][tx] + 2] + 16) >> 5;
            } 
            else
            {
                predSamplesY[ty][tx] = refY[tx + iIdx[ty][tx] + 1];
                predSamplesCr[ty][tx] = refCr[tx + iIdx[ty][tx] + 1];
                predSamplesCb[ty][tx] = refCb[tx + iIdx[ty][tx] + 1];
            }

            if ( mode == ANGULAR_26 && bsize < 32 )
            {
                if ( 0 == tx ) 
                {
                    uint8_t param = selxy[tx] + ((selyy[ty] - selyy[MINUS]) >> 1);
                    predSamplesY[ty][tx] = clip1Y(param);
                }

                __syncthreads();

            } // End of if ( mode == ANGULAR_26 && bsize < 32 )

        } // End of else if ( mode >= ANGULAR_18 )

        else if ( mode > DC_MODE && mode < ANGULAR_18 )
        {
            if ( 4 == bsize )
            {

                if ( 0 == ty )
                {
                    refY[tx] = selyy[-1 + tx];
                    refCr[tx] = pycr[-1 + tx];
                    refCb[tx] = pycb[-1 + tx];
                    if ( 0 == tx )
                    {
                        refY[bsize+tx] = selyy[-1 + (tx + bsize)];
                        refCr[bsize+tx] = pycr[-1 + (tx + bsize)];
                        refCb[bsize+tx] = pycb[-1 + (tx + bsize)];
                    }
                }

                if (ipa[mode] < 0)
                {
                    if ( ((bsize * ipa[mode]) >> 5) < -1 )
                    {
                        if ( 1 == ty )
                        {
                            refY[-(tx + 1)] = selxy[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                        if ( 2 == ty )
                        {
                            refCr[-(tx + 1)] = pxcr[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                            refCb[-(tx + 1)] = pxcb[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                    } // End of if ( ((bsize * ipa[mode]) >> 5) < -1 )
                } // End of if (ipa[mode] < 0)
                else
                {
                    if ( 3 == ty )
                    {
                        refY[tx + bsize + 1] = selyy[-1 + tx + bsize + 1];
                        refCr[tx + bsize + 1] = pycr[-1 + tx + bsize + 1];
                        refCb[tx + bsize + 1] = pycb[-1 + tx + bsize + 1];
                    }
                } // End of else of if (ipa[mode] < 0)

            } // End of if ( 4 == bsize )
            else
            {
 
                if ( 0 == ty )
                {
                    refY[tx] = selyy[-1 + tx];
                    if ( 0 == tx )
                        refY[bsize + tx] = selyy[-1 + (tx + bsize)];
                }
                if ( 1 == ty )
                {
                    refCr[tx] = pycr[-1 + tx];
                    if ( 0 == tx )
                        refCr[bsize+tx] = pycr[-1 + (tx + bsize)];
                }
                if ( 2 == ty )
                {
                    refCb[tx] = pycb[-1 + tx];
                    if ( 0 == tx )
                        refCb[bsize+tx] = pycb[-1 + (tx + bsize)];
                }
                if (ipa[mode] < 0)
                {
                    if ( ((bsize * ipa[mode]) >> 5) < -1 )
                    {
                        if ( 3 == ty )
                        {
                            refY[-(tx + 1)] = selxy[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                        if ( 4 == ty )
                        {
                            refCr[-(tx + 1)] = pxcr[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                        if ( 5 == ty )
                        {
                            refCb[-(tx + 1)] = pxcb[ -1 + (( tx + 1 ) * ia[mode-11] + 128) >> 8];
                        }
                    } // End of if ( ((bsize * ipa[mode]) >> 5) < -1 )
                } // End of if (ipa[mode] < 0)
                else
                {
                    if ( 6 == ty )
                        refY[tx + bsize + 1] = selyy[-1 + tx + bsize + 1];
                    if ( 7 == ty )
                    {
                        refCr[tx + bsize + 1] = pycr[-1 + tx + bsize + 1];
                        if ( bsize == 8 )
                            refCb[tx + bsize + 1] = pycb[-1 + tx + bsize + 1];
                    }
                    if ( 8 == ty && bsize != 8 )
                    {
                        refCb[tx + bsize + 1] = pycb[-1 + tx + bsize + 1];
                    }
                } // End of else of if (ipa[mode] < 0)
                
            } // End of else of if ( 4 == bsize )

            // Load iIdx and iFact
            iIdx[ty][tx] = ( (tx + 1) * ipa[mode] ) >> 5;
            iFact[ty][tx] = ( (tx + 1) * ipa[mode] ) & 31;

            if ( iFact[ty][tx] != 0 )
            {
                predSamplesY[ty][tx] = ((32 - iFact[ty][tx]) * refY[ty + iIdx[ty][tx] + 1] + iFact[ty][tx] * refY[ty + iIdx[ty][tx] + 2] + 16) >> 5;
                predSamplesCr[ty][tx] = ((32 - iFact[ty][tx]) * refCr[ty + iIdx[ty][tx] + 1] + iFact[ty][tx] * refCr[ty + iIdx[ty][tx] + 2] + 16) >> 5;
                predSamplesCb[ty][tx] = ((32 - iFact[ty][tx]) * refCb[ty + iIdx[ty][tx] + 1] + iFact[ty][tx] * refCb[ty + iIdx[ty][tx] + 2] + 16) >> 5;
            }
            else
            {
                predSamplesY[ty][tx] = refY[ty + iIdx[ty][tx] + 1];
                predSamplesCr[ty][tx] = refCr[ty + iIdx[ty][tx] + 1];
                predSamplesCb[ty][tx] = refCb[ty + iIdx[ty][tx] + 1];
            }

            if ( mode == ANGULAR_10 && bsize < 32 )
            {
                if ( 0 == tx )
                    predSamplesY[ty][tx] = clip1Y(( (selyy[ty]) + ((selxy[tx]-selyy[MINUS])>>1) ));

                __syncthreads();

            } // End of if ( mode == ANGULAR_10 && bsize < 32 )

        } // End of else if ( mode > ANGULAR_1 && mode < ANGULAR_18 )

      
        ///////////////////
        // STEP 4: HADAMARD
        ///////////////////
        // finally calculation of SATD values for different modes
        // have A matrix which is a shared memory
        // all the threads fill the 'A' array

        if(bsize == 4)
        {
           // everybody computes the difference of pixels
           ay[ty][tx]  = predSamplesY[ty][tx]  - y[row*width + col];
           acr[ty][tx] = predSamplesCr[ty][tx] - cr[row*width + col];
           acb[ty][tx] = predSamplesCb[ty][tx] - cb[row*width + col];

           // construct the B-matrix : 8 threads are working
           if(tx < 2)
           {
               hby[ty][tx] = (ay[ty][2*tx] + ay[ty][2*tx + 1]) + ((ay[ty][2*tx] - ay[ty][2*tx + 1]) << BITS_PER_SUM);
               bcr[ty][tx] = (acr[ty][2*tx] + acr[ty][2*tx + 1]) + ((acr[ty][2*tx] - acr[ty][2*tx+1]) << BITS_PER_SUM);
               bcb[ty][tx] = (acb[ty][2*tx] + acb[ty][2*tx + 1]) + ((acb[ty][2*tx] - acb[ty][2*tx+1]) << BITS_PER_SUM);
           }

           __syncthreads();

           if(tx == 3)
           {  
               // 4 threads work to calculate the value
               if(ty == 0)
               {
                  int a0 = ay[3][0];
                  int a1 = ay[3][1];
                  int a2 = ay[3][2];
                  int a3 = ay[3][3];

                  int sumy  = 0 ;
                  int symcr = 0 ;
                  int sumcb = 0 ;

                  for (int i = 0; i < 2; i++)
                  {
                      HADAMARD4(a0,a1,a2,a3, hby[0][i], hby[1][i], hby[2][i], hby[3][i]);
                      a0 = abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
                      y_satd_shared[mode] += ((sum_t)a0) + (a0 >> BITS_PER_SUM);
                  }
                  y_satd_shared[mode] = (y_satd_shared[mode] << 8) | mode;
              }
              if(ty == 1)
              {

                  int a0 = acr[3][0];
                  int a1 = acr[3][1];
                  int a2 = acr[3][2];
                  int a3 = acr[3][3];

                  for (int i = 0; i < 2; i++)
                  {
                      HADAMARD4(a0,a1,a2,a3, bcr[0][i], bcr[1][i], bcr[2][i], bcr[3][i]);
                      a0 = abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
                      cr_satd_shared[mode] += (a0) + (a0 >> BITS_PER_SUM);
                  }
                  cr_satd_shared[mode] = (cr_satd_shared[mode] << 8) | mode;
              }
              if(ty == 2)
              {

                  int a0 = acb[3][0];
                  int a1 = acb[3][1];
                  int a2 = acb[3][2];
                  int a3 = acb[3][3];

                  for (int i = 0; i < 2; i++)
                  {
                      HADAMARD4(a0,a1,a2,a3, bcb[0][i], bcb[1][i], bcb[2][i], bcb[3][i]);
                      a0 = abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
                      cb_satd_shared[mode] += (a0) + (a0 >> BITS_PER_SUM);
                  }
                  cb_satd_shared[mode] = (cb_satd_shared[mode] << 8) | mode;
              }
          }      
          // TO DO : Also store the sum values appropriately into the resultant array
          // TO DO : Write the same HADAMARD4 macro from the serial code

       }  // if ( 4 == bsize) // end of SATD 4 COMPUTATION

 */       
    } // End of for(int mode =0;mode <35;mode++)

/*
    
    __syncthreads();

    if ( 0 == ty && 0 == tx )
    {
       sort(y_satd_shared);
       extract(y_satd_shared, res_y, y_modes);

       sort(cr_satd_shared);
       extract(cr_satd_shared, res_cr, cr_modes);

       sort(cb_satd_shared);
       extract(cb_satd_shared, res_cb, cb_modes);
    }

    

    
*/
} // End of kernel function hevcPredictionKernel()


