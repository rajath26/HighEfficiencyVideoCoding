
/*****************************************************************************
 * Copyright (C) 2013 x265 project
 *
 * Authors: Gopu Govindaswamy <gopu@govindaswamy.org>
 *          Mandar Gurav <mandar@multicorewareinc.com>
 *          Mahesh Pittala <mahesh@multicorewareinc.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@multicorewareinc.com.
 *****************************************************************************/

#include "ece408_competition.h"
#include "primitives.h"
#include "test/intrapredharness.h"
#include "cpu.h"
#include "TLibCommon/TComRom.h"
#include "TLibEncoder/TEncCfg.h"

#include "input/input.h"
#include "output/output.h"
#include "common.h"
#include "x265.h"
#include "getopt.h"
#include "PPA/ppa.h"

#include "encoder.h"
#include "TLibCommon/TComYuv.h"
#include "TLibCommon/TComPic.h"
#include "TLibCommon/TComPicYuv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <algorithm>

#include "kernel.cu"

//Define this to verify the student intra prediction against the reference version
#define VERIFY

using namespace x265;

void ece408_setup_pic(x265_encoder *enc, TComPic* &pic_ptr, int cu_size);
void ece408_load_pic(x265_encoder *enc, x265_picture *pic_in, TComPic *pic);
ece408_intra_pred_result *ece408_competition_ref(Encoder *encoder, x265_picture *pics_in, TComPic* &pic, int num_frames);
ece408_intra_pred_result *ece408_competition(ece408_frame *imgs, int num_frames);
bool ece408_compare(ece408_intra_pred_result *ref, ece408_intra_pred_result *student, int num_frames);

Pel *refAbove1, *refAbove2, *refLeft1, *refLeft2;
Pel*      predBuf;
int       predBufStride;
int       predBufHeight;
TComYuv pred_yuv;
TComYuv orig_yuv;
TComSPS sps;
TComPPS pps;
x265_param param;

ALIGN_VAR_32(Pel, tmp[33 * 64 * 64]);
ALIGN_VAR_32(Pel, buf_trans[64 * 64]);

static const char short_options[] = "o:f:F:r:i:b:s:q:m:hwV";
static const struct option long_options[] =
{
#if HIGH_BIT_DEPTH
    { "depth",          required_argument, NULL, 0 },
#endif
    { "help",                 no_argument, NULL, 'h' },
    { "cpuid",          required_argument, NULL, 0 },
    { "threads",        required_argument, NULL, 0 },
    { "preset",         required_argument, NULL, 'p' },
    { "tune",           required_argument, NULL, 't' },
    { "frame-threads",  required_argument, NULL, 'F' },
    { "log",            required_argument, NULL, 0 },
    { "csv",            required_argument, NULL, 0 },
    { "y4m",                  no_argument, NULL, 0 },
    { "no-progress",          no_argument, NULL, 0 },
    { "output",         required_argument, NULL, 'o' },
    { "input",          required_argument, NULL, 0 },
    { "input-depth",    required_argument, NULL, 0 },
    { "input-res",      required_argument, NULL, 0 },
    { "input-csp",      required_argument, NULL, 0 },
    { "fps",            required_argument, NULL, 0 },
    { "frame-skip",     required_argument, NULL, 0 },
    { "frames",         required_argument, NULL, 'f' },
    { "recon",          required_argument, NULL, 'r' },
    { "recon-depth",    required_argument, NULL, 0 },
    { "no-wpp",               no_argument, NULL, 0 },
    { "wpp",                  no_argument, NULL, 0 },
    { "ctu",            required_argument, NULL, 's' },
    { "tu-intra-depth", required_argument, NULL, 0 },
    { "tu-inter-depth", required_argument, NULL, 0 },
    { "me",             required_argument, NULL, 0 },
    { "subme",          required_argument, NULL, 'm' },
    { "merange",        required_argument, NULL, 0 },
    { "max-merge",      required_argument, NULL, 0 },
    { "rdpenalty",      required_argument, NULL, 0 },
    { "no-rect",              no_argument, NULL, 0 },
    { "rect",                 no_argument, NULL, 0 },
    { "no-amp",               no_argument, NULL, 0 },
    { "amp",                  no_argument, NULL, 0 },
    { "no-early-skip",        no_argument, NULL, 0 },
    { "early-skip",           no_argument, NULL, 0 },
    { "no-fast-cbf",          no_argument, NULL, 0 },
    { "fast-cbf",             no_argument, NULL, 0 },
    { "no-tskip",             no_argument, NULL, 0 },
    { "tskip",                no_argument, NULL, 0 },
    { "no-tskip-fast",        no_argument, NULL, 0 },
    { "tskip-fast",           no_argument, NULL, 0 },
    { "no-constrained-intra", no_argument, NULL, 0 },
    { "constrained-intra",    no_argument, NULL, 0 },
    { "refresh",        required_argument, NULL, 0 },
    { "keyint",         required_argument, NULL, 'i' },
    { "rc-lookahead",   required_argument, NULL, 0 },
    { "bframes",        required_argument, NULL, 'b' },
    { "bframe-bias",    required_argument, NULL, 0 },
    { "b-adapt",        required_argument, NULL, 0 },
    { "ref",            required_argument, NULL, 0 },
    { "no-weightp",           no_argument, NULL, 0 },
    { "weightp",              no_argument, NULL, 'w' },
    { "crf",            required_argument, NULL, 0 },
    { "bitrate",        required_argument, NULL, 0 },
    { "qp",             required_argument, NULL, 'q' },
    { "aq-mode",        required_argument, NULL, 0 },
    { "cbqpoffs",       required_argument, NULL, 0 },
    { "crqpoffs",       required_argument, NULL, 0 },
    { "rd",             required_argument, NULL, 0 },
    { "no-signhide",          no_argument, NULL, 0 },
    { "signhide",             no_argument, NULL, 0 },
    { "no-lft",               no_argument, NULL, 0 },
    { "lft",                  no_argument, NULL, 0 },
    { "no-sao",               no_argument, NULL, 0 },
    { "sao",                  no_argument, NULL, 0 },
    { "sao-lcu-bounds", required_argument, NULL, 0 },
    { "sao-lcu-opt",    required_argument, NULL, 0 },
    { "no-ssim",              no_argument, NULL, 0 },
    { "ssim",                 no_argument, NULL, 0 },
    { "no-psnr",              no_argument, NULL, 0 },
    { "psnr",                 no_argument, NULL, 0 },
    { "hash",           required_argument, NULL, 0 },
    { "no-strong-intra-smoothing", no_argument, NULL, 0 },
    { "strong-intra-smoothing",    no_argument, NULL, 0 },
    { 0, 0, 0, 0 }
};

struct CLIOptions
{
    Input*  input;
    Output* recon;
    std::fstream bitstreamFile;
    bool bProgress;
    bool bForceY4m;
    uint32_t totalbytes;

    uint32_t frameSkip;         // number of frames to skip from the beginning
    uint32_t framesToBeEncoded; // number of frames to encode

    int64_t startTime;
    int64_t prevUpdateTime;

    /* in microseconds */
    static const int UPDATE_INTERVAL = 250000;

    CLIOptions()
    {
        input = NULL;
        recon = NULL;
        framesToBeEncoded = frameSkip = totalbytes = 0;
        bProgress = true;
        bForceY4m = false;
        startTime = x265_mdate();
        prevUpdateTime = 0;
    }

    void destroy();
    void writeNALs(const x265_nal* nal, uint32_t nalcount);
    void printVersion(x265_param *par);
    void showHelp(x265_param *par);
    bool parse(int argc, char **argv, x265_param* par);
};

void CLIOptions::destroy()
{
    if (input)
        input->release();
    input = NULL;
    if (recon)
        recon->release();
    recon = NULL;
}

void CLIOptions::writeNALs(const x265_nal* nal, uint32_t nalcount)
{
    PPAScopeEvent(bitstream_write);
    for (uint32_t i = 0; i < nalcount; i++)
    {
        bitstreamFile.write((const char*)nal->p_payload, nal->i_payload);
        totalbytes += nal->i_payload;
        nal++;
    }
}

void CLIOptions::printVersion(x265_param *par)
{
    fprintf(stderr, "x265 [info]: HEVC encoder version %s\n", x265_version_str);
    fprintf(stderr, "x265 [info]: build info %s\n", x265_build_info_str);
    x265_setup_primitives(par, 0);
}

void CLIOptions::showHelp(x265_param *par)
{
    x265_param_default(par);
    printVersion(par);
#define H0 printf
#define OPT(value) (value ? "enabled" : "disabled")
    H0("\nSyntax: x265 [options] infile [-o] outfile\n");
    H0("    infile can be YUV or Y4M\n");
    H0("    outfile is raw HEVC bitstream\n");
    H0("\nExecutable Options:\n");
    H0("-h/--h                           Show this help text and exit\n");
    H0("-V/--version                     Show version info and exit\n");
    H0("   --cpuid                       Limit SIMD capability bitmap 0:auto 1:None. Default:0\n");
    H0("   --threads                     Number of threads for thread pool (0: detect CPU core count, default)\n");
    H0("-F/--frame-threads               Number of concurrently encoded frames. Default %d\n", par->frameNumThreads);
    H0("   --log                         Logging level 0:ERROR 1:WARNING 2:INFO 3:DEBUG -1:NONE. Default %d\n", par->logLevel);
    H0("   --csv                         Comma separated log file, log level >= 3 frame log, else one line per run\n");
    H0("   --y4m                         Parse input stream as YUV4MPEG2 regardless of file extension\n");
    H0("   --no-progress                 Disable CLI progress reports\n");
    H0("-o/--output                      Bitstream output file name\n");
    H0("\nInput Options:\n");
    H0("   --input                       Raw YUV or Y4M input file name\n");
    H0("   --input-depth                 Bit-depth of input file (YUV only) Default %d\n", par->inputBitDepth);
    H0("   --input-res                   Source picture size [w x h], auto-detected if Y4M\n");
    H0("   --input-csp                   Source color space parameter, auto-detected if Y4M\n");
    H0("   --fps                         Source frame rate, auto-detected if Y4M\n");
    H0("   --frame-skip                  Number of frames to skip at start of input file\n");
    H0("-f/--frames                      Number of frames to be encoded. Default all\n");
    H0("\nQuad-Tree analysis:\n");
    H0("   --[no-]wpp                    Enable Wavefront Parallel Processing. Default %s\n", OPT(par->bEnableWavefront));
    H0("-s/--ctu                         Maximum CU size (default: 64x64). Default %d\n", par->maxCUSize);
    H0("   --tu-intra-depth              Max TU recursive depth for intra CUs. Default %d\n", par->tuQTMaxIntraDepth);
    H0("   --tu-inter-depth              Max TU recursive depth for inter CUs. Default %d\n", par->tuQTMaxInterDepth);
    H0("\nTemporal / motion search options:\n");
    H0("   --me                          Motion search method 0:dia 1:hex 2:umh 3:star 4:full. Default %d\n", par->searchMethod);
    H0("-m/--subme                       Amount of subpel refinement to perform (0:least .. 7:most). Default %d \n", par->subpelRefine);
    H0("   --merange                     Motion search range. Default %d\n", par->searchRange);
    H0("   --[no-]rect                   Enable rectangular motion partitions Nx2N and 2NxN. Default %s\n", OPT(par->bEnableRectInter));
    H0("   --[no-]amp                    Enable asymmetric motion partitions, requires --rect. Default %s\n", OPT(par->bEnableAMP));
    H0("   --max-merge                   Maximum number of merge candidates. Default %d\n", par->maxNumMergeCand);
    H0("   --[no-]early-skip             Enable early SKIP detection. Default %s\n", OPT(par->bEnableEarlySkip));
    H0("   --[no-]fast-cbf               Enable Cbf fast mode \n \t\t\t\t Default : %s\n", OPT(par->bEnableCbfFastMode));
    H0("\nSpatial / intra options:\n");
    H0("   --rdpenalty                   penalty for 32x32 intra TU in non-I slices. 0:disabled 1:RD-penalty 2:maximum. Default %d\n", par->rdPenalty);
    H0("   --[no-]tskip                  Enable intra transform skipping. Default %s\n", OPT(par->bEnableTransformSkip));
    H0("   --[no-]tskip-fast             Enable fast intra transform skipping. Default %s\n", OPT(par->bEnableTSkipFast));
    H0("   --[no-]strong-intra-smoothing Enable strong intra smoothing for 32x32 blocks. Default %s\n", OPT(par->bEnableStrongIntraSmoothing));
    H0("   --[no-]constrained-intra      Constrained intra prediction (use only intra coded reference pixels) Default %s\n", OPT(par->bEnableConstrainedIntra));
    H0("\nSlice decision options:\n");
    H0("   --refresh                     Intra refresh type - 0:none, 1:CDR, 2:IDR (default: CDR) Default %d\n", par->decodingRefreshType);
    H0("-i/--keyint                      Max intra period in frames. Default %d\n", par->keyframeMax);
    H0("   --rc-lookahead                Number of frames for frame-type lookahead (determines encoder latency) Default %d\n", par->lookaheadDepth);
    H0("   --bframes                     Maximum number of consecutive b-frames (now it only enables B GOP structure) Default %d\n", par->bframes);
    H0("   --bframe-bias                 Bias towards B frame decisions. Default %d\n", par->bFrameBias);
    H0("   --b-adapt                     0 - none, 1 - fast, 2 - full (trellis) adaptive B frame scheduling. Default %d\n", par->bFrameAdaptive);
    H0("   --ref                         max number of L0 references to be allowed (1 .. 16) Default %d\n", par->maxNumReferences);
    H0("-w/--[no-]weightp                Enable weighted prediction in P slices. Default %s\n", OPT(par->bEnableWeightedPred));
    H0("\nQP, rate control and rate distortion options:\n");
    H0("   --bitrate                     Target bitrate (kbps), implies ABR. Default %d\n", par->rc.bitrate);
    H0("-q/--qp                          Base QP for CQP mode. Default %d\n", par->rc.qp);
    H0("   --cbqpoffs                    Chroma Cb QP Offset. Default %d\n", par->cbQpOffset);
    H0("   --crqpoffs                    Chroma Cr QP Offset. Default %d\n", par->crQpOffset);
    H0("   --rd                          Level of RD in mode decision 0:least....2:full RDO. Default %d\n", par->rdLevel);
    H0("   --[no-]signhide               Hide sign bit of one coeff per TU (rdo). Default %s\n", OPT(par->bEnableSignHiding));
    H0("\nLoop filter:\n");
    H0("   --[no-]lft                    Enable Loop Filter. Default %s\n", OPT(par->bEnableLoopFilter));
    H0("\nSample Adaptive Offset loop filter:\n");
    H0("   --[no-]sao                    Enable Sample Adaptive Offset. Default %s\n", OPT(par->bEnableSAO));
    H0("   --sao-lcu-bounds              0: right/bottom boundary areas skipped  1: non-deblocked pixels are used. Default %d\n", par->saoLcuBoundary);
    H0("   --sao-lcu-opt                 0: SAO picture-based optimization, 1: SAO LCU-based optimization. Default %d\n", par->saoLcuBasedOptimization);
    H0("\nQuality reporting metrics:\n");
    H0("   --[no-]ssim                   Enable reporting SSIM metric scores. Default %s\n", OPT(par->bEnableSsim));
    H0("   --[no-]psnr                   Enable reporting PSNR metric scores. Default %s\n", OPT(par->bEnablePsnr));
    H0("\nReconstructed video options (debugging):\n");
    H0("-r/--recon                       Reconstructed raw image YUV or Y4M output file name\n");
    H0("   --recon-depth                 Bit-depth of reconstructed raw image file. Default 8\n");
    H0("\nSEI options:\n");
    H0("   --hash                        Decoded Picture Hash SEI 0: disabled, 1: MD5, 2: CRC, 3: Checksum. Default %d\n", par->decodedPictureHashSEI);
#undef OPT
#undef H0
    exit(0);
}

bool CLIOptions::parse(int argc, char **argv, x265_param* par)
{
    int berror = 0;
    int help = 0;
    int cpuid = 0;
    int reconFileBitDepth = 8;
    const char *inputfn = NULL;
    const char *reconfn = NULL;
    const char *bitstreamfn = NULL;
    const char *inputRes = NULL;
    const char *preset = "medium";
    const char *tune = "psnr";

    /* Presets are applied before all other options. */
    for (optind = 0;; )
    {
        int c = getopt_long(argc, argv, short_options, long_options, NULL);
        if (c == -1)
            break;
        if (c == 'p')
            preset = optarg;
        if (c == 't')
            tune = optarg;
        else if (c == '?')
            return true;
    }

    if (x265_param_default_preset(par, preset, tune) < 0)
        return true;

    for (optind = 0;; )
    {
        int long_options_index = -1;
        int c = getopt_long(argc, argv, short_options, long_options, &long_options_index);
        if (c == -1)
        {
            break;
        }

        switch (c)
        {
        case 'h':
            showHelp(par);
            break;

        case 'V':
            printVersion(par);
            exit(0);

        default:
            if (long_options_index < 0 && c > 0)
            {
                for (size_t i = 0; i < sizeof(long_options) / sizeof(long_options[0]); i++)
                {
                    if (long_options[i].val == c)
                    {
                        long_options_index = (int)i;
                        break;
                    }
                }

                if (long_options_index < 0)
                {
                    /* getopt_long might have already printed an error message */
                    if (c != 63)
                        x265_log(NULL, X265_LOG_WARNING, "internal error: short option '%c' has no long option\n", c);
                    return true;
                }
            }
            if (long_options_index < 0)
            {
                x265_log(NULL, X265_LOG_WARNING, "short option '%c' unrecognized\n", c);
                return true;
            }
#define OPT(longname) \
    else if (!strcmp(long_options[long_options_index].name, longname))

            if (0) ;
            OPT("cpuid") cpuid = atoi(optarg);
            OPT("frames") this->framesToBeEncoded = (uint32_t)atoi(optarg);
            OPT("preset") preset = optarg;
            OPT("tune") tune = optarg;
            OPT("no-progress") this->bProgress = false;
            OPT("frame-skip") this->frameSkip = (uint32_t)atoi(optarg);
            OPT("output") bitstreamfn = optarg;
            OPT("input") inputfn = optarg;
            OPT("recon") reconfn = optarg;
            OPT("input-depth") par->inputBitDepth = (uint32_t)atoi(optarg);
            OPT("recon-depth") reconFileBitDepth = (uint32_t)atoi(optarg);
            OPT("input-res") inputRes = optarg;
            OPT("y4m") bForceY4m = true;
            else
                berror |= x265_param_parse(par, long_options[long_options_index].name, optarg);

            if (berror)
            {
                const char *name = long_options_index > 0 ? long_options[long_options_index].name : argv[optind - 2];
                x265_log(NULL, X265_LOG_ERROR, "invalid argument: %s = %s\n", name, optarg);
                return true;
            }
#undef OPT
        }
    }

    if (optind < argc && !inputfn)
        inputfn = argv[optind++];
    if (optind < argc && !bitstreamfn)
        bitstreamfn = argv[optind++];
    if (optind < argc)
    {
        x265_log(par, X265_LOG_WARNING, "extra unused command arguments given <%s>\n", argv[optind]);
        return true;
    }

    if (argc <= 1 || help)
        showHelp(par);

    if (inputfn == NULL || bitstreamfn == NULL)
    {
        x265_log(par, X265_LOG_ERROR, "input or output file not specified, try -V for help\n");
        return true;
    }
    this->input = Input::open(inputfn, par->inputBitDepth, bForceY4m);
    if (!this->input || this->input->isFail())
    {
        x265_log(par, X265_LOG_ERROR, "unable to open input file <%s>\n", inputfn);
        return true;
    }
    if (this->input->getWidth())
    {
        /* parse the width, height, frame rate from the y4m file */
        par->sourceCsp = this->input->getColorSpace();
        par->sourceWidth = this->input->getWidth();
        par->sourceHeight = this->input->getHeight();
        par->frameRate = (int)this->input->getRate();
    }
    else if (inputRes)
    {
        this->input->setColorSpace(par->sourceCsp);
        sscanf(inputRes, "%dx%d", &par->sourceWidth, &par->sourceHeight);
        this->input->setDimensions(par->sourceWidth, par->sourceHeight);
        this->input->setBitDepth(par->inputBitDepth);
    }
    else if (par->sourceHeight <= 0 || par->sourceWidth <= 0 || par->frameRate <= 0)
    {
        x265_log(par, X265_LOG_ERROR, "YUV input requires source width, height, and rate to be specified\n");
        return true;
    }
    else
    {
        this->input->setDimensions(par->sourceWidth, par->sourceHeight);
        this->input->setBitDepth(par->inputBitDepth);
    }

    if (reconFileBitDepth > 0)
    {
        if (reconFileBitDepth != par->inputBitDepth)
        {
            x265_log(par, X265_LOG_ERROR, "Bit depth of the recon file should be the same as input bit depth\n");
            /* TODO: Support recon files with bitdepth > input bit depth??*/
            return true;
        }
    }
    else
    {
        reconFileBitDepth = par->inputBitDepth;
    }

    int guess = this->input->guessFrameCount();
    if (this->frameSkip)
    {
        this->input->skipFrames(this->frameSkip);
    }

    uint32_t fileFrameCount = guess < 0 ? 0 : (uint32_t)guess;
    if (this->framesToBeEncoded && fileFrameCount)
        this->framesToBeEncoded = X265_MIN(this->framesToBeEncoded, fileFrameCount - this->frameSkip);
    else if (fileFrameCount)
        this->framesToBeEncoded = fileFrameCount - this->frameSkip;

    if (par->logLevel >= X265_LOG_INFO)
    {
        if (this->framesToBeEncoded == 0)
            fprintf(stderr, "%s  [info]: %dx%d %dHz %s, unknown frame count\n", input->getName(),
                    par->sourceWidth, par->sourceHeight, par->frameRate,
                    (par->sourceCsp >= X265_CSP_I444) ? "C444" : (par->sourceCsp >= X265_CSP_I422) ? "C422" : "C420");
        else
            fprintf(stderr, "%s  [info]: %dx%d %dHz %s, frames %u - %d of %d\n", input->getName(),
                    par->sourceWidth, par->sourceHeight, par->frameRate,
                    (par->sourceCsp >= X265_CSP_I444) ? "C444" : (par->sourceCsp >= X265_CSP_I422) ? "C422" : "C420",
                    this->frameSkip, this->frameSkip + this->framesToBeEncoded - 1, fileFrameCount);
    }

    this->input->startReader();

    if (reconfn)
    {
        this->recon = Output::open(reconfn, par->sourceWidth, par->sourceHeight, reconFileBitDepth, par->frameRate, par->sourceCsp);
        if (this->recon->isFail())
        {
            x265_log(par, X265_LOG_WARNING, "unable to write reconstruction file\n");
            this->recon->release();
            this->recon = 0;
        }
    }

#if HIGH_BIT_DEPTH
    if (par->inputBitDepth != 12 && par->inputBitDepth != 10 && par->inputBitDepth != 8)
    {
        x265_log(par, X265_LOG_ERROR, "Only bit depths of 8, 10, or 12 are supported\n");
        return true;
    }
#else
    if (par->inputBitDepth != 8)
    {
        x265_log(par, X265_LOG_ERROR, "not compiled for bit depths greater than 8\n");
        return true;
    }
#endif // if HIGH_BIT_DEPTH

    this->bitstreamFile.open(bitstreamfn, std::fstream::binary | std::fstream::out);
    if (!this->bitstreamFile)
    {
        x265_log(NULL, X265_LOG_ERROR, "failed to open bitstream file <%s> for writing\n", bitstreamfn);
        return true;
    }

    x265_setup_primitives(par, cpuid);
    printVersion(par);
    return false;
}

int main(int argc, char *argv[])
{
    CLIOptions   cliopt;

    if (cliopt.parse(argc, argv, &param))
    {
        cliopt.destroy();
        exit(1);
    }

    param.bEnableStrongIntraSmoothing = false; //No strong intra smoothing for competition

    x265_encoder *encoder = x265_encoder_open(&param);
    if (!encoder)
    {
        x265_log(&param, X265_LOG_ERROR, "failed to open encoder\n");
        cliopt.destroy();
        x265_cleanup();
        exit(1);
    }

	Encoder *encoder_c = static_cast<Encoder*>(encoder);

    //Initialize arrays for storing neighboring pixel values
    refAbove1 = (Pel*)X265_MALLOC(Pel, 3 * MAX_CU_SIZE);
    refAbove2 = (Pel*)X265_MALLOC(Pel, 3 * MAX_CU_SIZE);
    refLeft1 = (Pel*)X265_MALLOC(Pel, 3 * MAX_CU_SIZE);
    refLeft2 = (Pel*)X265_MALLOC(Pel, 3 * MAX_CU_SIZE);
    predBufHeight  = ((MAX_CU_SIZE + 2) << 4);
    predBufStride = ((MAX_CU_SIZE  + 8) << 4);
    predBuf = new Pel[predBufStride * predBufHeight];

    //Save globals so we can restore them at the end
    //We need to restore the original values before destroy()ing data structures because many of the destroy() functions
    //use these globals to determine the size of their arrays
    int g_maxCUDepth_bak = g_maxCUDepth;
    int g_addCUDepth_bak = g_addCUDepth;
    int g_maxCUWidth_bak = g_maxCUWidth;
    int g_maxCUHeight_bak = g_maxCUHeight;

    g_maxCUDepth = 0; //Disallow recursion to decompose frames into a regular grid of equal size CUs.
    g_addCUDepth = 0;
    //NOTE: has to be after x265_encoder_open() call, since that calls x265_set_globals(), which resets g_maxCUDepth.
    x265_picture pic_orig;
    x265_picture *pic_in = &pic_orig;

    x265_picture_init(&param, pic_in);

    uint32_t inFrameCount = 0;

    TComPic *pic = NULL;

    while (1)
    {
        pic_orig.poc = inFrameCount;
        if (cliopt.framesToBeEncoded && inFrameCount >= cliopt.framesToBeEncoded)
            break;
        else if (cliopt.input->readPicture(pic_orig))
            inFrameCount++;
        else
            break;

        ece408_intra_pred_result *ref = ece408_competition_ref(encoder_c, pic_in, pic, 1);
        
        ece408_frame frame(param.sourceWidth, param.sourceHeight, pic_in);

        //Uncomment this one to run the student version
        ece408_intra_pred_result *student = ece408_competition(&frame, 1);
        //Uncomment this one instead to run the reference version twice (to test the compare function)
        //ece408_intra_pred_result *student = ece408_competition_ref(encoder_c, pic_in, pic, 1);

#ifdef VERIFY
        if(!ece408_compare(ref, student, 1)) {
        	printf("Error in frame %d\n", inFrameCount);
        	exit(1);
        }
#endif
        for(int i = 0; i < 4*1; i++) {
        	ref[i].destroy();
            student[i].destroy();
        }
        delete[] ref;
        delete[] student;
    }
    
#ifdef VERIFY
    printf("Success!\n");
#endif

    //Restore globals
    g_maxCUDepth = g_maxCUDepth_bak;
    g_addCUDepth = g_addCUDepth_bak;
    g_maxCUWidth = g_maxCUWidth_bak;
    g_maxCUHeight = g_maxCUHeight_bak;

    pic->destroy(param.bframes);
    delete pic;

    x265_encoder_close(encoder);

    X265_FREE(refAbove1);
    X265_FREE(refAbove2);
    X265_FREE(refLeft1);
    X265_FREE(refLeft2);
    delete[] predBuf;

    orig_yuv.destroy();
    pred_yuv.destroy();

    x265_cleanup(); /* Free library singletons */
    cliopt.destroy();

    return 0;
}

void ece408_setup_pic(x265_encoder *enc, TComPic* &pic_ptr, int cu_size) {
    Encoder *encoder = static_cast<Encoder*>(enc);

    g_maxCUHeight = cu_size;
    g_maxCUWidth = cu_size;
    #ifdef VERBOSE
    printf("%u x %u CU size\n", g_maxCUHeight, g_maxCUWidth);
    #endif
    if(!pic_ptr) {
        pic_ptr = new TComPic;
        pic_ptr->create(encoder);
    }
    else {
        pic_ptr->destroy(param.bframes);
	    pic_ptr->create(encoder);
    }

    // initialize SPS
    encoder->initSPS(&sps);
    pps.setSPS(&sps);
    encoder->initPPS(&pps);
    TComSlice* slice = pic_ptr->getSlice();
    slice->setSPS(&sps);
    slice->setPPS(&pps);
    slice->setSliceBits(0);
    slice->setPic(pic_ptr);
    slice->initSlice();

	#ifdef VERBOSE     
    printf("%u x %u CUs, %u CUs in frame\n", pic_ptr->getFrameWidthInCU(), pic_ptr->getFrameHeightInCU(), pic_ptr->getNumCUsInFrame());
    #endif
}

void ece408_load_pic(x265_encoder *enc, x265_picture *pic_in, TComPic *pic) {
    Encoder *encoder = static_cast<Encoder*>(enc);
    pic->getPicYuvOrg()->copyFromPicture(*pic_in, encoder->m_pad);
    //Need to copy the original picture to the reconstructed member too because TComPattern gets its neighbor samples from there.
    //TODO can we just make the Rec member point at the Org member and avoid 2 copies?
    pic->getPicYuvRec()->copyFromPicture(*pic_in, encoder->m_pad);
}

//channel = 0 for luma, 1 for cb, 2 for cr
void ece408_intra_pred_channel(TComPic *pic, TComDataCU *cu, int luma_size, int channel, int32_t *sad_ptr) {
//#define VERBOSE
#ifdef VERBOSE
	int cuAddr = cu->getAddr();
   	printf("refAbove1: ");
   	for(int i = 0; i < 64*3; i++)
   		printf("%d ", refAbove1[i]);
   	printf("\n");
   	printf("refAbove2: ");
   	for(int i = 0; i < 64*3; i++)
   		printf("%d ", refAbove2[i]);
   	printf("\n");
   	printf("refLeft1: ");
   	for(int i = 0; i < 64*3; i++)
   		printf("%d ", refLeft1[i]);
   	printf("\n");
   	printf("refLeft2: ");
   	for(int i = 0; i < 64*3; i++)
   		printf("%d ", refLeft2[i]);
   	printf("\n");
#endif
	int chroma_size = luma_size >> 1;
	bool luma = (channel == 0);
	bool cb = (channel == 1);
	bool cr = (channel == 2);
	int size = luma ? luma_size : chroma_size;
	Pel* orig_pel   = luma ? orig_yuv.getLumaAddr(0, size) : (cb ? orig_yuv.getCbAddr(0, size) : orig_yuv.getCrAddr(0, size));
    Pel* pred_pel   = luma ? pred_yuv.getLumaAddr(0, size) : (cb ? pred_yuv.getCbAddr(0, size) : pred_yuv.getCrAddr(0, size));
    uint32_t stride = luma ? pred_yuv.getStride() : pred_yuv.getCStride();

    Pel *pAboveUnfilt = (cr ? refAbove2 : refAbove1) + size - 1;
    Pel *pAboveFilt = luma ? (refAbove2 + size - 1) : pAboveUnfilt;
    Pel *pLeftUnfilt = (cr ? refLeft2 : refLeft1) + size - 1;
    Pel *pLeftFilt = luma ? (refLeft2  + size - 1) : pLeftUnfilt;

    int nLog2SizeMinus2 = g_convertToBit[size];
    pixelcmp_t sa8d = primitives.sa8d[nLog2SizeMinus2];

    #ifdef VERBOSE
    printf("CU %u channel %d Orig:\n", cuAddr, channel);
    for(int row = 0; row < size; row++) {
        for(int col = 0; col < size; col++) {
            printf("%02X ", orig_pel[row*size + col]);
        }
        printf("\n");
    }
    #endif

    int sad;

    Pel *above = (luma && size >= 8) ? pAboveFilt : pAboveUnfilt;
    Pel *left  = (luma && size >= 8) ? pLeftFilt : pLeftUnfilt;

    //TODO check to make sure we're filtering in all the right conditions
    primitives.intra_pred_planar(above + 1, left + 1, pred_pel, stride, size);
    sad = sa8d(orig_pel, stride, pred_pel, stride);
    *(sad_ptr++) = sad;
    #ifdef VERBOSE
    printf("CU %u channel %d, planar SAD = %d\n", cuAddr, channel, sad);
    #endif

    //TODO check to make sure we're filtering in all the right conditions
    primitives.intra_pred_dc(pAboveUnfilt + 1, pLeftUnfilt + 1, pred_pel, stride, size, (size <= 16));
    sad = sa8d(orig_pel, stride, pred_pel, stride);
    *(sad_ptr++) = sad;
    #ifdef VERBOSE
    printf("CU %u channel %d, size = %d, stride = %d, DC:\n", cuAddr, channel, size, stride);
    for(int row = 0; row < size; row++) {
    	for(int col = 0; col < size; col++) {
    		printf("%02X ", pred_pel[row*size+col]);
    	}
    	printf("\n");
    }
    printf("SAD = %d\n", sad);
    #endif

    primitives.transpose[nLog2SizeMinus2](buf_trans, orig_pel, stride);
    //TODO check to make sure we're filtering in all the right conditions
    primitives.intra_pred_allangs[nLog2SizeMinus2](tmp, pAboveUnfilt, pLeftUnfilt, pAboveFilt, pLeftFilt, (luma && (size <= 16)));
    #ifdef VERBOSE
    printf("CU %u channel %d, angular SAD = ", cuAddr, channel);
    #endif
    for (int mode = 2; mode < 35; mode++)
    {
        bool modeHor = (mode < 18);
        Pel *cmp = (modeHor ? buf_trans : orig_pel);
        intptr_t srcStride = (modeHor ? size : stride);
    #ifdef VERBOSE
    	printf("Pred mode %d\n", mode);
    	for(int r = 0; r < size; r++) {
    		for(int c = 0; c < size; c++)
    			printf("%02X ", tmp[(mode-2) * (size * size) + r * size + c]);
    		printf("\n");
    	}
   	#endif
        sad = sa8d(cmp, srcStride, &tmp[(mode - 2) * (size * size)], size);
        *(sad_ptr++) = sad;
    #ifdef VERBOSE
        printf("%d, ", sad);
    #endif
    }
    #ifdef VERBOSE
    printf("\n");
    #endif
}
//#undef VERBOSE

// TComPattern doesn't have a version for chroma that creates buffered references
// NOTE: No filtering of neighbor samples is done for chroma
void initAdiPatternChroma(TComPattern *pat, TComDataCU* cu, uint32_t zOrderIdxInPart, uint32_t partDepth, Pel* adiBuf, int strideOrig, int heightOrig,
                                 Pel* refAboveCb, Pel* refLeftCb, Pel* refAboveCr, Pel* refLeftCr)
{
    pat->initAdiPatternChroma(cu, zOrderIdxInPart, partDepth, adiBuf, strideOrig, heightOrig);
    uint32_t cuWidth   = cu->getWidth(0) >> partDepth >> 1; //>> 1 to account for 4:2:0 subsampling
    uint32_t cuHeight  = cu->getHeight(0) >> partDepth >> 1;
    uint32_t cuWidth2  = cuWidth << 1;
    uint32_t cuHeight2 = cuHeight << 1;

    refAboveCb += cuWidth - 1;
    refLeftCb += cuWidth - 1;

    refAboveCr += cuWidth - 1;
    refLeftCr += cuWidth - 1;

    memcpy(refAboveCb, adiBuf, (cuWidth2 + 1) * sizeof(Pel));
    memcpy(refAboveCr, adiBuf + ADI_BUF_STRIDE * (2 * cuHeight + 1), (cuWidth2 + 1) * sizeof(Pel));

    for (unsigned int k = 0; k < cuHeight2 + 1; k++)
    {
        refLeftCb[k] = adiBuf[k * ADI_BUF_STRIDE];
        refLeftCr[k] = (adiBuf + ADI_BUF_STRIDE * (cuHeight2 + 1))[k * ADI_BUF_STRIDE];
    }
}

//luma_size is the (square) block size of luma blocks, chroma blocks are assumed (luma_size/2)x(luma_size/2)
void ece408_intra_pred(TComPic *pic, TComDataCU *cu, int luma_size, int32_t *y_ptr, int32_t *cb_ptr, int32_t *cr_ptr) {
	int cuAddr = cu->getAddr();
    orig_yuv.copyFromPicLuma(pic->getPicYuvOrg(), cuAddr, 0);
    if(luma_size > 4)
        orig_yuv.copyFromPicChroma(pic->getPicYuvOrg(), cuAddr, 0);

	cu->getPattern()->initPattern(cu, /*initTrDepth*/ 0, /*partOffset*/ 0);
    cu->getPattern()->initAdiPattern(cu, /*partOffset*/ 0, /*initTrDepth*/ 0, predBuf, predBufStride, predBufHeight, refAbove1, refLeft1, refAbove2, refLeft2);

	ece408_intra_pred_channel(pic, cu, luma_size, 0, y_ptr);

	if(luma_size > 4) { //No 2x2 chroma blocks, and 4x4 chroma blocks are covered with 8x8 luma
		initAdiPatternChroma(cu->getPattern(), cu, 0, 0, predBuf, predBufStride, predBufHeight, refAbove1, refLeft1, refAbove2, refLeft2);
		ece408_intra_pred_channel(pic, cu, luma_size, 1, cb_ptr);
		ece408_intra_pred_channel(pic, cu, luma_size, 2, cr_ptr);
	}
}

ece408_intra_pred_result *ece408_competition_ref(Encoder *encoder, x265_picture *pics_in, TComPic* &pic, int num_frames) { 
	ece408_intra_pred_result *ret = new ece408_intra_pred_result[4*num_frames]; //8x8,16x16,32x32,64x64
	ece408_intra_pred_result *cur_result = ret;

	for(int i = 0; i < num_frames; i++) {
		for(int luma_size_shift = 2; luma_size_shift <= 5; luma_size_shift++) {
	        int luma_size = 1 << luma_size_shift; // luma_size x luma_size luma PBs
	        cur_result->create(param.sourceWidth, param.sourceHeight, luma_size);

	        int32_t *y_satd_results = cur_result->y_satd_results;
		    uint8_t *y_modes = cur_result->y_modes;
		    int32_t *cb_satd_results = cur_result->cb_satd_results;
		    uint8_t *cb_modes = cur_result->cb_modes;
		    int32_t *cr_satd_results = cur_result->cr_satd_results;
		    uint8_t *cr_modes = cur_result->cr_modes;

	        ece408_setup_pic(encoder, pic, luma_size);
	        ece408_load_pic(encoder, &pics_in[i], pic);

	        //TODO would like not to have to destroy and rebuild the entire TComPic for different CU sizes, since the underlying data arrays are the same.
	        orig_yuv.destroy();
	        orig_yuv.create(luma_size, luma_size, CHROMA_420);
	        pred_yuv.destroy();
	        pred_yuv.create(luma_size, luma_size, CHROMA_420);

            //cuIndex != cuAddr because pic includes partial CUs that fall off the right and bottom edges of the frame.
            //In our case, since we're not subdividing the CUs at all, partial CUs are meaningless (they are never actually coded in an HEVC bitstream),
            //so we only care about full CUs.
            for(unsigned int cuIndex = 0; cuIndex < (unsigned int)((param.sourceWidth/luma_size)*(param.sourceHeight/luma_size)); cuIndex++) {
                unsigned int cuAddr = ((cuIndex / (param.sourceWidth/luma_size))*(pic->getFrameWidthInCU())) + (cuIndex % (param.sourceWidth/luma_size)); //TODO structure for better CSE
	            TComDataCU* cu = pic->getCU(cuAddr);
	            cu->initCU(pic, cuAddr);

	            ece408_intra_pred(pic,
	            	              cu,
	            	              luma_size,
	            	              &(y_satd_results[35*cuIndex]),
	            	              &(cb_satd_results[35*cuIndex]),
	            	              &(cr_satd_results[35*cuIndex]));
	            //printf("SATD results: ");
	            //for(int l = 0; l < 35; l++) {
	            //	printf("(%d, %d, %d, %d) ", l, y_satd_results[35*cuIndex+l], cb_satd_results[35*cuIndex+l], cr_satd_results[35*cuIndex+l]);
	            //}
	            //printf("\n");
	            for(int mode = 0; mode < 35; mode++) {
	            	y_satd_results[35*cuIndex + mode] = (y_satd_results[35*cuIndex + mode] << 8) | mode;
                    if(luma_size > 4) {
	            	  cb_satd_results[35*cuIndex + mode] = (cb_satd_results[35*cuIndex + mode] << 8) | mode;
	            	  cr_satd_results[35*cuIndex + mode] = (cr_satd_results[35*cuIndex + mode] << 8) | mode;
                    }
	            }
	            std::sort(&(y_satd_results[35*cuIndex]), &(y_satd_results[35*cuIndex+35]));
                if(luma_size > 4) {
	               std::sort(&(cb_satd_results[35*cuIndex]), &(cb_satd_results[35*cuIndex+35]));
	               std::sort(&(cr_satd_results[35*cuIndex]), &(cr_satd_results[35*cuIndex+35]));
                }
	            for(int mode = 0; mode < 35; mode++) {
	            	y_modes[35*cuIndex+mode] = (y_satd_results[35*cuIndex+mode] & 0xFF);
	            	y_satd_results[35*cuIndex+mode] >>= 8;
                    if(luma_size > 4) {
	            	  cb_modes[35*cuIndex+mode] = (cb_satd_results[35*cuIndex+mode] & 0xFF);
	            	  cb_satd_results[35*cuIndex+mode] >>= 8;
	            	  cr_modes[35*cuIndex+mode] = (cr_satd_results[35*cuIndex+mode] & 0xFF);
	            	  cr_satd_results[35*cuIndex+mode] >>= 8;
                    }
	            }
	        }
#ifdef MODE_HIST
	        int ymode_hist[35], cbmode_hist[35], crmode_hist[35];
	        for(int l = 0; l < 35; l++) {
	        	ymode_hist[l] = cbmode_hist[l] = crmode_hist[l] = 0;
	        }
	        for(int l = 0; l < (35*((param.sourceWidth/luma_size)*(param.sourceHeight/luma_size))); l += 1) { //Change to += 35 for histogram of best modes
	        	ymode_hist[y_modes[l]]++;
                if(luma_size > 4) {
	        	  cbmode_hist[cb_modes[l]]++;
	        	  crmode_hist[cr_modes[l]]++;
                }
	        }
	        printf("ymode hist: ");
	        for(int l = 0; l < 35; l++)
	        	printf("%d ", ymode_hist[l]);
            if(luma_size > 4) {
    	        printf("\ncbmode hist: ");
    	        for(int l = 0; l < 35; l++)
    	        	printf("%d ", cbmode_hist[l]);
    	        printf("\ncrmode hist: ");
    	        for(int l = 0; l < 35; l++)
    	        	printf("%d ", crmode_hist[l]);
            }
	        printf("\n");
#endif
	        cur_result++;
	    }
	}
	return ret;
}

//TODO sort student results by satd result *and* mode number to make sure we have *exactly* the same bytes in
//both arrays, even if several modes have the same SATD value.
//We want to do the sort here so that students are not required to (it's not necessary in a real x265 use case).
bool ece408_compare(ece408_intra_pred_result *ref, ece408_intra_pred_result *student, int num_frames) {
	if(student == NULL) {
		printf("Student result array pointer is NULL\n");
		return false;
	}
	for(int i = 0; i < (4*num_frames); i++) {
		if(ref[i].luma_block_size != student[i].luma_block_size) {
			printf("Ref result %d luma block size = %d, student = %d\n", i, ref[i].luma_block_size, student[i].luma_block_size);
			return false;
		}
		if(ref[i].num_blocks != student[i].num_blocks) {
			printf("Ref result %d num_blocks = %d, student = %d\n", i, ref[i].num_blocks, student[i].num_blocks);
			return false;
		}
		if(memcmp(ref[i].y_modes, student[i].y_modes, 35*ref[i].num_blocks*sizeof(*ref[i].y_modes))) {
			printf("Result %d, ref and student y_modes mismatched\n", i);
			return false;
		}
		if(memcmp(ref[i].y_satd_results, student[i].y_satd_results, 35*ref[i].num_blocks*sizeof(*ref[i].y_satd_results))) {
			printf("Result %d, ref and student y_satd_results mismatched\n", i);
			return false;
		}
		if(ref[i].luma_block_size > 4) {
			if(memcmp(ref[i].cb_modes, student[i].cb_modes, 35*ref[i].num_blocks*sizeof(*ref[i].cb_modes))) {
				printf("Result %d, ref and student cb_modes mismatched\n", i);
				return false;
			}
			if(memcmp(ref[i].cb_satd_results, student[i].cb_satd_results, 35*ref[i].num_blocks*sizeof(*ref[i].cb_satd_results))) {
				printf("Result %d, ref and student cb_satd_results mismatched\n", i);
				return false;
			}
			if(memcmp(ref[i].cr_modes, student[i].cr_modes, 35*ref[i].num_blocks*sizeof(*ref[i].cr_modes))) {
				printf("Result %d, ref and student cr_modes mismatched\n", i);
				return false;
			}
			if(memcmp(ref[i].cr_satd_results, student[i].cr_satd_results, 35*ref[i].num_blocks*sizeof(*ref[i].cr_satd_results))) {
				printf("Result %d, ref and student cr_satd_results mismatched\n", i);
				return false;
			}
		}
	}
	return true;
}

ece408_intra_pred_result *ece408_competition(ece408_frame *imgs, int num_frames) {
	//Fill in your own!
	(void)imgs;
	ece408_intra_pred_result *ret = new ece408_intra_pred_result[4*num_frames]; //8x8,16x16,32x32,64x64
	ece408_intra_pred_result *cur_result = ret;

        cudaError_t cuda_ret;

        uint8_t *d_y,
                *d_cr,
                *d_cb;

        unsigned int y_size = ((imgs->width) * (imgs->height));
        unsigned int cr_size,
                     cb_size;

        // TO DO : do we need a ceil here ? 
        cr_size = cb_size = (y_size/2);

        // Allocate global memorcy for y, cr, cb components of the frame
        cuda_ret = cudaMalloc((void **) &d_y, y_size);
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaMalloc((void **) &d_cr, cr_size);
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaMalloc((void **) &d_cb, cb_size);
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaDeviceSynchronize();
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaMemcpy(d_y, imgs->y, y_size, cudaMemcpyHostToDevice);
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaMemcpy(d_cr, imgs->cr, cr_size, cudaMemcpyHostToDevice);
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaMemcpy(d_cb, imgs->cb, cb_size, cudaMemcpyHostToDevice);
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaDeviceSynchronize();
        if ( cuda_ret != cudaSuccess )
        {
            printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

	for(int i = 0; i < num_frames; i++) {
		for(int luma_size_shift = 2; luma_size_shift <= 5; luma_size_shift++) {
	        int luma_size = 1 << luma_size_shift; // luma_size x luma_size luma PBs
	        cur_result->create(32, 32, luma_size);

                // Start
 
                int32_t *d_res_y;
                int32_t *d_res_cr;
                int32_t *d_res_cb;
                uint8_t *d_y_modes;
                uint8_t *d_cr_modes;
                uint8_t *d_cb_modes;

                unsigned int y_res_size = (35 * (cur_result->num_blocks)); 
                unsigned int cr_res_size,
                             cb_res_size;

                cr_res_size = cb_res_size = y_res_size;

                // Allocate result in the device 
                cuda_ret = cudaMalloc((void **) &d_res_y, y_res_size);
                if ( cuda_ret != cudaSuccess )
                {
                    printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                cuda_ret = cudaMalloc((void **) &d_y_modes, y_res_size);
                if ( cuda_ret != cudaSuccess )
                {
                    printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                if ( luma_size > 4 )
                {
                    cuda_ret = cudaMalloc((void **) &d_res_cr, cr_res_size);
                    if ( cuda_ret != cudaSuccess )
                    {
                        printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                        exit(EXIT_FAILURE);
                    }

                    cuda_ret = cudaMalloc((void **) &d_res_cb, cb_res_size);
                    if ( cuda_ret != cudaSuccess )
                    {
                        printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                        exit(EXIT_FAILURE);
                    }

                    cuda_ret = cudaMalloc((void **) &d_cr_modes, cr_res_size);
                    if ( cuda_ret != cudaSuccess )
                    {
                        printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                        exit(EXIT_FAILURE);
                    }

                    cuda_ret = cudaMalloc((void **) &d_cb_modes, cb_res_size);
                    if ( cuda_ret != cudaSuccess )
                    {
                        printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                        exit(EXIT_FAILURE);
                    }
                }

                cuda_ret = cudaDeviceSynchronize();
                if ( cuda_ret != cudaSuccess )
                {
                    printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                // Grid dimension
                dim3 dimGrid = dim3((int)ceil((imgs->width)/(float)luma_size), (int)ceil((imgs->height)/(float)luma_size), 1);

                // Block dimension
                dim3 dimBlock = dim3(luma_size, luma_size, 1);

                int neighbour_array_size = luma_size*2+1;
                hevcPredictionKernel<<<dimGrid, dimBlock, (neighbour_array_size * sizeof(uint8_t))>>>(d_y, d_cr, d_cb, d_res_y, d_res_cr, d_res_cb, d_y_modes, d_cr_modes, d_cb_modes, imgs->height, imgs->width);

                cuda_ret = cudaDeviceSynchronize();
                if ( cuda_ret != cudaSuccess )
                {
                    printf("\n%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

	        cur_result++;
	    }
	}
	return ret;
}
