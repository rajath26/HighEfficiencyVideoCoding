#! /bin/bash

cd ../build/linux

make -j4 
RET=$?

RUN=$1

cp ece408_competition ../../source

cd ../../source


if [ ${RET} -eq 0 ]
then
    if [ ${RUN} -eq 1 ]
    then 
       submitjob ./ece408_competition --input /home/johnso87/install/News_ProRes.yuv --cpuid 1 --input-res 16x16 --input-csp i420 --fps 23.96 -f 2 /dev/null
    fi
fi

