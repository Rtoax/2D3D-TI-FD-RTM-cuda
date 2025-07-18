NVCC := /usr/local/cuda-12.9/bin/nvcc

TARGETS := tti2d_acousticFD
TARGETS += tti2d_acousticRTM
TARGETS += tti2d_elasticFD
TARGETS += vti2d_FD_RTM
TARGETS += vti3Dfd
TARGETS += vti3Drtm

CFLAGS := -Wno-deprecated-gpu-targets
LDFLAGS := -Wno-deprecated-gpu-targets

all: ${TARGETS}

tti2d_acousticFD: Toa_gpu_2dtti_fd_1orderfunction.o
tti2d_elasticFD: Toa_gpu_2dtti_fd_elastic.o
tti2d_acousticRTM: Toa_gpu_2dtti_rtm_adcigs_1orderfunciton.o
vti2d_FD_RTM: Toa_gpu2DvtiFdRtmAdcigsLaplace.o
vti3Dfd: Toa_gpu_3dvti_fd_1orderfunciton.o
vti3Drtm: Toa_gpu_3dvti_rtm_adcigs_1orderfunction.o

%.o: %.cu
	${NVCC} -c -o $(@) $(<) ${CFLAGS}

${TARGETS}: %:
	${NVCC} -o ${@} ${^} ${LDFLAGS}

clean:
	rm -f ${TARGETS} *.o *~
