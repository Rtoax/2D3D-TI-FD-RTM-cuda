ALL:
	nvcc -o tti2d_acousticFD   Toa_gpu_2dtti_fd_1orderfunction.cu
	nvcc -o tti2d_elasticFD    Toa_gpu_2dtti_fd_elastic.cu
	nvcc -o tti2d_acousticRTM  Toa_gpu_2dtti_rtm_adcigs_1orderfunciton.cu
	nvcc -o vti2d_FD_RTM       Toa_gpu2DvtiFdRtmAdcigsLaplace.cu
	nvcc -o vti3Dfd Toa_gpu_3dvti_fd_1orderfunciton.cu
	nvcc -o vti3Drtm Toa_gpu_3dvti_rtm_adcigs_1orderfunction.cu
clean:
	rm -f *.o *~
