clear all
clc
close all

mex Filtering3D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

filter_size = 17;
filter = randn(filter_size,filter_size,filter_size);
filter = filter/sum(abs(filter(:)));
volume = randn(512,512,512);

filter_response_cpu = convn(volume,filter,'same');

[filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,0);

imagesc([filter_response_cpu(:,:,size(volume,3)/2) filter_response_gpu_texture(:,:,size(volume,3)/2) filter_response_gpu_texture_unrolled(:,:,size(volume,3)/2) filter_response_gpu_shared(:,:,size(volume,3)/2) filter_response_gpu_shared_unrolled(:,:,size(volume,3)/2) ])

texture_tot = sum(abs(filter_response_cpu(:) - filter_response_gpu_texture(:)))
texture_max = max(abs(filter_response_cpu(:) - filter_response_gpu_texture(:)))

texture_tot_valid = sum(sum(sum(abs(filter_response_cpu((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2) - filter_response_gpu_texture((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2))))) 
texture_max_valid = max(max(max(abs(filter_response_cpu((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2) - filter_response_gpu_texture((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2)))))

texture_unrolled_tot = sum(abs(filter_response_cpu(:) - filter_response_gpu_texture_unrolled(:)))
texture_unrolled_max = max(abs(filter_response_cpu(:) - filter_response_gpu_texture_unrolled(:)))

texture_unrolled_tot_valid = sum(sum(sum(abs(filter_response_cpu((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2) - filter_response_gpu_texture_unrolled((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2)))))
texture_unrolled_max_valid = max(max(max(abs(filter_response_cpu((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2) - filter_response_gpu_texture_unrolled((filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2,(filter_size-1)/2+1:end-(filter_size-1)/2)))))

shared_tot = sum(abs(filter_response_cpu(:) - filter_response_gpu_shared(:)))
shared_max = max(abs(filter_response_cpu(:) - filter_response_gpu_shared(:)))

shared_unrolled_tot = sum(abs(filter_response_cpu(:) - filter_response_gpu_shared_unrolled(:)))
shared_unrolled_max = max(abs(filter_response_cpu(:) - filter_response_gpu_shared_unrolled(:)))

time_texture
time_texture_unrolled
time_shared
time_shared_unrolled
time_fft

%figure; imagesc(abs(filter_response_cpu - filter_response_gpu))