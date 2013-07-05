clear all
clc
close all

%---------------------------------------------------------------------------------------------------------------------
% README 1
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

%---------------------------------------------------------------------------------------------------------------------
% README 2
% Due to a current bug in CUDA, the texture address mode "cudaAddressModeBorder"
% does not yield zeros when data outside the texture is requested. Due to
% this the errors for textures are measured for the valid part of the
% convolution.
%---------------------------------------------------------------------------------------------------------------------


%mex -g Filtering3D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/
mex Filtering3D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

filter_size = 11;
filter = randn(filter_size,filter_size,filter_size);
filter = filter/sum(abs(filter(:)));
volume = randn(256,256,256);
%volume = randn(512,512,512);

% Flip filters for conv3 
% (since conv3 performs correlation and not convolution)
filter_ = flipdim(filter,1);
filter_ = flipdim(filter_,2);
filter_ = flipdim(filter_,3);
tic
filter_response_cpu = conv3(volume,filter_);
toc

[filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,0);

imagesc([filter_response_cpu(:,:,round(size(volume,3)/2)) filter_response_gpu_texture(:,:,round(size(volume,3)/2)) filter_response_gpu_texture_unrolled(:,:,round(size(volume,3)/2)) filter_response_gpu_shared(:,:,round(size(volume,3)/2)) filter_response_gpu_shared_unrolled(:,:,round(size(volume,3)/2)) ])

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
