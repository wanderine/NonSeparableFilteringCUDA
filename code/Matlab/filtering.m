clear all
clc
close all

mex FilteringCUDA.cpp -lcudart -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

filter = randn(7,7);
filter = filter/sum(abs(filter(:)));
image = randn(4096,4096);

filter_response_cpu = conv2(image,filter,'same');

filter_response_gpu = FilteringCUDA(image,filter);

%imagesc([filter_response_cpu filter_response_gpu])
figure; imagesc(filter_response_cpu)
figure; imagesc(filter_response_gpu)

sum(abs(filter_response_cpu(:) - filter_response_gpu(:)))
max(abs(filter_response_cpu(:) - filter_response_gpu(:)))
figure; imagesc(abs(filter_response_cpu - filter_response_gpu))