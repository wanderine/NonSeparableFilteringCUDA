clear all
clc
close all

%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------


mex Filtering4D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

%%
% Loop over filter sizes

volumes = randn(128,128,128,32);

N = length(3:2:17);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

% HALO 4
i = 1;
for filter_size = 3:2:9
    filter = randn(filter_size,filter_size,filter_size,filter_size);
    filter = filter/sum(abs(filter(:)));    
    [filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_shared, time_shared_unrolled time_fft]  = Filtering4D(volumes,filter,1);
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;    
    fft_times(i) = time_fft;
    megavoxels(i) = size(volumes,1)*size(volumes,2)*size(volumes,3)*size(volumes,4)/1000000;
    i = i + 1;
end


% HALO 8
for filter_size = 11:2:17
    filter = randn(filter_size,filter_size,filter_size,filter_size);
    filter = filter/sum(abs(filter(:)));    
    [filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_shared, time_shared_unrolled time_fft]  = Filtering4D(volumes,filter,1);
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;    
    fft_times(i) = time_fft;
    megavoxels(i) = size(volumes,1)*size(volumes,2)*size(volumes,3)*size(volumes,4)/1000000;
    i = i + 1;
end


figure
plot(sizes,megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(sizes,megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(sizes,megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Filter size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
legend('Shared','Shared unrolled','FFT')

print -dpng benchmark_4D_filtering_filter_sizes_volume_size_128x128x128x32.png

%%
% Loop over volume sizes for filter size 7 x 7 x 7 x 7
% HALO 4
filter = randn(7,7,7,7);
filter = filter/sum(abs(filter(:)));

N = length(64:32:512);
texture_times = zeros(N,1);
texture_times_unrolled = zeros(N,1);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

sizes = 64:32:512;

i = 1;
for size = sizes
    volume = randn(size,size,size);
    [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,1);
    texture_times(i) = time_texture;
    texture_times_unrolled(i) = time_texture_unrolled;
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;   
    fft_times(i) = time_fft;
    megavoxels(i) = size*size*size/1000000;
    i = i + 1;
end

close all
figure
plot(sizes, megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
hold on
plot(sizes, megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
hold on
plot(sizes, megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(sizes, megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(sizes, megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Volume size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')

print -dpng benchmark_3D_filtering_volume_sizes_7x7x7.png

%%

% Loop over volume sizes for filter size 11 x 11 x 11 x 11 
% HALO 8
filter = randn(11,11,11,11);
filter = filter/sum(abs(filter(:)));

N = length(64:32:512);
texture_times = zeros(N,1);
texture_times_unrolled = zeros(N,1);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

sizes = 64:32:512;

i = 1;
for size = sizes
    volume = randn(size,size,size);
    [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,1);
    texture_times(i) = time_texture;
    texture_times_unrolled(i) = time_texture_unrolled;
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;    
    fft_times(i) = time_fft;
    megavoxels(i) = size*size*size/1000000;
    i = i + 1;
end

figure
plot(sizes,megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
hold on
plot(sizes,megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
hold on
plot(sizes,megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(sizes,megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(sizes,megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Volume size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')

print -dpng benchmark_3D_filtering_volume_sizes_13x13x13.png

