clear all
clc
close all

mex Filtering3D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

% Loop over filter sizes

% volume = randn(256,256,256);
% 
% N = length(3:2:17);
% texture_times = zeros(N,1);
% texture_times_unrolled = zeros(N,1);
% shared_times = zeros(N,1);
% shared_times_unrolled = zeros(N,1);
% fft_times = zeros(N,1);
% megavoxels = zeros(N,1);
% 
% sizes = 3:2:17;
% 
% % HALO 4
% i = 1;
% for size = 3:2:9
%     filter = randn(size,size,size);
%     filter = filter/sum(abs(filter(:)));    
%     [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled time_fft]  = Filtering3D(volume,filter,1);
%     texture_times(i) = time_texture;
%     texture_times_unrolled(i) = time_texture_unrolled;
%     shared_times(i) = time_shared;
%     shared_times_unrolled(i) = time_shared_unrolled;    
%     fft_times(i) = time_fft;
%     megavoxels(i) = 256*256*256/1000000;
%     i = i + 1;
% end
% 
% 
% % HALO 8
% for size = 11:2:17
%     filter = randn(size,size,size);
%     filter = filter/sum(abs(filter(:)));    
%     [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled time_fft]  = Filtering3D(volume,filter,1);
%     texture_times(i) = time_texture;
%     texture_times_unrolled(i) = time_texture_unrolled;
%     shared_times(i) = time_shared;
%     shared_times_unrolled(i) = time_shared_unrolled;    
%     fft_times(i) = time_fft;
%     megavoxels(i) = 256*256*256/1000000;
%     i = i + 1;
% end
% 
% 
% figure
% plot(sizes,megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
% hold on
% plot(sizes,megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
% hold on
% plot(sizes,megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
% hold on
% plot(sizes,megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
% hold on
% plot(sizes,megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
% hold off
% set(gca,'FontSize',15)
% xlabel('Filter size','FontSize',15)
% ylabel('Megavoxels / second','FontSize',15)
% legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')
% 
% print -dpng benchmark_3D_filtering_filter_sizes_volume_size_256x256x256.png

%%
% Loop over volume sizes for filter size 9 x 9 x 9
% HALO 4
% filter = randn(9,9,9);
% filter = filter/sum(abs(filter(:)));
% 
% N = length(64:32:512);
% texture_times = zeros(N,1);
% texture_times_unrolled = zeros(N,1);
% shared_times = zeros(N,1);
% shared_times_unrolled = zeros(N,1);
% fft_times = zeros(N,1);
% megavoxels = zeros(N,1);
% 
% sizes = 64:32:512;
% 
% i = 1;
% for size = sizes
%     volume = randn(size,size,size);
%     [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,1);
%     texture_times(i) = time_texture;
%     texture_times_unrolled(i) = time_texture_unrolled;
%     shared_times(i) = time_shared;
%     shared_times_unrolled(i) = time_shared_unrolled;   
%     fft_times(i) = time_fft;
%     megavoxels(i) = size*size*size/1000000;
%     i = i + 1;
% end
% 
% close all
% figure
% plot(sizes, megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
% hold on
% plot(sizes, megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
% hold on
% plot(sizes, megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
% hold on
% plot(sizes, megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
% hold on
% plot(sizes, megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
% hold off
% set(gca,'FontSize',15)
% xlabel('Volume size','FontSize',15)
% ylabel('Megavoxels / second','FontSize',15)
% legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')
% 
% print -dpng benchmark_3D_filtering_volume_sizes_9x9x9.png

%%

% Loop over volume sizes for filter size 17 x 17 x 17
% HALO 8
filter = randn(17,17,17);
filter = filter/sum(abs(filter(:)));

N = length(64:32:512);
texture_times = zeros(N,1);
texture_times_unrolled = zeros(N,1);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

sizes = 64:32:448;

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
%plot(sizes,megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
%hold on
%plot(sizes,megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
%hold on
plot(sizes,megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(sizes,megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(sizes,megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Volume size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
%legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')
legend('Shared','Shared unrolled','FFT')

print -dpng benchmark_3D_filtering_volume_sizes_17x17x17.png

