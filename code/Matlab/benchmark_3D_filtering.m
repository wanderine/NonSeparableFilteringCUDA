%  	 Non-separable 2D, 3D and 4D Filtering with CUDA
%    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%-----------------------------------------------------------------------------


%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all


mex Filtering3D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

%%
% Loop over filter sizes

volume = randn(256,256,256);

N = length(3:2:17);
texture_times = zeros(N,1);
texture_times_unrolled = zeros(N,1);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

filter_sizes = 3:2:17;

% HALO 4
i = 1;
for filter_size = 3:2:9
    filter = randn(filter_size,filter_size,filter_size);
    filter = filter/sum(abs(filter(:)));    
    [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled time_fft]  = Filtering3D(volume,filter,1);
    texture_times(i) = time_texture;
    texture_times_unrolled(i) = time_texture_unrolled;
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;    
    fft_times(i) = time_fft;
    megavoxels(i) = size(volume,1)*size(volume,2)*size(volume,3)/1000000;
    i = i + 1;
end


% HALO 8
for filter_size = 11:2:17
    filter = randn(filter_size,filter_size,filter_size);
    filter = filter/sum(abs(filter(:)));    
    [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled time_fft]  = Filtering3D(volume,filter,1);
    texture_times(i) = time_texture;
    texture_times_unrolled(i) = time_texture_unrolled;
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;    
    fft_times(i) = time_fft;
    megavoxels(i) = size(volume,1)*size(volume,2)*size(volume,3)/1000000;
    i = i + 1;
end


figure
plot(filter_sizes,megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
hold on
plot(filter_sizes,megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
hold on
plot(filter_sizes,megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(filter_sizes,megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(filter_sizes,megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Filter size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')

print -dpng benchmark_3D_filtering_filter_sizes_volume_size_256x256x256.png

%%
% Loop over volume sizes for filter size 7 x 7 x 7
% HALO 4
filter_size = 7;
filter = randn(filter_size,filter_size,filter_size);
filter = filter/sum(abs(filter(:)));

N = length(64:32:512);
texture_times = zeros(N,1);
texture_times_unrolled = zeros(N,1);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

volume_sizes = 64:32:512;

i = 1;
for volume_size = volume_sizes
    volume = randn(volume_size,volume_size,volume_size);
    [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,1);
    texture_times(i) = time_texture;
    texture_times_unrolled(i) = time_texture_unrolled;
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;   
    fft_times(i) = time_fft;
    megavoxels(i) = volume_size*volume_size*volume_size/1000000;
    i = i + 1;
end

close all
figure
plot(volume_sizes, megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
hold on
plot(volume_sizes, megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
hold on
plot(volume_sizes, megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(volume_sizes, megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(volume_sizes, megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Volume size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')

print -dpng benchmark_3D_filtering_volume_sizes_7x7x7.png

%%

% Loop over volume sizes for filter size 13 x 13 x 13
% HALO 8
filter_size = 13;
filter = randn(filter_size,filter_size,filter_size);
filter = filter/sum(abs(filter(:)));

N = length(64:32:512);
texture_times = zeros(N,1);
texture_times_unrolled = zeros(N,1);
shared_times = zeros(N,1);
shared_times_unrolled = zeros(N,1);
fft_times = zeros(N,1);
megavoxels = zeros(N,1);

volume_sizes = 64:32:512;

i = 1;
for volume_size = volume_sizes
    volume = randn(volume_size,volume_size,volume_size);
    [filter_response_gpu_texture, filter_response_gpu_texture_unrolled, filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_texture, time_texture_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering3D(volume,filter,1);
    texture_times(i) = time_texture;
    texture_times_unrolled(i) = time_texture_unrolled;
    shared_times(i) = time_shared;
    shared_times_unrolled(i) = time_shared_unrolled;   
    fft_times(i) = time_fft;
    megavoxels(i) = volume_size*volume_size*volume_size/1000000;
    i = i + 1;
end

figure
plot(volume_sizes,megavoxels ./ texture_times * 1000, 'b','LineWidth',2)
hold on
plot(volume_sizes,megavoxels ./ texture_times_unrolled * 1000, 'r','LineWidth',2)
hold on
plot(volume_sizes,megavoxels ./ shared_times * 1000, 'g','LineWidth',2)
hold on
plot(volume_sizes,megavoxels ./ shared_times_unrolled * 1000, 'k','LineWidth',2)
hold on
plot(volume_sizes,megavoxels ./ fft_times * 1000, 'c','LineWidth',2)
hold off
set(gca,'FontSize',15)
xlabel('Volume size','FontSize',15)
ylabel('Megavoxels / second','FontSize',15)
legend('Texture','Texture unrolled','Shared','Shared unrolled','FFT')

print -dpng benchmark_3D_filtering_volume_sizes_13x13x13.png

