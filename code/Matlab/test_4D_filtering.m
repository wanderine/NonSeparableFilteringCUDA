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

mex Filtering4D.cpp -lcudart -lcufft -lFilteringCUDA -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/Filtering/

filter_size = 11;
filter = randn(filter_size,filter_size,filter_size,filter_size);
filter = filter/sum(abs(filter(:)));
volumes = randn(128,128,128,32);
%volumes = randn(129,129,129,33);

tic
filter_response_cpu = convn(volumes,filter,'same');
toc

[filter_response_gpu_shared, filter_response_gpu_shared_unrolled, time_shared, time_shared_unrolled, time_fft]  = Filtering4D(volumes,filter,0);

imagesc([filter_response_cpu(:,:,round(size(volumes,3)/2),round(size(volumes,4)/2)) filter_response_gpu_shared(:,:,round(size(volumes,3)/2),round(size(volumes,4)/2)) filter_response_gpu_shared_unrolled(:,:,round(size(volumes,3)/2),round(size(volumes,4)/2)) ])

shared_tot = sum(abs(filter_response_cpu(:) - filter_response_gpu_shared(:)))
shared_max = max(abs(filter_response_cpu(:) - filter_response_gpu_shared(:)))

shared_unrolled_tot = sum(abs(filter_response_cpu(:) - filter_response_gpu_shared_unrolled(:)))
shared_unrolled_max = max(abs(filter_response_cpu(:) - filter_response_gpu_shared_unrolled(:)))


time_shared
time_shared_unrolled
time_fft
