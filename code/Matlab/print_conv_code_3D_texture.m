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

clc
FILTER_SIZE = 17;

x_offset = -(FILTER_SIZE - 1)/2;
for column = FILTER_SIZE - 1:-1:0
    %clc
    y_offset = -(FILTER_SIZE - 1)/2;
    for row = FILTER_SIZE - 1:-1:0
        
        z_offset = -(FILTER_SIZE - 1)/2;
        for rod = FILTER_SIZE - 1:-1:0
            
            if (x_offset ) < 0
                sign_x = '-';
            else
                sign_x = '+';
            end
            
            if (y_offset ) < 0
                sign_y = '-';
            else
                sign_y = '+';
            end
            
            if (z_offset ) < 0
                sign_z = '-';
            else
                sign_z = '+';
            end
            
            if (y_offset ) ~= 0
                number_y = num2str(y_offset / sign(y_offset ));
            else
                number_y = '0';
            end
            
            if (x_offset ) ~= 0
                number_x = num2str(x_offset  / sign(x_offset));
            else
                number_x = '0';
            end
            
            if (z_offset ) ~= 0
                number_z = num2str(z_offset  / sign(z_offset));
            else
                number_z = '0';
            end            
            
            code = ['    sum += tex3D(tex_Volume, x ' sign_x ' ' number_x '.0f + 0.5f, y ' sign_y ' ' number_y '.0f + 0.5f, z ' sign_z ' ' number_z '.0f + 0.5f) * c_Filter_' num2str(FILTER_SIZE) 'x' num2str(FILTER_SIZE) 'x' num2str(FILTER_SIZE) '[' num2str(rod) '][' num2str(row) '][' num2str(column) '];' ];
            disp(code)
            
            z_offset = z_offset + 1;
        end
        y_offset = y_offset + 1;
    end
    x_offset = x_offset + 1;
    %pause
end