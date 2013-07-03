clc
FILTER_SIZE = 5;

x_offset = -(FILTER_SIZE - 1)/2;
for column = FILTER_SIZE - 1:-1:0
    %clc
    y_offset = -(FILTER_SIZE - 1)/2;
	for row = FILTER_SIZE - 1:-1:0
            
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

            code = ['    sum += tex2D(tex_Image, x ' sign_x ' ' number_x '.0f + 0.5f, y ' sign_y ' ' number_y '.0f + 0.5f) * c_Filter_' num2str(FILTER_SIZE) 'x' num2str(FILTER_SIZE) '[' num2str(row) '][' num2str(column) '];' ];
            disp(code)   
                        			
		y_offset = y_offset + 1;
    end
	x_offset = x_offset + 1;
    %pause
end