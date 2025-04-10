

function [points] = optimal_points(x_bs, y_bs, x_c, y_c, P_bs, P_uav, ...
    bw_bs, bw_uav, h_uav, h_bs, h_relay, capacity_thresh, var_n)

   
    d = abs(sqrt((x_c - x_bs)^2 + (y_c -  y_bs)^2));
    theta = atan2((y_c - y_bs), (x_c - x_bs));

   
    syms x
    capacity_bs = bw_bs*log(1 + P_bs/((x^2 + (h_relay-h_bs)^2) * var_n));
    capacity_uav = bw_uav*log(1 + P_uav/(((x-d)^2 + (h_uav-h_relay)^2) * var_n));
    
    
    intersection_bs_thresh = vpasolve(capacity_bs == capacity_thresh, x, ...
        [0, d]);
    intersection_uav_thresh = vpasolve(capacity_uav == capacity_thresh, x, ...
        [0, d]);
    intersection_both_capacities = vpasolve(capacity_uav == capacity_bs, x, ...
        [0, d]);
    val_intersection_both_capacities = subs(capacity_uav, intersection_both_capacities);
  
    if (val_rintesection_both_capacities > capacity_thresh)

      x1 = x_bs + cos(theta) * intersection_both_capacities;
        y1 = y_bs + sin(theta) * intersection_both_capacities;
        points = [x1, y1; x1, y1];
    elseif (isempty(intersection_bs_thresh) || isempty(intersection_uav_thresh))

        points = [x_bs, y_bs; x_bs, y_bs];
    else
        x1 = x_bs + cos(theta) * intersection_bs_thresh;
        y1 = y_bs + sin(theta) * intersection_bs_thresh;
        x2 = x_bs + cos(theta) * intersection_uav_thresh;
        y2 = y_bs + sin(theta) * intersection_uav_thresh;
        points = [x1, y1; x2, y2];
    end