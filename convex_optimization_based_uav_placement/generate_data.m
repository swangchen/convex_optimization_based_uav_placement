

function [data] = generate_data(num_of_clusters, start_range_mean, ... 
    end_range_mean, start_range_var, end_range_var, data_points_per_cluster)

   

    data = [];

    for i=1:num_of_clusters
        
        
        mu = start_range_mean + (end_range_mean - start_range_mean).* ...
            rand(2,1);


        while (true)
        
          A = -1 + 2.*rand(2, 2);
          if (rank(A) == 2);
           
              break; 
          end    
        end
        sigma = A' * A;
        sigma = start_range_var + (end_range_var - start_range_var).*sigma;

 
        data_per_cluster = mvnrnd(mu, sigma, data_points_per_cluster);
        data = [data; data_per_cluster];
    end