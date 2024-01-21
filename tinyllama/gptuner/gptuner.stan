functions {

  matrix sqrtexp_kernel(matrix X, real alpha, real rho) {
    
    int N = rows(X);
    matrix[N, N] K;
    for (i in 1:N) {
      for (j in 1:N) {
        K[i, j] = square(alpha)*exp(-dot_self(X[i, :]-X[j, :])/2*square(rho)); 
      }
    }
    return K;
  }  

  matrix sqrtexp_kernel_star(matrix X1, matrix X2, real alpha, real rho) {

    int N_train = rows(X1);
    int N_val = rows(X2);
    matrix[N_train, N_val] K_star;
    
    for (i in 1:N_train) {
      for (j in 1:N_val) {
        K_star[i, j] = square(alpha)*exp(-dot_self(X1[i]-X2[j])/2*square(rho));  
      }
    }
    return K_star;
  }
}


data {

  // number of training samples
  int N_train;

  // Number of test samples
  int N_val;

  // dimension of the hyperparameters
  int M;

  // matrix containing all training samples
  matrix[N_train, M] X_train;

  // matrix containing all test samples
  matrix[N_val, M] X_test;

  // vector containing all losses
  vector[N_train] Y_train;

}


transformed data {

  vector[N_train] mu = rep_vector(0, N_train);

}


parameters {

  real alpha;
  real rho;

}


model {

  alpha ~ std_normal();
  rho ~ inv_gamma(5, 5);

  matrix[N_train, N_train] K;  
  K = sqrtexp_kernel(X_train, alpha, rho);
  # Y_train ~ multi_normal(mu, K);

  // cholesky_decompose: when working with small matrices the differences in computational speed between the two approaches will not be noticeable, but for larger matrices (N≳100) the Cholesky decomposition version will be faster.

  matrix[N_train, N_train] L_K;
  L_K = cholesky_decompose(K);
  Y_train ~ multi_normal_cholesky(mu, L_K);
}

generated quantities {  
  
  matrix[N_train, N_train] K;  
  K = sqrtexp_kernel(X_train, alpha, rho);

  matrix[N_train, N_val] K_star;
  matrix[N_val, N_val] K_star_star;

  K_star = sqrtexp_kernel_star(X_train, X_test, alpha, rho);
  K_star_star = sqrtexp_kernel(X_test, alpha, rho);

  vector[N_val] mu_star;
  matrix[N_val, N_val] sigma_star;

  mu_star = K_star'*inverse(K)*Y_train;
  sigma_star = K_star_star - K_star'*inverse(K)*K_star;

  vector[N_val] Y_test;
  Y_test = multi_normal_rng(mu_star, sigma_star); 

}
