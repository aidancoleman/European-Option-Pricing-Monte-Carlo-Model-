#include <stdio.h>
#include <stdlib.h>
#include <string.h> //for string comparison
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_qrng.h> //for quasi random number generation (sobol sequences)
/*=======Monte Carlo pricer for a European option=======================
 
 Simulate one or more normal or log-normal variants which represent market data points or risk factors, calculate expected payoff under filtration, payoff under each possible path.
 Call option payoff (European option):
 Max(S(T) - Strike, 0).... S(T) is the spot price of the underlying asset to which the option contract pertains at expiry
 
 Put option payoff (European option):
 Max(Strike - S(T), 0)
 
 The Black-Scholes assumption: the price of an asset at expiry follows geometric Brownian motion and that the log of the price of the stock is a random walk process. Thus, formula for S(T):
 
 S(T) = S(t_0)exp{(r-(1/2)(sigma)^2)*tau + sigma*sqrt{tau}*z}, where z is a standard normal variate and tau = T - t_0 (time to expiry of option)
 
 n = number of Monte Carlo samples
 st_0 = Spot price of underlying asset today
 strike = Strike price of option
 sigma = Annual volatility of underlying asset
 r = interest rate
 prng_type =  user inputs pseudo random number generation type to apply: "taus" for Tausworthe, "sobol" for quasi RNG sobol sequence between 0 and 1, or default to the mersenne twister
 var_reduc = user inputs "antithetic" if they wish to use antithetic variates for MC estimate variance reduction
 option_type = user inputs whether the European option is a call or put
 
 */
double mc_option_pricer(const int n, const double st_0, const double strike, const double sigma, double r, const char* prng_type, const char* var_reduc, const char* option_type){
    int tau = 1; //assume option expires in one year
    double a = (r - (0.5)*(sigma)*(sigma))*tau;
    double b = (sigma*sqrt(tau));
    int antithetic = (strcmp(var_reduc, "antithetic")==0); //set to 1 if true, 0 otherwise
    int sobol = (strcmp(prng_type, "sobol")==0);
    gsl_rng* rng = NULL; //null pointers for rng and qrng, we decide which to use based on command line arguments passed
    gsl_qrng* q = NULL;
    if(sobol){
        q = gsl_qrng_alloc(gsl_qrng_sobol, 2); //if sobol is true, use gsl_qrng_alloc(gsl_qrng_sobol, 2) to allocate space for two sobol sequence numbers uniformly distributed from 0 to 1 to use in our polar algorithm to generate standard normal variates z1 and z2
    }else {
        if(strcmp(prng_type, "taus")==0){
            rng = gsl_rng_alloc(gsl_rng_taus); //set rng to gsl_rng_alloc(gsl_rng_taus) to Tausworthe prng algo
        }else{
            rng = gsl_rng_alloc(gsl_rng_mt19937); //else default to Mersenne Twister for prng
        }
        unsigned long seed = 7;
        gsl_rng_set(rng, seed); //set seed for prng
    }
    
    int samples = 0;
    double total_payoff = 0.0;
    while (samples < n){
        double u1, u2, v1, v2, z1, z2, s;
        do{
        if(sobol){
            double u[2];
            gsl_qrng_get(q, u); //use gsl_qrng_get to get a 2 dimensional sobol sequence object from q and store it in the array u
            u1 = u[0];
            u2 = u[1];
        }
        else{
            u1 = gsl_rng_uniform(rng);
            u2 = gsl_rng_uniform(rng);
        }
                v1 = 2.0*u1 - 1.0;
                v2 = 2.0*u2 - 1.0;
                s = (v1)*(v1) + (v2)*(v2);
            }while(s > 1.0);
        z1 = v1*(sqrt(-2.0*log(s)/s));
        z2 = v2*(sqrt(-2.0*log(s)/s));
        if(antithetic){
            z2 = -z1; //antithetic variates if desired
        }
        //now have two pair of standard normal random variates
       // first Monte Carlo sample using z1
        double path_payoff = 0.0;
        {
            double S_T = st_0 * exp(a + b * z1);
            if(strcmp(option_type, "call")==0){
                path_payoff = (S_T - strike) > 0 ? (S_T - strike) : 0.0; //call option payoff
            }else{
                path_payoff = (strike - S_T) > 0 ? (strike - S_T) : 0.0; //put option payoff
            }
            total_payoff += path_payoff;
            samples++;
        }
        
        // If we haven't reached n samples, use z2 for a second sample
        if (samples < n) {
            double S_T = st_0 * exp(a + b * z2);
            if(strcmp(option_type, "call")==0){
                path_payoff = (S_T - strike) > 0 ? (S_T - strike) : 0.0; //call option payoff
            }else{
                path_payoff = (strike - S_T) > 0 ? (strike - S_T) : 0.0; //put option payoff
            }
            total_payoff += path_payoff;
            samples++;
        }
    }
    if(sobol)
        gsl_qrng_free(q);
    else
        gsl_rng_free(rng);

    // Discount the average payoff back to present value
    double present_value = exp(-r * tau) * (total_payoff / n);
    return present_value;
}

int main(void) {
    int n = 10000;
    double st_0 = 100.0;
    double strike = 105.0;
    double sigma = 0.2;
    double r = 0.3;
    
    // Supply the variance reduction, prng/qrng, and option type parameters:
    const char* prng_type = "sobol";        // Options: "taus", "sobol", or "mt"
    const char* var_reduc = "antithetic";  // Options: "antithetic" or "none"
    const char* option_type = "call";      // Options: "call" or "put"
    
    double price = mc_option_pricer(n, st_0, strike, sigma, r, prng_type, var_reduc, option_type);
    printf("European %s Option Price: %f\n", option_type, price);
    return 0;
}
