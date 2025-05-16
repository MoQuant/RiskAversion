#include <iostream>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <math.h>
#include <map>
#include <sstream>

using namespace boost::property_tree;

// Polygon.io stock price API
std::string polygon_data(std::string ticker, std::string T0, std::string T1)
{
    // Paste in your own polygon api key
    return "https://api.polygon.io/v2/aggs/ticker/" + ticker + "/range/1/day/" + T0 + "/" + T1 + "?adjusted=true&sort=asc&limit=400&apiKey=";
}

// Conversion of bytes to string callback function
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output)
{
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

// Get Request Function
std::string GET(const std::string& url)
{
    CURL* curl;
    CURLcode res;
    std::string response;

    curl = curl_easy_init();
    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        curl_easy_cleanup(curl);
    }
    else
    {
        std::cerr << "Failed to initialize CURL" << std::endl;
    }
    return response;
}

// Fetches all of the stock prices with given stock tickers
std::map<std::string, std::vector<double>> Typhoon(std::vector<std::string> tickers, std::string T0, std::string T1)
{
    std::map<std::string, std::vector<double>> result;
    for(auto & tick : tickers){
        // Returns string response
        std::string response = GET(polygon_data(tick, T0, T1));
        std::stringstream ss(response);
        ptree df;

        // Parse string response into JSON using Boost
        read_json(ss, df);

        for(ptree::const_iterator it = df.begin(); it != df.end(); ++it){
            if(it->first == "results"){
                for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
                    for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                        if(kt->first == "l"){
                            // Pull all last prices for each stock into a map vector
                            result[tick].push_back(kt->second.get_value<double>());
                        }
                    }
                }
            }
        }
        std::cout << tick << " has imported" << std::endl;
        sleep(2);
    }

    return result;
}

// Print out 2D vectors dimensions
void dims(std::vector<std::vector<double>> x){
    std::cout << "Matrix: (" << x.size() << "," << x[0].size() << ")" << std::endl;
}

// Print out 2D vector
void printer(std::vector<std::vector<double>> x){
    for(auto & t : x){
        for(auto & q : t){
            std::cout << q << "\t";
        }
        std::cout << std::endl;
    }
    dims(x);
}

// Matrix Multiplication Function
std::vector<std::vector<double>> MMULT(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    std::vector<std::vector<double>> z;
    std::vector<double> temp;
    double total = 0;
    for(int i = 0; i < x.size(); ++i){
        temp.clear();
        for(int j = 0; j < y[0].size(); ++j){
            total = 0;
            for(int k = 0; k < y.size(); ++k){
                total += x[i][k]*y[k][j];
            }
            temp.push_back(total);
        }
        z.push_back(temp);
    }
    return z;
}

// Matrix Transpose Function
std::vector<std::vector<double>> TRANSPOSE(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> y;
    std::vector<double> temp;
    for(int i = 0; i < x[0].size(); ++i){
        temp.clear();
        for(int j = 0; j < x.size(); ++j){
            temp.push_back(x[j][i]);
        }
        y.push_back(temp);
    }
    return y;
}

// Multiplies a coeffecient by a 2D Vector (Matrix)
std::vector<std::vector<double>> FACTOR(double a, std::vector<std::vector<double>> matrix){
    for(int i = 0; i < matrix.size(); ++i){
        for(int j = 0; j < matrix[0].size(); ++j){
            matrix[i][j] *= a;
        }
    }
    return matrix;
}

// Adds two matrices together
std::vector<std::vector<double>> ADD_SAME(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            x[i][j] += y[i][j];
        }
    }
    return x;
}

// Calculates the Inverse of a matrix with Gaussian Elimination
std::vector<std::vector<double>> INVERSE(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> I;
    std::vector<double> itemp;
    int n = x.size();
    for(int i = 0; i < n; ++i){
        itemp.clear();
        for(int j = 0; j < n; ++j){
            if(i == j){
                itemp.push_back(1.0);
            } else {
                itemp.push_back(0.0);
            }
        }
        I.push_back(itemp);
    }

    for(int i = 1; i < n; ++i){
        for(int j = 0; j < i; ++j){
            double A = x[i][j];
            double B = x[j][j];
            for(int k = 0; k < n; ++k){
                x[i][k] = x[i][k] - (A/B)*x[j][k];
                I[i][k] = I[i][k] - (A/B)*I[j][k];
            }
        }
    }

    for(int i = 1; i < n; ++i){
        for(int j = 0; j < i; ++j){
            double A = x[j][i];
            double B = x[i][i];
            for(int k = 0; k < n; ++k){
                x[j][k] = x[j][k] - (A/B)*x[i][k];
                I[j][k] = I[j][k] - (A/B)*I[i][k];
            }
        }
    }

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            I[i][j] /= x[i][i];
        }
    }

    return I;
}

// Black-Litterman Portfolio Optimization Function
std::vector<double> BlackLitterman(double risk, std::vector<std::vector<double>> ror){

    // Converts a 1D vector to a 2D singular vector
    auto v2m = [](std::vector<double> xi){
        std::vector<std::vector<double>> y;
        for(auto & i : xi){
            y.push_back({i});
        }
        return y;
    };

    // Converts a 2D singular vector into a 1D vector
    auto m2v = [](std::vector<std::vector<double>> xi){
        std::vector<double> y;
        for(auto & e : xi){
            y.push_back(e[0]);
        }
        return y;
    };
    
    std::vector<std::vector<double>> XMU, pi, sigma, wm, omega, P, Q, ones, ER, init_weights;
    std::vector<double> result, tempxmu;
    double tau = 0.05;

    // Dimensions of dataset 
    int m = ror.size(), n = ror[0].size();

    // Initial weights weighted equally for each stock
    for(int i = 0; i < n; ++i){
        init_weights.push_back({1/(double) n});
    }

    // Builds a vector of ones for the number of rows
    for(int i = 0; i < m; ++i){
        ones.push_back({1.0});
    }

    // This matrix contains which stocks are outperformed by other stocks with -1 signaling underperformance and 1 signaling outperformance
    P = {{1, 0, 0, -1, 0, 0, 0},
         {0, -1, 0, 0, 1, 0, 0},
         {0, 0, 1, -1, 0, 0, 0},
         {1, 0, 0, 0, -1, 0, 0},
         {0, 0, 0, -1, 1, 0, 0},
         {-1, 0, 0, 0, 0, 0, 1},
         {0, 0, 0, 0, 1, -1, 0}};

    // Generate the mean singular matrix vector
    Q = FACTOR((1/(double) m), MMULT(TRANSPOSE(ones), ror));

    // Subtract the mean from the rate of returns and build a new matrix called XMU
    for(int i = 0; i < m; ++i){
        tempxmu.clear();
        for(int j = 0; j < n; ++j){
            tempxmu.push_back(ror[i][j] - Q[0][j]);
        }
        XMU.push_back(tempxmu);
    }

    // Calculate the covariance matrix
    sigma = MMULT(TRANSPOSE(XMU), XMU);

    // Solve for pi (not trig pi) and multiply matrix by risk aversion variable
    pi = FACTOR(risk, MMULT(sigma, init_weights));

    // Solve for omega by multiplying the tau scaling constant by the covariance matrix,
    // and then multiplying the assumptions matrix (P) as P.dot(tauC).dot(P^T).
    // Once the multiplication has happened zero all values of the omega matrix other than the diagonal.
    omega = MMULT(P, MMULT(FACTOR(tau, sigma), TRANSPOSE(P)));
    for(int i = 0; i < omega.size(); ++i){
        for(int j = 0; j < omega.size(); ++j){
            if(i != j){
                omega[i][j] = 0.0;
            }
        }
    }

    std::vector<std::vector<double>> B, A;

    // Solve for the Black-Litterman model
    B = ADD_SAME(MMULT(INVERSE(FACTOR(tau, sigma)), pi), MMULT(TRANSPOSE(P), MMULT(INVERSE(omega), TRANSPOSE(Q))));
    A = ADD_SAME(INVERSE(FACTOR(tau, sigma)), MMULT(TRANSPOSE(P), MMULT(INVERSE(omega), P)));

    // Calculate the Expected Return
    ER = MMULT(INVERSE(A), B);

    // Take the inverse of the covariance matrix and multiply it by the Expected Return vector and divide 
    // the resulting vector by the risk aversion variable in order to solve for the portfolio weights
    wm = FACTOR(1.0/risk, MMULT(INVERSE(sigma), ER));

    // Convert the weights to a 1D vector and return them
    result = m2v(wm);

    return result;
}

int main()
{
    // Time parameters for stock data
    std::string T0 = "2024-01-01", T1 = "2025-01-14";

    // Declare the stocks to be placed into the portfolio
    std::vector<std::string> tickers = {"NOC","MSFT","NVDA","PLTR","ORCL","GOOGL","VZ"};

    // Import all of the stock price data
    std::map<std::string, std::vector<double>> stock_data = Typhoon(tickers, T0, T1);

    // Calculate the rate of returns for all of the stocks in the map
    std::vector<std::vector<double>> ror;
    std::vector<double> tror;
    for(auto & tick : tickers){
        tror.clear();
        std::vector<double> prices = stock_data[tick];
        for(int i = 1; i < prices.size(); ++i){
            tror.push_back(prices[i]/prices[i-1] - 1.0);
        }
        ror.push_back(tror);
    }

    // Transpose the rate of return so each stock is represented by a column rather than a row
    ror = TRANSPOSE(ror);

    // Solve for the optimal portfolio weights with a risk parameter of 2.5
    std::vector<double> weights = BlackLitterman(2.5, ror);

    // Print the weights
    for(int i = 0; i < weights.size(); ++i){
        std::cout << tickers[i] << ": " << weights[i] << std::endl;
    }
    


    return 0;
}
