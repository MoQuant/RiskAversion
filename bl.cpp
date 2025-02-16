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

std::string polygon_data(std::string ticker, std::string T0, std::string T1)
{
    // Paste in your own polygon api key
    return "https://api.polygon.io/v2/aggs/ticker/" + ticker + "/range/1/day/" + T0 + "/" + T1 + "?adjusted=true&sort=asc&limit=400&apiKey=";
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output)
{
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

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

std::map<std::string, std::vector<double>> Typhoon(std::vector<std::string> tickers, std::string T0, std::string T1)
{
    std::map<std::string, std::vector<double>> result;
    for(auto & tick : tickers){
        std::string response = GET(polygon_data(tick, T0, T1));
        std::stringstream ss(response);
        ptree df;
        read_json(ss, df);

        for(ptree::const_iterator it = df.begin(); it != df.end(); ++it){
            if(it->first == "results"){
                for(ptree::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt){
                    for(ptree::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt){
                        if(kt->first == "l"){
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

void dims(std::vector<std::vector<double>> x){
    std::cout << "Matrix: (" << x.size() << "," << x[0].size() << ")" << std::endl;
}

void printer(std::vector<std::vector<double>> x){
    for(auto & t : x){
        for(auto & q : t){
            std::cout << q << "\t";
        }
        std::cout << std::endl;
    }
    dims(x);
}

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

std::vector<std::vector<double>> FACTOR(double a, std::vector<std::vector<double>> matrix){
    for(int i = 0; i < matrix.size(); ++i){
        for(int j = 0; j < matrix[0].size(); ++j){
            matrix[i][j] *= a;
        }
    }
    return matrix;
}

std::vector<std::vector<double>> ADD_SAME(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            x[i][j] += y[i][j];
        }
    }
    return x;
}

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

std::vector<double> BlackLitterman(double risk, std::vector<std::vector<double>> ror){
    auto v2m = [](std::vector<double> xi){
        std::vector<std::vector<double>> y;
        for(auto & i : xi){
            y.push_back({i});
        }
        return y;
    };

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

    int m = ror.size(), n = ror[0].size();

    for(int i = 0; i < n; ++i){
        init_weights.push_back({1/(double) n});
    }

    for(int i = 0; i < m; ++i){
        ones.push_back({1.0});
    }

    P = {{1, 0, 0, -1, 0, 0, 0},
         {0, -1, 0, 0, 1, 0, 0},
         {0, 0, 1, -1, 0, 0, 0},
         {1, 0, 0, 0, -1, 0, 0},
         {0, 0, 0, -1, 1, 0, 0},
         {-1, 0, 0, 0, 0, 0, 1},
         {0, 0, 0, 0, 1, -1, 0}};

    Q = FACTOR((1/(double) m), MMULT(TRANSPOSE(ones), ror));
    
    for(int i = 0; i < m; ++i){
        tempxmu.clear();
        for(int j = 0; j < n; ++j){
            tempxmu.push_back(ror[i][j] - Q[0][j]);
        }
        XMU.push_back(tempxmu);
    }

    sigma = MMULT(TRANSPOSE(XMU), XMU);

    pi = FACTOR(risk, MMULT(sigma, init_weights));
    omega = MMULT(P, MMULT(FACTOR(tau, sigma), TRANSPOSE(P)));
    for(int i = 0; i < omega.size(); ++i){
        for(int j = 0; j < omega.size(); ++j){
            if(i != j){
                omega[i][j] = 0.0;
            }
        }
    }

    std::vector<std::vector<double>> B, A;

    B = ADD_SAME(MMULT(INVERSE(FACTOR(tau, sigma)), pi), MMULT(TRANSPOSE(P), MMULT(INVERSE(omega), TRANSPOSE(Q))));
    A = ADD_SAME(INVERSE(FACTOR(tau, sigma)), MMULT(TRANSPOSE(P), MMULT(INVERSE(omega), P)));

    ER = MMULT(INVERSE(A), B);

    wm = FACTOR(1.0/risk, MMULT(INVERSE(sigma), ER));

    result = m2v(wm);

    return result;
}

int main()
{
    std::string T0 = "2024-01-01", T1 = "2025-01-14";

    std::vector<std::string> tickers = {"NOC","MSFT","NVDA","PLTR","ORCL","GOOGL","VZ"};

    std::map<std::string, std::vector<double>> stock_data = Typhoon(tickers, T0, T1);
    
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

    ror = TRANSPOSE(ror);

    std::vector<double> weights = BlackLitterman(2.5, ror);
    
    for(int i = 0; i < weights.size(); ++i){
        std::cout << tickers[i] << ": " << weights[i] << std::endl;
    }
    


    return 0;
}