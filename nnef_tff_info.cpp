#include "nnef.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

template <typename T>
T sqr( const T x )
{
    return x * x;
}

template <typename T>
T relative_data_difference( const size_t n, const T* data1, const T* data2 )
{
    T diff = 0;
    T range = 0;
    for ( size_t i = 0; i < n; ++i )
    {
        diff += sqr(data2[i] - data1[i]);
        range += sqr(data1[i]);
    }
    return std::sqrt(diff / range);
}

int volume(const nnef::Tensor& tensor)
{
    const std::vector<int>& shape = tensor.shape;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <typename T>
T relative_difference( const nnef::Tensor& tensor1, const nnef::Tensor& tensor2 )
{
    size_t volume1 = volume(tensor1);
    size_t volume2 = volume(tensor2);
    return relative_data_difference(std::min(volume1, volume2),
        (const T*) tensor1.data.data(),
        (const T*) tensor2.data.data());
}

void print_tensor_header( std::ostream& os, const nnef::Tensor& tensor )
{
    os << tensor.dtype << std::endl;
    for ( size_t i = 0; i < tensor.shape.size(); ++i )
    {
        if ( i )
        {
            os << " ";
        }
        os << "1.." << tensor.shape[i];
    }
    os << std::endl;
}

template <typename T>
void print_data( std::ostream& os, const size_t n, const T* data )
{
    for ( size_t i = 0; i < n; ++i )
    {
        os << data[i] << std::endl;
    }
}

void print_tensor_data( std::ostream& os, const nnef::Tensor& tensor )
{
    size_t vol = volume(tensor);
    if ( tensor.dtype == "scalar" )
    {
        print_data(os, vol, (float *)tensor.data.data());
    }
    else if ( tensor.dtype == "integer" )
    {
        print_data(os, vol, (int *)tensor.data.data());
    }
    else if ( tensor.dtype == "boolean" )
    {
        print_data(os, vol, (bool *)tensor.data.data());
    }
}

std::ostream& operator<<( std::ostream& os, const nnef::Tensor& tensor )
{
    print_tensor_header(os, tensor);
    print_tensor_data(os, tensor);
    return os;
}

int main( int argc, const char * argv[] )
{
    std::string error;

    if ( argc == 2 )
    {
        nnef::Tensor tensor;
        if ( !nnef::read_tensor(argv[1], tensor, error) )
        {
            std::cerr << error << std::endl;
            return -1;
        }
        std::cout << tensor;
    }
    else if ( argc == 3 )
    {
        nnef::Tensor tensor1, tensor2;
        if ( !nnef::read_tensor(argv[1], tensor1, error) )
        {
	    std::cerr << error << std::endl;
            return -2;
        }
        if ( !nnef::read_tensor(argv[2], tensor2, error) )
        {
            std::cerr << error << std::endl;
            return -3;
        }
        float diff = relative_difference<float>(tensor1, tensor2);
        std::cout << "tensor #1:" << std::endl;
        print_tensor_header(std::cout, tensor1);
        std::cout << "tensor #2:" << std::endl;
        print_tensor_header(std::cout, tensor2);
        std::cout << "relative difference:" << std::endl;
        std::cout << diff << std::endl;
    }
    else
    {
        std::cerr << "Only 1 (info) or 2 (compare) parameters supported" << std::endl;
        std::cerr << error << std::endl;
        return -1;
    }
    
    return 0;
}
