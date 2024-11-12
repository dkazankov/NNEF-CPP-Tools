/*
 * Copyright (c) 2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnef.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <ctime>
#include <memory>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif


const std::set<std::string> lowered =
{
    "separable_conv",
    "separable_deconv",
    "rms_pool",
    "local_response_normalization",
    "local_mean_normalization",
    "local_variance_normalization",
    "local_contrast_normalization",
    "l1_normalization",
    "l2_normalization",
    "batch_normalization",
    "area_downsample",
    "nearest_downsample",
    "nearest_upsample",
    "linear_quantize",
    "logarithmic_quantize",
    "leaky_relu",
    "prelu",
    "clamp",
};

std::string read_file( const char* fn )
{
    std::ifstream is(fn);
    if ( !is )
    {
        throw std::runtime_error("file not found: " + std::string(fn));
    }
    
    return std::string((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
}

bool read_inputs_from_cin( nnef::Graph& graph, std::string& error )
{
    for ( auto& input : graph.inputs )
    {
        auto& tensor = graph.tensors.at(input);
        if ( !nnef::read_tensor(std::cin, tensor, error) )
        {
            return false;
        }
    }
    return true;
}

bool read_inputs_from_file( nnef::Graph& graph, const std::vector<std::string>& inputs, std::string& error )
{
    size_t idx = 0;
    for ( auto& input : graph.inputs )
    {
        auto& tensor = graph.tensors.at(input);
        if ( !nnef::read_tensor(inputs[idx++], tensor, error) )
        {
            return false;
        }
    }
    return true;
}

bool write_output_to_cout( const nnef::Graph& graph, std::string& error )
{
    for ( auto& output : graph.outputs )
    {
        auto& tensor = graph.tensors.at(output);
        if ( !nnef::write_tensor(std::cout, tensor, error) )
        {
            return false;
        }
    }
    return true;
}

bool write_output_to_file( const nnef::Graph& graph, const std::vector<std::string>& outputs, std::string& error )
{
    size_t idx = 0;
    for ( auto& output : graph.outputs )
    {
        auto& tensor = graph.tensors.at(output);
        if ( !nnef::write_tensor(outputs[idx++], tensor, error) )
        {
            return false;
        }
    }
    return true;
}

std::ostream& operator<<( std::ostream& os, const std::vector<int>& v )
{
    os << '[';
    for ( size_t i = 0; i < v.size(); ++i )
    {
        if ( i )
        {
            os << ',';
        }
        os << v[i];
    }
    os << ']';
    return os;
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... );
    if( size_s <= 0 )
    {
        throw std::runtime_error( "Error during formatting" );
    }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size+1 ] );
    std::snprintf( buf.get(), size+1, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size );
}


void write_tensors(const nnef::Graph& graph, bool(cond)(const nnef::Operation& op), const std::string& path, std::string& error)
{
    std::cerr << "Trace:{" << std::endl;
    int iop = 0;
    for ( const auto& operation : graph.operations )
    {
        iop++;
        if ( !cond(operation) )
        {
            continue;
        }
        std::cerr << "operation \"" << operation.name << "\", output (";
        int iout = 0;
        for ( const auto& output : operation.outputs )
        {
            iout++;
            if (iout > 1)
            {
                std::cerr << ", ";
            }
            std::cerr << output.first << " => " << output.second;
            std::string id;
            if (output.second.kind() == nnef::Value::Kind::String)
            {
                id = output.second.string();
            }
            else if (output.second.kind() == nnef::Value::Kind::Identifier)
            {
                id = output.second.identifier();
            }
            else
            {
                std::cerr << ": " << output.second.kind();
                continue;
            }
            std::string name = string_format("trace%03u-%s.dat", iop, id.c_str());
            const auto& tensor = graph.tensors.at(id);
            if ( !nnef::write_tensor(path + "/" + name, tensor, error) )
            {
                std::cerr << error << std::endl;
            }
        }
        std::cerr << ")" << std::endl;
    }
    std::cerr << "}" << std::endl;
}

int main( int argc, const char * argv[] )
{
    if ( argc < 2 )
    {
        std::cerr << "Input file name must be provided" << std::endl;
        return -1;
    }
    
    const std::string path = argv[1];
    std::string stdlib;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    bool trace = false;
    std::string trace_path("");
    
    for ( size_t i = 2; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if ( arg == "--stdlib" )
        {
            if ( ++i == argc )
            {
                std::cerr << "Stdlib file name must be provided after --stdlib; ignoring option" << std::endl;
            }
            try
            {
                stdlib = read_file(argv[i]);
            }
            catch ( std::runtime_error e )
            {
                std::cerr << e.what() << std::endl;
            }
        }
        else if ( arg == "--input" )
        {
            while ( i + 1 < argc && *argv[i+1] != '-' )
            {
                inputs.push_back(argv[++i]);
            }
            if ( inputs.size() == 0 )
            {
                std::cerr << "Input file name(s) must be provided after --input; ignoring option" << std::endl;
            }
        }
        else if ( arg == "--output" )
        {
            while ( i + 1 < argc && *argv[i+1] != '-' )
            {
                outputs.push_back(argv[++i]);
            }
            if ( outputs.size() == 0 )
            {
                std::cerr << "Output file name(s) must be provided after --output; ignoring option" << std::endl;
            }
        }
        else if ( arg == "--trace" )
        {
            trace = true;
            while ( i + 1 < argc && *argv[i+1] != '-' )
            {
                trace_path = argv[++i];
            }
        }
        else
        {
            std::cerr << "Unrecognized option: " << argv[i] << "; ignoring" << std::endl;
        }
    }
    
    nnef::Graph graph;
    std::string error;

    std::time_t start_time = std::time(nullptr);
    
    std::cerr << "Loading graph..." << std::endl;
    if ( !nnef::load_graph(path, graph, error, stdlib, lowered) )
    {
        std::cerr << error << std::endl;
        return -1;
    }

    std::map<std::string, std::vector<int>> input_shapes;
    if ( !inputs.empty() || !_isatty(_fileno(stdin)) )
    {
	    std::cerr << "Reading inputs..." << std::endl;
        bool read = !inputs.empty() ? read_inputs_from_file(graph, inputs, error) : read_inputs_from_cin(graph, error);
        if ( !read )
        {
            std::cerr << error << std::endl;
            return -1;
        }
        for ( auto& input : graph.inputs )
        {
            input_shapes.emplace(input, graph.tensors.at(input).shape);
        }
    }
    
    std::cerr << "Infering shapes..." << std::endl;
    if ( !nnef::infer_shapes(graph, error, input_shapes) )
    {
        std::cerr << error << std::endl;
        return -1;
    }
    
    std::cerr << "Allocating buffers..." << std::endl;
    if ( !nnef::allocate_buffers(graph, error) )
    {
        std::cerr << error << std::endl;
        return -1;
    }

    std::time_t end_time = std::time(nullptr);
    std::cerr << "Complete in " << (end_time - start_time) << " s" << std::endl;

    if ( trace )
    {
        write_tensors(graph, [](const nnef::Operation& op) {return op.name == "variable";}, trace_path, error);
    }
    
    start_time = std::time(nullptr);

    std::cerr << "Executing model: " << path << " ";
    if ( !nnef::execute(graph, error) )
    {
        std::cerr << error << std::endl;
        return -1;
    }
    
    end_time = std::time(nullptr);
    std::cerr << (end_time - start_time) << " s" << std::endl;
    
    if ( trace )
    {
        write_tensors(graph, [](const nnef::Operation& op) {return op.name != "external" && op.name != "variable";}, trace_path, error);
    }

    if ( !outputs.empty() || !_isatty(_fileno(stdout)) )
    {
        bool write = !outputs.empty() ? write_output_to_file(graph, outputs, error) : write_output_to_cout(graph, error);
        if ( !write )
        {
            std::cerr << error << std::endl;
            return -1;
        }
    }

    return 0;
}
