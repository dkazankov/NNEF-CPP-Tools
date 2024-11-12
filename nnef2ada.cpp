#include "nnef.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
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

std::string tensor_type( const std::string& dtype )
{
    if ( dtype == "scalar" )
    {
        return "Real";
    }
    else if ( dtype == "integer" )
    {
        return "Integer";
    }
    else if ( dtype == "logical" )
    {
        return "Boolean";
    }
    return dtype;
}

std::string tensor_rank( size_t rank )
{
    switch (rank)
    {
        case 1:
            return "Vector";
        case 2:
            return "Matrix";
        case 3:
            return "Tensor_3D";
        case 4:
            return "Tensor_4D";
    };
    return "Tensor";
}

std::string tensor_extents( const std::vector<int>& shape )
{
    std::ostringstream ext;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i) ext << ", ";
        ext << "1.." << shape[i];
    }
    return ext.str();
}

static std::vector<std::string> op_list;

std::string tensor_id( const std::string& id )
{
    auto pos = std::find(op_list.cbegin(), op_list.cend(), id);
    return (pos == op_list.end()? id: id + "_0");
}

std::string tensor_typename( const nnef::Tensor& tensor )
{
    return tensor_type(tensor.dtype) + "_" + tensor_rank(tensor.shape.size());
}

std::string tensor_declaration( const nnef::Tensor& tensor )
{
    return tensor_id(tensor.name) + ": " + tensor_typename(tensor) + " (" + tensor_extents(tensor.shape) + ");";
}

std::ostringstream& operator <<( std::ostringstream& os, const nnef::Value& value )
{
    switch (value.kind())
    {
        case nnef::Value::Kind::None:
            os << "None";
            break;
        case nnef::Value::Kind::String:
            os << value.string();
            break;
        case nnef::Value::Kind::Identifier:
            os << value.identifier();
            break;
        case nnef::Value::Kind::Logical:
            os << (value.logical()? "true": "false");
            break;
        case nnef::Value::Kind::Integer:
            os << value.integer();
            break;
        case nnef::Value::Kind::Scalar:
            os << value.scalar();
            if ( (nnef::Value::integer_t) value.scalar() == value.scalar() )
                os << ".0";
            break;
        case nnef::Value::Kind::Array:
        case nnef::Value::Kind::Tuple:
            os << "(";
            for ( size_t i = 0; i < value.size(); ++i )
            {
                if (i) os << ", ";
                os << value[i];
            }
            os << ")";
            break;
    }
    return os;
}

std::string param_description( const nnef::Graph& graph, const nnef::Operation& op, const std::string& attr, const nnef::Value& value )
{
    if (attr == "border")
    {
        return attr + " => Border_Mode_" + value.string();
    }
    std::ostringstream os;
    switch (value.kind())
    {
        case nnef::Value::Kind::None:
            os << "None";
            break;
        case nnef::Value::Kind::String:
            os << value.string();
            break;
        case nnef::Value::Kind::Identifier:
            os << value.identifier();
            break;
        case nnef::Value::Kind::Logical:
            os << (value.logical()? "true": "false");
            break;
        case nnef::Value::Kind::Integer:
            if (attr == "axis" || attr == "axis_start")
                os << value.integer() + 1;
            else
                os << value.integer();
            break;
        case nnef::Value::Kind::Scalar:
            os << value.scalar();
            if ( (nnef::Value::integer_t) value.scalar() == value.scalar() ) os << ".0";
            break;
        case nnef::Value::Kind::Array:
        case nnef::Value::Kind::Tuple:
            if (value.size() == 0)
            {
                const auto& input = op.inputs[0];
                const auto& id = input.second.identifier();
                const auto& tensor = graph.tensors.at(id);
                size_t rank = tensor.shape.size();
                if ( op.name == "conv" ) rank -= 2;
                if ( attr == "padding" ) os << "Padding_Auto";
                else if ( attr == "stride" ) os << "Default_Stride";
                else if ( attr == "dilation" ) os << "Default_Dilation";
                else
                {
                    os << "(";
                    for ( size_t i = 0; i < rank; ++i )
                    {
                        if (i) os << ", ";
                        // if ( attr == "padding" ) os << "(0, 0)";
                        // if ( attr == "stride" ) os << "1";
                        // else if ( attr == "dilation" ) os << "1";
                        // else os << "0";
                        os << "0";
                    }
                    os << ")";
                }
            }
            else if (value.size() == 1)
            {
                if (attr == "axes")
                    os << "(1 => " << value[0].integer() + 1 << ")";
                else
                    os << "(1 => " << value[0] << ")";
            }
            else
            {
                os << "(";
                for ( size_t i = 0; i < value.size(); ++i )
                {
                    if (i) os << ", ";
                    if (attr == "axes")
                        os << value[i].integer()+1;
                    else
                        os << value[i];
                }
                os << ")";
            }
            break;
    }
    return attr + " => " + os.str();
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
    
    for ( size_t i = 2; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if ( arg == "--stdlib" )
        {
            if ( i+1 >= argc && *argv[i+1] != '-' )
            {
                std::cerr << "Stdlib file name must be provided after --stdlib; ignoring option" << std::endl;
            }
            else
            try
            {
                stdlib = read_file(argv[i]);
            }
            catch ( std::runtime_error e )
            {
                std::cerr << e.what() << std::endl;
            }
        }
        else
        {
            std::cerr << "Unrecognized option: " << argv[i] << "; ignoring" << std::endl;
        }
    }

    nnef::Graph graph;
    std::string error;
    
    if ( !nnef::load_graph(path, graph, error, stdlib, lowered) )
    {
        std::cerr << error << std::endl;
        return -2;
    }

    if ( !nnef::infer_shapes(graph, error) )
    {
        std::cerr << error << std::endl;
        return -3;
    }

    std::ostringstream declarations;
    std::ostringstream load;
    std::ostringstream fwd_decl;
    std::ostringstream fwd_text;

    std::set<std::string> external_types;
    std::set<std::string> variable_types;
    std::set<std::string> output_types;
    std::set<std::string> all_types;

    size_t i = 0;

    for ( const auto& operation : graph.operations )
    {
        auto pos = std::find(op_list.cbegin(), op_list.cend(), operation.name);
        if (pos == op_list.end())
        {
            op_list.push_back(operation.name);
        }
    }
    op_list.push_back("local_response_normalization");

    for ( const auto& operation : graph.operations )
    {
	    if (operation.name == "external")
	    {
            for ( const auto& output : operation.outputs )
            {
                const auto& id = output.second.identifier();
                const auto& tensor = graph.tensors.at(id);
                const auto& shape = operation.attribs.get("shape");
                load << "    " + operation.name + " (\"" + tensor.name + "\", " + tensor_id(tensor.name) + ");" << std::endl;
                external_types.insert(tensor_typename(tensor));
                all_types.insert(tensor_typename(tensor));
            }
	    }
	    else if (operation.name == "variable")
	    {
            for ( const auto& output : operation.outputs )
            {
                const auto& id = output.second.identifier();
                const auto& tensor = graph.tensors.at(id);
                const auto& label = operation.attribs.get("label");
                const auto& shape = operation.attribs.get("shape");
                declarations << "    " << tensor_declaration(tensor) << std::endl;
                load << "    " + operation.name + " (\"" + label.string() + "\", " + tensor_id(tensor.name) + ");" << std::endl;
                variable_types.insert(tensor_typename(tensor));
                all_types.insert(tensor_typename(tensor));
            }
	    }
        else
        {
            size_t i = 0;
            fwd_text << "        " << operation.name << " (";

            for ( ; i<operation.inputs.size(); ++i )
            {
                if (i) fwd_text << ", ";
                const auto& input = operation.inputs[i];
                if ((operation.name == "add" || operation.name == "mul") &&
                    i == 0 && input.second.kind() != nnef::Value::Kind::Identifier &&
                    operation.inputs.size() >= 2 && operation.inputs[i+1].second.kind() == nnef::Value::Kind::Identifier)
                {
                    const auto& input1 = operation.inputs[i+1];
                    fwd_text << input.first << " => " << tensor_id(input1.second.identifier());
                    fwd_text << ", ";
                    fwd_text << input1.first << " => " << input.second;
                    ++i;
                }
                else if (input.second.kind() == nnef::Value::Kind::Identifier)
                {
                    auto& id = input.second.identifier();
                    fwd_text << input.first << " => " << tensor_id(id);
                }
                else
                {
                    fwd_text << input.first << " => " << input.second;
                }
            }

            if ( operation.name != "reshape" )
            {
                for ( const auto& attr : operation.attribs )
                {
                    if (i++) fwd_text << ", ";
                    fwd_text << param_description(graph, operation, attr.first, attr.second);
                }
            }

            for ( const auto& output : operation.outputs )
            {
                const auto& id = output.second.identifier();
                const auto& tensor = graph.tensors.at(id);
                if (i++) fwd_text << ", ";
                auto pos = std::find(graph.outputs.cbegin(), graph.outputs.cend(), id);
                if (pos == graph.outputs.end())
                {
                    fwd_decl << "        " << tensor_declaration(tensor) << std::endl;
                }
                else
                {
                    output_types.insert(tensor_typename(tensor));
                }
                fwd_text << output.first << " => " << tensor_id(output.second.identifier());
                all_types.insert(tensor_typename(tensor));
            }
            fwd_text << ");" << std::endl;
        }
    }

    std::cout << "-- " << graph.name << ".ads" << std::endl;
    std::cout << "with Generic_Real_Arrays;" << std::endl;
    std::cout << "with Generic_Real_Arrays.Operators;" << std::endl;
    std::cout << "package " << graph.name << " is" << std::endl;
    std::cout << "    pragma Preelaborate;" << std::endl;
    std::cout << "    package Real_Arrays is new Generic_Real_Arrays(Real => Float);" << std::endl;
    std::cout << "    package Operators is new Real_Arrays.Operators;" << std::endl;
    std::cout << "    use Real_Arrays;" << std::endl;
    std::cout << "    use Operators;" << std::endl;
    for ( const auto& input : graph.inputs )
    {
        const auto& tensor = graph.tensors.at(input);
        std::cout << "    " << tensor_declaration(tensor) << std::endl;
    }
    for ( const auto& output : graph.outputs )
    {
        const auto& tensor = graph.tensors.at(output);
        std::cout << "    " << tensor_declaration(tensor) << std::endl;
    }
    std::cout << declarations.str();
    std::cout << fwd_decl.str();
    std::cout << "    procedure Forward;" << std::endl;
    std::cout << "end " << graph.name << ";" << std::endl;

    std::cout << "-- " << graph.name << ".adb" << std::endl;
    std::cout << "package body " << graph.name << " is" << std::endl;
    std::cout << "    procedure Forward is" << std::endl;
    std::cout << "    begin" << std::endl;
    std::cout << fwd_text.str();
    std::cout << "    end Forward;" << std::endl;
    std::cout << "end " << graph.name << ";" << std::endl;

    std::cout << "-- " << graph.name << "_run.adb" << std::endl;
    std::cout << "with " << graph.name << "; use " << graph.name << ";" << std::endl;
    std::cout << "use " << graph.name << ".Real_Arrays;" << std::endl;
    std::cout << "procedure " + graph.name + "_Run is" << std::endl;
    for (auto type_name: external_types)
    {
        std::cout << "    procedure External (Var_Name: String; Tensor: out " + type_name + ") is" << std::endl;
        std::cout << "    begin" << std::endl;
        std::cout << "        null;" << std::endl;
        std::cout << "    end External;" << std::endl;
    }
    for (auto type_name: variable_types)
    {
        std::cout << "    procedure Variable (Var_Name: String; Tensor: out " + type_name + ") is" << std::endl;
        std::cout << "    begin" << std::endl;
        std::cout << "        null;" << std::endl;
        std::cout << "    end Variable;" << std::endl;
    }
    for (auto type_name: output_types)
    {
        std::cout << "    procedure Output (Tensor: " + type_name + "; Var_Name: String) is" << std::endl;
        std::cout << "    begin" << std::endl;
        std::cout << "        null;" << std::endl;
        std::cout << "    end Output;" << std::endl;
    }

    std::cout << "begin" << std::endl;
    std::cout << load.str();
    std::cout << "    Forward;" << std::endl;
    for ( const auto& output : graph.outputs )
    {
        const auto& tensor = graph.tensors.at(output);
        std::cout << "    Output (" + tensor_id(tensor.name) + ", \"" + tensor.name + "\");" << std::endl;
    }
    std::cout << "end " + graph.name + "_Run;" << std::endl;
    
    return 0;
}
