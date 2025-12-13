# TileTarget: Compilation target for cuTile
#
# Holds the typed IR and compilation state for a kernel.

include("interpreter.jl")

"""
    TileTarget

Holds information about a function being compiled to Tile IR.
"""
struct TileTarget
    mi::MethodInstance
    ci::CodeInfo
    rettype::Type
    argtypes::Vector{Any}
end

function TileTarget(@nospecialize(f), @nospecialize(argtypes::Type{<:Tuple}))
    ci, rettype = get_typed_ir(f, argtypes)
    mi = get_method_instance(f, argtypes)
    # Extract argument types from the Tuple type
    arg_types = argtypes === Tuple{} ? Any[] : collect(argtypes.parameters)
    TileTarget(mi, ci, rettype, arg_types)
end

# Accessors
code(target::TileTarget) = target.ci.code
ssatypes(target::TileTarget) = target.ci.ssavaluetypes
slottypes(target::TileTarget) = target.ci.slottypes
nargs(target::TileTarget) = length(target.argtypes)

using Core: SlotNumber

"""
    Translation

Maps Julia SSA values to Tile IR values during compilation.
"""
mutable struct Translation
    # Maps Julia SSA values/arguments to Tile IR values
    args::Dict{Int, Value}          # Argument index -> Value
    results::Dict{Int, Value}       # SSA index -> Value
    slots::Dict{Int, Value}         # Slot number -> Value (for unoptimized IR)

    # Maps Tile IR values to their types
    value_types::Dict{Int, TypeId}  # Value.id -> TypeId

    # Maps Tile IR values to their tile shapes (for tiles only)
    tile_shapes::Dict{Int, Vector{Int}}  # Value.id -> shape

    # Maps Tile IR values to their grid axis (for bid results only)
    value_grid_axis::Dict{Int, Int}  # Value.id -> axis (0=x, 1=y, 2=z)

    # Maps Tile IR values to their Julia types
    value_julia_types::Dict{Int, Any}  # Value.id -> Julia type

    # Unified argument field map: (arg_idx, field_or_nothing) -> flat Values
    # For regular args: (i, nothing) -> [value]
    # For struct fields: (i, :fieldname) -> [values...]
    arg_flat_values::Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}

    # Original types for destructured arguments
    arg_types::Dict{Int, Type}  # Argument index -> Julia type

    # Tables for the bytecode
    type_table::TypeTable
    string_table::StringTable
    constant_table::ConstantTable

    # Code builder
    code_builder::CodeBuilder

    # Type cache: Julia type -> TypeId
    type_cache::Dict{Type, TypeId}

    # Token for memory ordering (created at kernel start)
    token::Union{Value, Nothing}
end

function Translation(writer::BytecodeWriter)
    Translation(
        Dict{Int, Value}(),
        Dict{Int, Value}(),
        Dict{Int, Value}(),
        Dict{Int, TypeId}(),
        Dict{Int, Vector{Int}}(),
        Dict{Int, Int}(),
        Dict{Int, Any}(),
        Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}(),
        Dict{Int, Type}(),
        writer.type_table,
        writer.string_table,
        writer.constant_table,
        CodeBuilder(writer.string_table, writer.constant_table, writer.type_table),
        Dict{Type, TypeId}(),
        nothing  # token initialized later
    )
end

# Get the Tile IR type of a Value
function get_value_type(tr::Translation, val::Value)
    get(tr.value_types, val.id, nothing)
end

# Set the Tile IR type of a Value
function set_value_type!(tr::Translation, val::Value, type_id::TypeId)
    tr.value_types[val.id] = type_id
end

# Get the tile shape for a Value
function get_tile_shape(tr::Translation, val::Value)
    get(tr.tile_shapes, val.id, nothing)
end

# Set the tile shape for a Value
function set_tile_shape!(tr::Translation, val::Value, shape::Vector{Int})
    tr.tile_shapes[val.id] = shape
end

# Get the grid axis for a Value (bid results)
function get_grid_axis(tr::Translation, val::Value)
    get(tr.value_grid_axis, val.id, nothing)
end

# Set the grid axis for a Value
function set_grid_axis!(tr::Translation, val::Value, axis::Int)
    tr.value_grid_axis[val.id] = axis
end

# Get the Julia type of a Value
function get_julia_type(tr::Translation, val::Value)
    get(tr.value_julia_types, val.id, nothing)
end

# Set the Julia type of a Value
function set_julia_type!(tr::Translation, val::Value, @nospecialize(jltype))
    tr.value_julia_types[val.id] = jltype
end

# Lookup values
function Base.getindex(tr::Translation, arg::Argument)
    get(tr.args, arg.n, nothing)
end

function Base.getindex(tr::Translation, ssa::SSAValue)
    get(tr.results, ssa.id, nothing)
end

function Base.getindex(tr::Translation, slot::SlotNumber)
    get(tr.slots, slot.id, nothing)
end

function Base.setindex!(tr::Translation, val::Value, arg::Argument)
    tr.args[arg.n] = val
end

function Base.setindex!(tr::Translation, val::Value, ssa::SSAValue)
    tr.results[ssa.id] = val
end

function Base.setindex!(tr::Translation, val::Value, slot::SlotNumber)
    tr.slots[slot.id] = val
end

# Resolve any value (Argument, SSAValue, SlotNumber) to its Tile IR Value
function resolve_value(tr::Translation, @nospecialize(val))
    if val isa Argument
        return tr[val]
    elseif val isa SSAValue
        return tr[val]
    elseif val isa SlotNumber
        return tr[val]
    else
        # Literal value or QuoteNode - not a tracked SSA value
        return nothing
    end
end

"""
    tile_type_for_julia!(tr::Translation, T) -> TypeId

Get or create a Tile IR type for a Julia type.
Handles Core.Const and other type wrappers.
"""
function tile_type_for_julia!(tr::Translation, @nospecialize(T))
    # Unwrap Core.Const and other type wrappers
    actual_type = unwrap_type(T)

    get!(tr.type_cache, actual_type) do
        _tile_type_for_julia!(tr, actual_type)
    end
end

"""
    unwrap_type(T) -> Type

Unwrap type wrappers like Core.Const to get the actual type.
"""
function unwrap_type(@nospecialize(T))
    if T isa Core.Const
        return typeof(T.val)
    elseif T isa Core.PartialStruct
        return T.typ
    elseif T isa Type
        return T
    else
        # Might be another wrapper or the type itself
        return T
    end
end

"""
    require_concrete_type(T, context::String)

Ensure a type is fully concrete (not a UnionAll). Throws an error with a helpful
message if the type has unbound type parameters.

This is used to enforce that all types in the kernel are fully specified at compile time.
"""
function require_concrete_type(@nospecialize(T), context::String)
    T_unwrapped = unwrap_type(T)
    if T_unwrapped isa UnionAll
        error("Type must be fully concrete in $context, got partial type: $T. " *
              "Ensure all type parameters are specified (e.g., use Tile{Float32, (16,)} " *
              "instead of Tile{Float32}).")
    end
    return T_unwrapped
end

function _tile_type_for_julia!(tr::Translation, @nospecialize(T::Type))
    tt = tr.type_table

    # Scalar types -> 0-D tile of the corresponding dtype
    if T === Bool
        return tile_type!(tt, I1(tt), Int[])
    elseif T === Int32 || T === UInt32
        return tile_type!(tt, I32(tt), Int[])
    elseif T === Int64 || T === UInt64
        return tile_type!(tt, I64(tt), Int[])
    elseif T === Float16
        return tile_type!(tt, F16(tt), Int[])
    elseif T === Float32
        return tile_type!(tt, F32(tt), Int[])
    elseif T === Float64
        return tile_type!(tt, F64(tt), Int[])
    elseif T === Nothing
        return Token(tt)  # Use token type for Nothing/Unit
    end

    # Pointers -> 0-D tile of pointer type
    if T <: Ptr
        elem_dtype = julia_to_tile_dtype!(tt, eltype(T))
        ptr_type = pointer_type!(tt, elem_dtype)
        return tile_type!(tt, ptr_type, Int[])
    end

    # Tile{T, Shape} -> tile type with appropriate dtype and shape
    # Both T and Shape must be fully specified (no UnionAll)
    if T isa DataType && T.name.name === :Tile
        if length(T.parameters) < 2
            error("Tile type must have both element type and shape specified, got: $T. " *
                  "Use Tile{Float32, (16,)} instead of Tile{Float32}.")
        end
        elem_type = T.parameters[1]
        shape_param = T.parameters[2]
        if !(shape_param isa Tuple)
            error("Tile shape must be a compile-time constant tuple, got: $shape_param in $T")
        end
        elem_dtype = julia_to_tile_dtype!(tt, elem_type)
        shape = collect(Int, shape_param)
        return tile_type!(tt, elem_dtype, shape)
    end

    error("Unsupported Julia type for Tile IR: $T")
end

"""
    is_float_type(T) -> Bool

Check if a Julia type is a floating-point type.
Handles Core.Const and other type wrappers.
"""
function is_float_type(@nospecialize(T))
    actual_type = unwrap_type(T)
    return actual_type <: AbstractFloat
end

"""
    is_int_type(T) -> Bool

Check if a Julia type is an integer type.
Handles Core.Const and other type wrappers.
"""
function is_int_type(@nospecialize(T))
    actual_type = unwrap_type(T)
    return actual_type <: Integer
end

#=============================================================================
 Struct Destructuring Helpers
=============================================================================#

"""
    should_destructure(T) -> Bool

Check if a type should be destructured into flat parameters.
Returns true for struct types that have runtime fields (like TileArray).
Requires T to be a concrete type (not a UnionAll).
"""
function should_destructure(@nospecialize(T))
    T = unwrap_type(T)

    # Must be a concrete struct type (not UnionAll)
    T isa DataType || return false
    # Must have fields (not a primitive or abstract type)
    isstructtype(T) || return false
    # Must not be a ghost type (zero-size)
    is_ghost_type(T) && return false
    # Must not be a primitive type
    isprimitivetype(T) && return false
    # Check if it's a TileArray
    T.name.name === :TileArray && return true
    # For now, only destructure TileArray
    return false
end

"""
    get_base_type(T) -> DataType

Get the base DataType from a type. Requires T to be concrete (not UnionAll).
"""
function get_base_type(@nospecialize(T))
    T isa DataType && return T
    error("Expected a concrete DataType, got: $T ($(typeof(T)))")
end

"""
    flat_field_count(T) -> Int

Count the number of flat parameters a type expands to.
Scalars expand to 1, NTuple{N,T} expands to N.
"""
flat_field_count(::Type{<:NTuple{N, T}}) where {N, T} = N
flat_field_count(::Type) = 1

"""
    total_flat_field_count(T) -> Int

Count the total number of flat parameters for all fields of a struct type.
"""
function total_flat_field_count(@nospecialize(T))
    count = 0
    for i in 1:fieldcount(T)
        ftype = fieldtype(T, i)
        count += flat_field_count(ftype)
    end
    return count
end

"""
    get_arg_flat_values(tr, arg_idx, field=nothing) -> Union{Vector{Value}, Nothing}

Get the flat values for an argument or its field.
- For regular args: `get_arg_flat_values(tr, i)` or `get_arg_flat_values(tr, i, nothing)`
- For struct fields: `get_arg_flat_values(tr, i, :fieldname)`
Returns nothing if not found.
"""
function get_arg_flat_values(tr::Translation, arg_idx::Int, field::Union{Nothing, Symbol}=nothing)
    get(tr.arg_flat_values, (arg_idx, field), nothing)
end

"""
    is_destructured_arg(tr, arg_idx) -> Bool

Check if an argument was destructured (has field entries in arg_flat_values).
"""
function is_destructured_arg(tr::Translation, arg_idx::Int)
    haskey(tr.arg_types, arg_idx)
end

"""
    get_arg_type(tr, arg_idx) -> Union{Type, Nothing}

Get the original Julia type for a destructured argument.
"""
function get_arg_type(tr::Translation, arg_idx::Int)
    get(tr.arg_types, arg_idx, nothing)
end
