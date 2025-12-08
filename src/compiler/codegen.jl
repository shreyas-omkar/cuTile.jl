# Codegen: Julia IR -> Tile IR bytecode
#
# Pattern matches on Julia SSA IR nodes and emits corresponding Tile IR operations.

include("target.jl")

using Core: SlotNumber

# Forward declarations for cuTile intrinsics
# These will be resolved when the module is fully loaded
const CUTILE_INTRINSICS = Dict{Symbol, Function}()

function register_intrinsic!(name::Symbol, func::Function)
    CUTILE_INTRINSICS[name] = func
end

"""
    emit_kernel!(writer, target; name, is_entry=true) -> Vector{UInt8}

Compile a TileTarget to Tile IR bytecode.
"""
function emit_kernel!(writer::BytecodeWriter, func_buf::Vector{UInt8},
                      target::TileTarget;
                      name::String = string(target.mi.def.name),
                      is_entry::Bool = true)
    tr = Translation(writer)
    tt = tr.type_table

    # Determine parameter types for the kernel
    param_types = TypeId[]
    for (i, argtype) in enumerate(target.argtypes)
        push!(param_types, tile_type_for_julia!(tr, argtype))
    end

    # Determine return types
    result_types = TypeId[]
    if target.rettype !== Nothing && target.rettype !== Union{}
        push!(result_types, tile_type_for_julia!(tr, target.rettype))
    end

    # Create the function
    cb = add_function!(writer, func_buf, name, param_types, result_types; is_entry)
    tr.code_builder = cb

    # Set up argument values
    # In Julia's CodeInfo:
    # - Slot 1 (_1) is the function itself
    # - Slots 2..n+1 (_2.._n+1) are the n function arguments
    arg_values = make_block_args!(cb, length(param_types))
    for (i, val) in enumerate(arg_values)
        # Slot i+1 corresponds to argument i
        tr[SlotNumber(i + 1)] = val
        # Also keep the Argument mapping for backwards compatibility
        tr[Argument(i + 1)] = val
    end

    # Emit each statement
    code_stmts = code(target)
    types = ssatypes(target)

    for (i, stmt) in enumerate(code_stmts)
        result_type = types[i]
        emit_statement!(tr, target, stmt, i, result_type)
    end

    finalize_function!(func_buf, cb, writer.debug_info)
end

"""
    emit_statement!(tr, target, stmt, idx, result_type)

Emit bytecode for a single SSA statement.
"""
function emit_statement!(tr::Translation, target::TileTarget,
                         @nospecialize(stmt), idx::Int, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # Handle different statement types
    if stmt isa ReturnNode
        emit_return!(tr, stmt)
    elseif stmt isa GotoNode
        # Simple goto - Tile IR uses structured control flow, skip for now
        # (will need restructuring pass for real control flow)
    elseif stmt isa GotoIfNot
        # Conditional goto - needs restructuring
        error("Control flow not yet supported: GotoIfNot")
    elseif stmt isa Expr
        val = emit_expr!(tr, target, stmt, idx, result_type)
        if val !== nothing
            tr[SSAValue(idx)] = val
        end
    elseif stmt isa GlobalRef
        # Global reference - might be a function or constant
        # In unoptimized IR, these are loaded as values but we don't emit anything
        # They're resolved when used in calls
    elseif stmt isa QuoteNode
        # Quoted literal - emit as constant
        val = emit_constant!(tr, stmt.value, result_type)
        if val !== nothing
            tr[SSAValue(idx)] = val
        end
    elseif stmt isa SlotNumber
        # Slot reference - copy the slot's value to this SSA position
        # This allows later SSA references to find the value
        slot_val = tr[stmt]
        if slot_val !== nothing
            tr[SSAValue(idx)] = slot_val
        end
    elseif stmt === nothing
        # No-op
    else
        @warn "Unhandled statement type" typeof(stmt) stmt
    end
end

"""
    emit_return!(tr, node::ReturnNode)

Emit a return operation.
"""
function emit_return!(tr::Translation, node::ReturnNode)
    cb = tr.code_builder

    if node.val === nothing || (node.val isa GlobalRef && node.val.name === :nothing)
        encode_ReturnOp!(cb, Value[])
    else
        val = resolve_value(tr, node.val)
        if val !== nothing
            encode_ReturnOp!(cb, [val])
        else
            # Try to emit as constant
            encode_ReturnOp!(cb, Value[])
        end
    end
end

"""
    emit_expr!(tr, target, expr::Expr, idx, result_type) -> Union{Value, Nothing}

Emit bytecode for an expression.
"""
function emit_expr!(tr::Translation, target::TileTarget,
                    expr::Expr, idx::Int, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if expr.head === :call
        return emit_call!(tr, target, expr, result_type)
    elseif expr.head === :invoke
        return emit_invoke!(tr, target, expr, result_type)
    elseif expr.head === :(=)
        # Assignment expression: (_slot = value)
        # In unoptimized IR, this stores a value to a slot
        lhs = expr.args[1]
        rhs = expr.args[2]

        # Emit the RHS
        val = emit_rhs!(tr, target, rhs, idx, result_type)

        # Store to the slot if LHS is a SlotNumber
        if lhs isa SlotNumber && val !== nothing
            tr[lhs] = val
        end

        return val
    elseif expr.head === :new
        # Struct construction - skip for now
        return nothing
    elseif expr.head === :foreigncall
        error("Foreign calls not supported in Tile IR")
    elseif expr.head === :boundscheck
        # Bounds checking - skip in GPU code
        return nothing
    else
        @warn "Unhandled expression head" expr.head expr
        return nothing
    end
end

"""
    emit_rhs!(tr, target, rhs, idx, result_type) -> Union{Value, Nothing}

Emit bytecode for the right-hand side of an assignment.
"""
function emit_rhs!(tr::Translation, target::TileTarget,
                   @nospecialize(rhs), idx::Int, @nospecialize(result_type))
    if rhs isa Expr
        return emit_expr!(tr, target, rhs, idx, result_type)
    elseif rhs isa SSAValue
        return tr[rhs]
    elseif rhs isa SlotNumber
        return tr[rhs]
    elseif rhs isa GlobalRef
        # Global reference being assigned - resolve but don't emit
        return nothing
    elseif rhs isa QuoteNode
        return emit_constant!(tr, rhs.value, result_type)
    else
        # Literal value - emit as constant
        return emit_constant!(tr, rhs, result_type)
    end
end

"""
    emit_call!(tr, target, expr::Expr, result_type) -> Union{Value, Nothing}

Emit bytecode for a function call.
"""
function emit_call!(tr::Translation, target::TileTarget,
                    expr::Expr, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    args = expr.args
    f = args[1]
    call_args = args[2:end]

    # Resolve the function being called
    # In unoptimized IR, the function may be an SSAValue that references
    # a GlobalRef statement earlier in the code
    func = resolve_function_in_ir(f, target)

    # Handle known functions
    return emit_known_call!(tr, func, call_args, result_type)
end

"""
    resolve_function_in_ir(f, target) -> Any

Resolve a function reference, handling SSAValues that reference GlobalRefs in the IR.
"""
function resolve_function_in_ir(@nospecialize(f), target::TileTarget)
    if f isa SSAValue
        # Look up what this SSA value refers to in the code
        stmt = code(target)[f.id]
        if stmt isa GlobalRef
            return getfield(stmt.mod, stmt.name)
        elseif stmt isa QuoteNode
            return stmt.value
        end
        # Might be a computed function - fall through
    end
    return resolve_function(f)
end

"""
    emit_invoke!(tr, target, expr::Expr, result_type) -> Union{Value, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(tr::Translation, target::TileTarget,
                      expr::Expr, @nospecialize(result_type))
    cb = tr.code_builder

    # invoke has: (MethodInstance, func, args...)
    mi = expr.args[1]
    f = expr.args[2]
    call_args = expr.args[3:end]

    # Resolve the function from the GlobalRef
    func = resolve_function(f)

    return emit_known_call!(tr, func, call_args, result_type)
end

"""
    resolve_function(f) -> Any

Resolve a function reference to its actual value.
"""
function resolve_function(@nospecialize(f))
    if f isa GlobalRef
        return getfield(f.mod, f.name)
    elseif f isa QuoteNode
        return f.value
    else
        return f
    end
end

"""
    emit_known_call!(tr, func, args, result_type) -> Union{Value, Nothing}

Emit bytecode for a known function call.
"""
function emit_known_call!(tr::Translation, @nospecialize(func),
                          args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # Check for cuTile intrinsics first
    if haskey(CUTILE_INTRINSICS, :bid) && func === CUTILE_INTRINSICS[:bid]
        return emit_bid!(tr, args, result_type)
    elseif haskey(CUTILE_INTRINSICS, :load) && func === CUTILE_INTRINSICS[:load]
        return emit_load!(tr, args, result_type)
    elseif haskey(CUTILE_INTRINSICS, :store) && func === CUTILE_INTRINSICS[:store]
        return emit_store!(tr, args, result_type)
    elseif haskey(CUTILE_INTRINSICS, :num_blocks) && func === CUTILE_INTRINSICS[:num_blocks]
        return emit_num_blocks!(tr, args, result_type)
    end

    # Core intrinsics that we can skip
    if func === Core.tuple
        # Tuple construction - this is a compile-time constant, don't emit anything
        # The tuple value is in the result_type as Core.Const((values...))
        return nothing
    elseif func === Base.getfield
        # Field access - might be getting tuple elements
        return nothing
    end

    # Handle result type for non-void functions
    res_type = result_type !== Nothing ? tile_type_for_julia!(tr, result_type) : nothing

    # Julia intrinsics for arithmetic
    if func === Base.add_float && res_type !== nothing
        return emit_add!(tr, args, result_type, res_type)
    elseif func === Base.sub_float && res_type !== nothing
        return emit_sub!(tr, args, result_type, res_type)
    elseif func === Base.mul_float && res_type !== nothing
        return emit_mul!(tr, args, result_type, res_type)
    elseif func === Base.add_int && res_type !== nothing
        return emit_add!(tr, args, result_type, res_type)
    elseif func === Base.sub_int && res_type !== nothing
        return emit_sub!(tr, args, result_type, res_type)
    elseif func === Base.mul_int && res_type !== nothing
        return emit_mul!(tr, args, result_type, res_type)
    end

    # High-level arithmetic (might be inlined to intrinsics)
    if func === Base.:(+) && res_type !== nothing
        return emit_add!(tr, args, result_type, res_type)
    elseif func === Base.:(-) && res_type !== nothing
        return emit_sub!(tr, args, result_type, res_type)
    elseif func === Base.:(*) && res_type !== nothing
        return emit_mul!(tr, args, result_type, res_type)
    end

    @warn "Unknown function call" func args
    return nothing
end

#=============================================================================
 cuTile Intrinsic Emitters
=============================================================================#

"""
    emit_bid!(tr, args, result_type) -> Value

Emit GetTileBlockIdOp for bid(axis).
"""
function emit_bid!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # bid(axis) - axis should be a constant 0, 1, or 2
    if length(args) != 1
        error("bid() requires exactly 1 argument (axis)")
    end

    axis_val = args[1]
    axis = extract_constant_int(axis_val)
    if axis === nothing
        error("bid() axis must be a compile-time constant")
    end
    if !(axis in (0, 1, 2))
        error("bid() axis must be 0, 1, or 2, got $axis")
    end

    # Result type is Int32 scalar tile
    res_type = tile_type!(tt, I32(tt), Int[])

    # GetTileBlockIdOp returns (x, y, z) - we select the one we want
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(cb, res_type, res_type, res_type)

    return (bid_x, bid_y, bid_z)[axis + 1]
end

"""
    emit_num_blocks!(tr, args, result_type) -> Value

Emit GetNumTileBlocksOp for num_blocks(axis).
"""
function emit_num_blocks!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) != 1
        error("num_blocks() requires exactly 1 argument (axis)")
    end

    axis_val = args[1]
    axis = extract_constant_int(axis_val)
    if axis === nothing
        error("num_blocks() axis must be a compile-time constant")
    end
    if !(axis in (0, 1, 2))
        error("num_blocks() axis must be 0, 1, or 2, got $axis")
    end

    res_type = tile_type!(tt, I32(tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(cb, res_type, res_type, res_type)

    return (nb_x, nb_y, nb_z)[axis + 1]
end

"""
    emit_load!(tr, args, result_type) -> Value

Emit load operation for ct.load(array; index, shape).
This creates: TensorView -> PartitionView -> LoadViewTkoOp.
"""
function emit_load!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # load(array, index, shape)
    if length(args) < 1
        error("load() requires at least an array argument")
    end

    array_val = resolve_value(tr, args[1])
    if array_val === nothing
        error("Cannot resolve array argument for load()")
    end

    # Parse index argument
    index_vals = Value[]
    if length(args) >= 2
        index_arg = args[2]
        v = resolve_value(tr, index_arg)
        if v !== nothing
            push!(index_vals, v)
        end
    end

    # Parse shape argument - extract constant tuple
    tile_shape = Int[16]  # Default
    if length(args) >= 3
        shape_arg = args[3]
        # Shape is typically a constant tuple like (16,)
        # Try to extract from the SSA type
        shape = extract_constant_tuple_from_arg(shape_arg, tr)
        if shape !== nothing
            tile_shape = collect(Int, shape)
        end
    end

    # Determine element type - default to Float32 for now
    # TODO: Extract from array type
    elem_type = Float32
    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create types
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # TensorView type (1D with dynamic shape/stride)
    ndim = length(tile_shape)
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = fill(DYNAMIC_SHAPE, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

    # For tensor view, we need size and stride values
    # In a proper implementation, these would come from kernel parameters
    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    scalar_i64 = tile_type!(tt, I64(tt), Int[])

    # Get grid size and use it as size (this gives us a dynamic value)
    # Multiply by tile_size to get total element count
    nb_x, _, _ = encode_GetNumTileBlocksOp!(cb, scalar_i32, scalar_i32, scalar_i32)

    # For now, use nb_x directly - in real code we'd convert to i64 and multiply
    # TensorView needs i64 values, but some Tile IR implementations accept i32
    # Use the i32 value for now - if this fails, we need type conversion ops
    size_val = nb_x
    stride_val = nb_x  # Use same value as placeholder

    # Create tensor view from pointer
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, [size_val], [stride_val])

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Use provided index or default to the first argument if it's an index
    if isempty(index_vals)
        # Default index of 0
        idx_type = tile_type!(tt, I32(tt), Int[])
        idx_bytes = reinterpret(UInt8, [Int32(0)])
        index_vals = [encode_ConstantOp!(cb, idx_type, collect(idx_bytes))]
    end

    # Load tile
    tile_val, _ = encode_LoadViewTkoOp!(cb, tile_type, token_type, partition, index_vals)

    # Track the type of this value for later operations
    set_value_type!(tr, tile_val, tile_type)

    return tile_val
end

"""
    extract_constant_tuple_from_arg(arg, tr) -> Union{Tuple, Nothing}

Extract a constant tuple value from an argument, handling SSA references.
"""
function extract_constant_tuple_from_arg(@nospecialize(arg), tr::Translation)
    if arg isa Tuple
        return arg
    elseif arg isa QuoteNode && arg.value isa Tuple
        return arg.value
    end
    # For SSA values, we'd need to look at the result type
    # For now, return nothing and let caller use default
    return nothing
end

"""
    emit_store!(tr, args, result_type)

Emit store operation for ct.store(array, index, tile).
Creates: TensorView -> PartitionView -> StoreViewTkoOp.
"""
function emit_store!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # store(array, index, tile)
    if length(args) < 3
        error("store() requires array, index, and tile arguments")
    end

    array_val = resolve_value(tr, args[1])
    if array_val === nothing
        error("Cannot resolve array argument for store()")
    end

    # Parse index argument
    index_vals = Value[]
    index_arg = args[2]
    v = resolve_value(tr, index_arg)
    if v !== nothing
        push!(index_vals, v)
    end

    # Parse tile argument
    tile_val = resolve_value(tr, args[3])
    if tile_val === nothing
        error("store() requires a tile argument")
    end

    # Get tile shape from the tile value's type if available
    tile_shape = [16]  # Default
    dtype = F32(tt)
    ndim = length(tile_shape)

    # TensorView type (1D with dynamic shape/stride)
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = fill(DYNAMIC_SHAPE, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

    # For tensor view, we need size and stride values
    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get grid size as proxy for array size (same pattern as load)
    nb_x, _, _ = encode_GetNumTileBlocksOp!(cb, scalar_i32, scalar_i32, scalar_i32)
    size_val = nb_x
    stride_val = nb_x

    # Create tensor view from pointer
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, [size_val], [stride_val])

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Use provided index or default to 0
    if isempty(index_vals)
        idx_type = tile_type!(tt, I32(tt), Int[])
        idx_bytes = reinterpret(UInt8, [Int32(0)])
        index_vals = [encode_ConstantOp!(cb, idx_type, collect(idx_bytes))]
    end

    token_type = Token(tt)
    encode_StoreViewTkoOp!(cb, token_type, tile_val, partition, index_vals)

    return nothing
end

#=============================================================================
 Helper functions for extracting constants
=============================================================================#

"""
    extract_constant_int(val) -> Union{Int, Nothing}

Try to extract a constant integer from an SSA value.
"""
function extract_constant_int(@nospecialize(val))
    if val isa Integer
        return Int(val)
    elseif val isa QuoteNode && val.value isa Integer
        return Int(val.value)
    end
    return nothing
end

"""
    extract_constant_tuple(val) -> Union{Tuple, Nothing}

Try to extract a constant tuple from an SSA value.
"""
function extract_constant_tuple(@nospecialize(val))
    if val isa Tuple
        return val
    elseif val isa QuoteNode && val.value isa Tuple
        return val.value
    end
    return nothing
end

"""
    emit_add!(tr, args, result_type, res_type) -> Value

Emit an addition operation.
"""
function emit_add!(tr::Translation, args::AbstractVector,
                   @nospecialize(result_type), res_type::TypeId)
    cb = tr.code_builder

    if length(args) != 2
        error("Addition requires exactly 2 arguments")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])

    if lhs === nothing || rhs === nothing
        error("Cannot resolve operands for addition")
    end

    # Use the type of the first operand if available (for tile operations)
    # This ensures tile operations use tile types, not scalar types
    actual_type = get_value_type(tr, lhs)
    if actual_type === nothing
        actual_type = res_type
    end

    if is_float_type(result_type)
        result = encode_AddFOp!(cb, actual_type, lhs, rhs)
    else
        result = encode_AddIOp!(cb, actual_type, lhs, rhs)
    end

    # Track the result type
    set_value_type!(tr, result, actual_type)
    return result
end

"""
    emit_sub!(tr, args, result_type, res_type) -> Value

Emit a subtraction operation.
"""
function emit_sub!(tr::Translation, args::AbstractVector,
                   @nospecialize(result_type), res_type::TypeId)
    cb = tr.code_builder

    if length(args) == 1
        # Unary negation - not yet implemented
        error("Unary negation not yet supported")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])

    if lhs === nothing || rhs === nothing
        error("Cannot resolve operands for subtraction")
    end

    if is_float_type(result_type)
        return encode_SubFOp!(cb, res_type, lhs, rhs)
    else
        return encode_SubIOp!(cb, res_type, lhs, rhs)
    end
end

"""
    emit_mul!(tr, args, result_type, res_type) -> Value

Emit a multiplication operation.
"""
function emit_mul!(tr::Translation, args::AbstractVector,
                   @nospecialize(result_type), res_type::TypeId)
    cb = tr.code_builder

    if length(args) != 2
        error("Multiplication requires exactly 2 arguments")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])

    if lhs === nothing || rhs === nothing
        error("Cannot resolve operands for multiplication")
    end

    if is_float_type(result_type)
        return encode_MulFOp!(cb, res_type, lhs, rhs)
    else
        return encode_MulIOp!(cb, res_type, lhs, rhs)
    end
end

"""
    emit_constant!(tr, value, result_type) -> Value

Emit a constant value.
"""
function emit_constant!(tr::Translation, @nospecialize(value), @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    res_type = tile_type_for_julia!(tr, result_type)

    # Convert value to bytes
    bytes = constant_to_bytes(value, result_type)
    return encode_ConstantOp!(cb, res_type, bytes)
end

"""
    constant_to_bytes(value, T) -> Vector{UInt8}

Convert a Julia value to bytes for a Tile IR constant.
"""
function constant_to_bytes(@nospecialize(value), @nospecialize(T::Type))
    if T === Bool
        return UInt8[value ? 0xff : 0x00]
    elseif T === Int32 || T === UInt32
        return reinterpret(UInt8, [Int32(value)])
    elseif T === Int64 || T === UInt64
        return reinterpret(UInt8, [Int64(value)])
    elseif T === Float32
        return reinterpret(UInt8, [Float32(value)])
    elseif T === Float64
        return reinterpret(UInt8, [Float64(value)])
    else
        error("Cannot convert $T to constant bytes")
    end
end

"""
    compile_kernel(f, argtypes; name=nothing, sm_arch="sm_100") -> Vector{UInt8}

Compile a Julia function to Tile IR bytecode.
"""
function compile_kernel(@nospecialize(f), @nospecialize(argtypes);
                        name::Union{String, Nothing} = nothing,
                        sm_arch::String = "sm_100")
    target = TileTarget(f, argtypes)

    kernel_name = name === nothing ? string(target.mi.def.name) : name

    buf = write_bytecode!(1) do writer, func_buf
        emit_kernel!(writer, func_buf, target; name=kernel_name)
    end

    return buf
end
