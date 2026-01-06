# high-level intrinsics
#
# These handle Julia built-ins and map them to Tile IR intrinsics.


## Julia intrinsics

# We normally do not support Julia intrinsics, instead using overlay methods to ensure
# Tile IR intrinsics are used, but a couple are retained for IRStructurizer support.

function emit_intrinsic!(ctx::CGCtx, func::Core.IntrinsicFunction, args, @nospecialize(result_type))
    if func === Core.Intrinsics.add_int
        emit_intrinsic!(ctx, Intrinsics.addi, args, result_type)

    # Signed less-than comparisons
    elseif func === Core.Intrinsics.slt_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThan, SignednessSigned], result_type)
    elseif func === Core.Intrinsics.sle_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThanOrEqual, SignednessSigned], result_type)

    # Unsigned less-than comparison
    elseif func === Core.Intrinsics.ult_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThan, SignednessUnsigned], result_type)

    else
        missing
    end
end


## Built-in functions

# We cannot overlay built-in functions

function emit_intrinsic!(ctx::CGCtx, ::typeof(===), args, @nospecialize(_))
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for ===")
    rhs === nothing && error("Cannot resolve RHS operand for ===")

    result_type = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa CGVal ? lhs.v : lhs
    rhs_v = rhs isa CGVal ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type, lhs_v, rhs_v;
                              predicate=CmpEqual, signedness=SignednessSigned)

    CGVal(result_v, result_type, Bool, Int[])
end

emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args, @nospecialize(result_type)) = nothing

emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args, @nospecialize(result_type)) = nothing

emit_intrinsic!(ctx::CGCtx, ::typeof(Base.donotdelete), args, @nospecialize(result_type)) = nothing


## Other

## XXX: Tile constructor
function emit_intrinsic!(ctx::CGCtx, ::Type{<:Tile}, args, @nospecialize(result_type))
    # Emit the scalar value
    source = emit_value!(ctx, args[1])

    # Get element type from result type, constant, or source jltype
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = if result_type_unwrapped <: Tile
        result_type_unwrapped.parameters[1]
    elseif source.constant !== nothing
        typeof(source.constant)
    else
        unwrap_type(source.jltype)
    end

    # Return as 0D tile type
    result_jltype = Tile{elem_type, ()}
    CGVal(source.v, source.type_id, result_jltype, source.shape)
end
