# high-level intrinsics
#
# these should be converted to regular code, where possible

## XXX: Tile constructor
function emit_intrinsic!(ctx::CodegenContext, ::Type{<:Tile}, args, @nospecialize(result_type))
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

# Skip tuple construction
emit_intrinsic!(ctx::CodegenContext, ::typeof(Core.tuple), args, @nospecialize(result_type)) = nothing

# Skip isa type assertions (inserted by Julia during inlining)
emit_intrinsic!(ctx::CodegenContext, ::typeof(isa), args, @nospecialize(result_type)) = nothing


#-----------------------------------------------------------------------------
# Scalar comparison operators (for loop bounds, etc.)
# These map to cuda_tile.cmpi.
#-----------------------------------------------------------------------------

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(===), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(==)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(!=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpNotEqual)

function emit_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    emit_int_cmp!(ctx, args, predicate, SignednessSigned)
end

function emit_int_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate, signedness::Signedness)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for comparison")
    rhs === nothing && error("Cannot resolve RHS operand for comparison")

    # Result type is a boolean (i1) scalar
    result_type = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa CGVal ? lhs.v : lhs
    rhs_v = rhs isa CGVal ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type, lhs_v, rhs_v;
                              predicate=predicate, signedness=signedness)

    CGVal(result_v, result_type, Bool, Int[])
end
