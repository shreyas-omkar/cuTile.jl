# Rewrite Patterns
#
# Declarative IR rewrite rules using the @rewrite framework (passes/rewrite.jl).

#=============================================================================
 Scalar View Elimination
=============================================================================#

# Eliminates redundant to_scalar(from_scalar(x, S)) chains that arise from
# Julia's broadcast system wrapping tile arithmetic in type-conversion ops.
# Intermediate broadcasts are handled by the pattern matcher's transparent
# op tracing (sees through single-use no-op broadcasts automatically).

const SVE_RULES = RewriteRule[
    @rewrite Intrinsics.to_scalar(Intrinsics.from_scalar(~x, ~_)) => ~x
]

scalar_view_elim_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, SVE_RULES)

#=============================================================================
 FMA Fusion
=============================================================================#

# Pattern-matches mul+add/sub into fma to reduce register pressure.
# Mirrors cuTile Python's fuse_mul_addsub in rewrite_patterns.py.

const FMA_RULES = RewriteRule[
    @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
            Intrinsics.fma(~x, ~y, ~z)
    @rewrite Intrinsics.addf(~z, one_use(Intrinsics.mulf(~x, ~y))) =>
            Intrinsics.fma(~x, ~y, ~z)
    @rewrite Intrinsics.subf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
            Intrinsics.fma(~x, ~y, Intrinsics.negf(~z))
]

fma_fusion_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, FMA_RULES)
