module cuTile

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArg,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp, Undef

using Base: compilerbarrier, donotdelete
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using CUDA_Tile_jll

using BFloat16s: BFloat16
public BFloat16

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Language definitions
include("language/types.jl")

# Compiler implementation
include("compiler/interface.jl")
include("compiler/codegen.jl")
include("compiler/intrinsics.jl")

# Language implementation
include("language/broadcast.jl")
include("language/overlays.jl")
include("language/arithmetic.jl")
include("language/math.jl")
include("language/operations.jl")
include("language/atomics.jl")

public launch, Tiled, ByTarget, @compiler_options, var"@."
launch(args...) = error("Please import CUDA.jl before using `cuTile.launch`.")

"""
    Tiled(x)

Wrapper that routes broadcast expressions through cuTile kernels.

    Tiled(B) .= A .+ A

Uses Julia's `Base.Broadcast` fusion machinery to build a `Broadcasted` tree,
then dispatches to a generic cuTile kernel that evaluates the tree on tiles.
"""
struct Tiled{A <: AbstractArray}
    parent::A
end
Tiled(x) = x  # passthrough for non-arrays (Numbers, etc.)
Base.parent(t::Tiled) = t.parent
Base.axes(t::Tiled) = axes(parent(t))
Base.size(t::Tiled) = size(parent(t))
Base.ndims(::Tiled{A}) where A = ndims(A)
Base.eltype(::Tiled{A}) where A = eltype(A)
Base.Broadcast.broadcastable(t::Tiled) = t

# Walk dotted AST, wrap value-position leaves in Tiled()
_wrap_tiled(x) = x  # literals pass through
_wrap_tiled(s::Symbol) = :($Tiled($s))
function _wrap_tiled(ex::Expr)
    if ex.head === :.=
        Expr(:.=, _wrap_tiled(ex.args[1]), _wrap_tiled(ex.args[2]))
    elseif ex.head === :. && length(ex.args) == 2 &&
           ex.args[2] isa Expr && ex.args[2].head === :tuple
        # f.(args...) — wrap args, NOT function position
        new_args = map(_wrap_tiled, ex.args[2].args)
        Expr(:., ex.args[1], Expr(:tuple, new_args...))
    else
        Expr(ex.head, map(_wrap_tiled, ex.args)...)
    end
end

"""
    @. expr

Like `Base.@.` but wraps every value-position leaf in `Tiled()`, routing
the broadcast through cuTile kernels.

    using cuTile; const ct = cuTile
    ct.@. C = A + sin(B)
    # equivalent to: Tiled(C) .= Tiled(A) .+ sin.(Tiled(B))
"""
macro var"."(ex)
    esc(_wrap_tiled(Base.Broadcast.__dot__(ex)))
end

end # module cuTile
