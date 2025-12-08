# TileInterpreter: Custom AbstractInterpreter for cuTile compilation
#
# This hooks into Julia's type inference to get fully-typed SSA IR
# that we can then translate to Tile IR bytecode.

using Core: MethodInstance, CodeInfo, SSAValue, Argument, GotoNode, GotoIfNot, ReturnNode
using Core.Compiler
const CC = Core.Compiler

"""
    TileInterpreter <: AbstractInterpreter

Custom interpreter for cuTile compilation.
Uses Julia's type inference machinery to get typed IR.
"""
struct TileInterpreter <: CC.AbstractInterpreter
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
end

function TileInterpreter(; world::UInt = Base.get_world_counter(), optimize::Bool = false)
    TileInterpreter(
        world,
        CC.InferenceParams(; aggressive_constant_propagation=false),
        CC.OptimizationParams(; inlining=optimize, inline_cost_threshold=optimize ? 10000 : 0),
        CC.InferenceResult[]
    )
end

# AbstractInterpreter interface implementation
CC.InferenceParams(interp::TileInterpreter) = interp.inf_params
CC.OptimizationParams(interp::TileInterpreter) = interp.opt_params
CC.get_inference_world(interp::TileInterpreter) = interp.world
CC.get_inference_cache(interp::TileInterpreter) = interp.inf_cache
CC.cache_owner(interp::TileInterpreter) = nothing

# Use default method table (no overlays for now)
CC.method_table(interp::TileInterpreter) = CC.CachedMethodTable(CC.InternalMethodTable(interp.world))

"""
    get_typed_ir(f, argtypes; world=Base.get_world_counter(), optimize=true) -> (CodeInfo, return_type)

Get the typed SSA IR for a function with the given argument types.
If optimize=true (default), returns optimized IR with clean SSA form.
If optimize=false, returns IR that preserves all intermediate assignments.
"""
function get_typed_ir(@nospecialize(f), @nospecialize(argtypes);
                      world::UInt = Base.get_world_counter(),
                      optimize::Bool = true)
    # Build the full signature type
    sig = Base.signature_type(f, argtypes)

    # Use Julia's built-in code_typed_by_type which handles optimize flag correctly
    results = Base.code_typed_by_type(sig; optimize, world)

    if isempty(results)
        error("Type inference failed for $f with argument types $argtypes")
    end

    ci, rettype = results[1]
    return ci, rettype
end

"""
    get_method_instance(f, argtypes; world=Base.get_world_counter()) -> MethodInstance

Get the MethodInstance for a function call.
"""
function get_method_instance(@nospecialize(f), @nospecialize(argtypes); world::UInt = Base.get_world_counter())
    tt = Base.signature_type(f, argtypes)
    match = Base._which(tt; world)
    return CC.specialize_method(match)
end
