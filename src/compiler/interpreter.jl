using Base.Experimental: @MethodTable

# Create the cuTile method table
@MethodTable cuTileMethodTable

# Create cached method table type based on version compatibility
const cuTileMethodTableView = CC.CachedMethodTable{CC.OverlayMethodTable}

function get_method_table_view(world::UInt)
    CC.CachedMethodTable(CC.OverlayMethodTable(world, cuTileMethodTable))
end

"""
Custom interpreter that supports overlay method tables for cuTile compilation.
This is necessary because NativeInterpreter has a fixed method_table type parameter.
"""
struct cuTileInterpreter <: CC.AbstractInterpreter
    world::UInt
    method_table::cuTileMethodTableView
    inf_cache::Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function cuTileInterpreter(world::UInt=Base.get_world_counter();
                           always_inline::Bool=true)
    method_table = get_method_table_view(world)
    inf_cache = Vector{CC.InferenceResult}()
    inf_params = CC.InferenceParams()
    opt_params = if always_inline
        CC.OptimizationParams(; inline_cost_threshold=typemax(Int))
    else
        CC.OptimizationParams()
    end
    return cuTileInterpreter(world, method_table, inf_cache, inf_params, opt_params)
end

# Required AbstractInterpreter interface methods
CC.InferenceParams(interp::cuTileInterpreter) = interp.inf_params
CC.OptimizationParams(interp::cuTileInterpreter) = interp.opt_params
CC.get_inference_cache(interp::cuTileInterpreter) = interp.inf_cache

# World age
@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(interp::cuTileInterpreter) = interp.world
else
    CC.get_world_counter(interp::cuTileInterpreter) = interp.world
end

# Method table - this enables the overlays
CC.method_table(interp::cuTileInterpreter) = interp.method_table

# Locking - not needed for non-cached compilation
CC.lock_mi_inference(::cuTileInterpreter, ::MethodInstance) = nothing
CC.unlock_mi_inference(::cuTileInterpreter, ::MethodInstance) = nothing

# Cache owner - we don't cache results to the global cache
CC.cache_owner(::cuTileInterpreter) = nothing

# Optimization flags
CC.may_optimize(::cuTileInterpreter) = true
CC.may_compress(::cuTileInterpreter) = true
CC.may_discard_trees(::cuTileInterpreter) = true

# Disable semi-concrete interpretation (broken with overlays per JuliaLang/julia#47349)
function CC.concrete_eval_eligible(interp::cuTileInterpreter,
    @nospecialize(f), result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    ret = @invoke CC.concrete_eval_eligible(interp::CC.AbstractInterpreter,
        f::Any, result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    if ret === :semi_concrete_eval
        return :none
    end
    return ret
end


#=============================================================================
 Public API
=============================================================================#

"""
    get_typed_ir(f, argtypes; world=Base.get_world_counter(), optimize=true, always_inline=true) -> (CodeInfo, return_type)

Get the typed SSA IR for a function with the given argument types.
Uses cuTile's method overlay table to redirect Base operations to cuTile intrinsics.
If optimize=true (default), returns optimized IR with clean SSA form.
If optimize=false, returns IR that preserves all intermediate assignments.
If always_inline=true (default), forces all functions to be inlined.
"""
function get_typed_ir(@nospecialize(f), @nospecialize(argtypes);
                      world::UInt = Base.get_world_counter(),
                      optimize::Bool = true,
                      always_inline::Bool = true)
    sig = Base.signature_type(f, argtypes)
    interp = cuTileInterpreter(world; always_inline)
    results = Base.code_typed_by_type(sig; optimize, world, interp)

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
