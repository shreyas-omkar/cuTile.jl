# Declarative IR Rewrite Pattern Framework
#
# Worklist-based fixpoint driver inspired by MLIR's GreedyPatternRewriteDriver.
# Patterns compile into pattern/rewrite node trees. The driver processes a LIFO
# worklist until fixpoint: when a rewrite fires, affected instructions are
# re-added to the worklist for further matching. Dead code cleanup is delegated
# to the pipeline's dce_pass!.
#
# Usage:
#   rules = RewriteRule[
#       @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
#               Intrinsics.fma(~x, ~y, ~z)
#       @rewrite Core.Intrinsics.slt_int(~x, ~y) =>
#               Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Signed))
#   ]
#   rewrite_patterns!(sci, rules)

using Core: SSAValue

#=============================================================================
 Pattern & Rewrite Nodes
=============================================================================#

abstract type PatternNode end
struct PCall <: PatternNode; func::Any; operands::Vector{PatternNode}; end
struct PBind <: PatternNode; name::Symbol; end
struct PTypedBind <: PatternNode; name::Symbol; type::Type; end
struct POneUse <: PatternNode; inner::PatternNode; end

abstract type RewriteNode end
struct RCall <: RewriteNode; func::Any; operands::Vector{RewriteNode}; end
struct RBind <: RewriteNode; name::Symbol; end
struct RConst <: RewriteNode; val::Any; end

"""
    RFunc(func)

Imperative rewrite node. The function is called with
`(sci, block, inst, match, driver)` and returns `true` if the rewrite was
applied, `false` to skip this rule and try the next one.
"""
struct RFunc <: RewriteNode; func::Function; end

struct RewriteRule
    lhs::PCall
    rhs::RewriteNode
    guard::Union{Function, Nothing}  # (match, driver) -> Bool, or nothing
end
RewriteRule(lhs::PCall, rhs::RewriteNode) = RewriteRule(lhs, rhs, nothing)

root_func(rule::RewriteRule) = rule.lhs.func

#=============================================================================
 @rewrite / @rewriter Macros
=============================================================================#

"""
    @rewrite lhs => rhs
    @rewrite(lhs => rhs, guard)

Compile a declarative rewrite rule. LHS: `func(args...)` matches calls,
`~x` binds (repeated names require equality), `~x::T` binds with type constraint,
`one_use(pat)` requires single use. RHS: `func(args...)` emits calls,
`~x` references bindings, `\$(expr)` injects a literal constant.
Optional `guard` is a function `(match, driver) -> Bool` checked after pattern match.
"""
macro rewrite(ex, guard=nothing)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewrite expects: lhs => rhs")
    g = guard === nothing ? :nothing : guard
    esc(:(RewriteRule($(_compile_lhs(ex.args[2])), $(_compile_rhs(ex.args[3])), $g)))
end

"""
    @rewriter lhs => func

Declarative pattern with imperative rewrite. LHS uses the same pattern syntax as
`@rewrite`. RHS is a function `(sci, block, inst, match, driver) -> Bool` that
performs the rewrite and returns `true`, or returns `false` to skip and try the
next rule.
"""
macro rewriter(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewriter expects: lhs => func")
    esc(:(RewriteRule($(_compile_lhs(ex.args[2])), RFunc($(ex.args[3])))))
end

function _compile_lhs(ex)
    ex isa Expr && ex.head === :call || error("@rewrite LHS: expected call, got $ex")
    f = ex.args[1]
    if f === :~
        inner = ex.args[2]
        if inner isa Expr && inner.head === :(::)
            return :(PTypedBind($(QuoteNode(inner.args[1])), $(inner.args[2])))
        end
        return :(PBind($(QuoteNode(inner))))
    end
    f === :one_use && return :(POneUse($(_compile_lhs(ex.args[2]))))
    :(PCall($f, PatternNode[$(_compile_lhs.(ex.args[2:end])...)]))
end

function _compile_rhs(ex)
    if ex isa Expr && ex.head === :$
        return :(RConst($(ex.args[1])))
    end
    ex isa Expr && ex.head === :call || error("@rewrite RHS: expected call or \$const, got $ex")
    f = ex.args[1]
    f === :~ && return :(RBind($(QuoteNode(ex.args[2]))))
    :(RCall($f, RewriteNode[$(_compile_rhs.(ex.args[2:end])...)]))
end

#=============================================================================
 Worklist
=============================================================================#

mutable struct Worklist
    list::Vector{Int}       # SSA indices (-1 = removed sentinel)
    member::Dict{Int, Int}  # ssa_idx -> position in list
end

Worklist() = Worklist(Int[], Dict{Int, Int}())

function Base.push!(wl::Worklist, ssa_idx::Int)
    haskey(wl.member, ssa_idx) && return
    push!(wl.list, ssa_idx)
    wl.member[ssa_idx] = length(wl.list)
end

function Base.pop!(wl::Worklist)
    while !isempty(wl.list)
        idx = pop!(wl.list)
        idx == -1 && continue
        delete!(wl.member, idx)
        return idx
    end
    return nothing
end

function remove!(wl::Worklist, ssa_idx::Int)
    pos = get(wl.member, ssa_idx, 0)
    pos == 0 && return
    wl.list[pos] = -1
    delete!(wl.member, ssa_idx)
end

Base.isempty(wl::Worklist) = isempty(wl.member)

#=============================================================================
 Driver State
=============================================================================#

struct DefEntry
    block::Block
    ssa_idx::Int
    func::Any
end

"""Operands of a DefEntry, read from the live IR."""
function _def_operands(entry::DefEntry)
    pos = findfirst(==(entry.ssa_idx), entry.block.body.ssa_idxes)
    pos === nothing && return Any[]
    call = resolve_call(entry.block.body.stmts[pos])
    call === nothing && return Any[]
    _, ops = call
    return ops
end

mutable struct RewriteDriver
    sci::StructuredIRCode
    defs::Dict{Int, DefEntry}
    dispatch::Dict{Any, Vector{RewriteRule}}
    worklist::Worklist
    max_rewrites::Int
end

"""Compute fresh use count for an SSA value."""
_use_count(driver::RewriteDriver, val::SSAValue) =
    length(uses(driver.sci.entry, val))

# Codegen no-ops that pattern matching traces through transparently.
_is_transparent(func) = func === Intrinsics.to_scalar ||
                         func === Intrinsics.from_scalar ||
                         func === Intrinsics.broadcast

#=============================================================================
 Notifications
=============================================================================#

"""Add operand-producing instructions to the worklist (enables cascading)."""
function _add_operands_to_worklist!(driver::RewriteDriver, entry::DefEntry)
    for op in _def_operands(entry)
        op isa SSAValue || continue
        haskey(driver.defs, op.id) && push!(driver.worklist, op.id)
    end
end

"""Add instructions that use `val` to the worklist (their operand changed)."""
function _add_users_to_worklist!(driver::RewriteDriver, ssa_idx::Int)
    for inst in users(driver.sci.entry, SSAValue(ssa_idx))
        push!(driver.worklist, inst.ssa_idx)
    end
end

"""Erase an instruction and notify the worklist."""
function _erase_op!(driver::RewriteDriver, entry::DefEntry)
    _add_operands_to_worklist!(driver, entry)
    pos = findfirst(==(entry.ssa_idx), entry.block.body.ssa_idxes)
    if pos !== nothing
        deleteat!(entry.block.body.ssa_idxes, pos)
        deleteat!(entry.block.body.stmts, pos)
        deleteat!(entry.block.body.types, pos)
    end
    delete!(driver.defs, entry.ssa_idx)
    remove!(driver.worklist, entry.ssa_idx)
end

"""Register a newly inserted instruction."""
function _notify_insert!(driver::RewriteDriver, block::Block, ssa_idx::Int, func)
    driver.defs[ssa_idx] = DefEntry(block, ssa_idx, func)
    push!(driver.worklist, ssa_idx)
end

#=============================================================================
 Matching
=============================================================================#

struct MatchResult
    bindings::Dict{Symbol, Any}
    matched_ssas::Vector{Int}
end

"""Merge bindings, requiring repeated names to bind the same value (=== equality)."""
function _merge_bindings!(dest::Dict{Symbol,Any}, src::Dict{Symbol,Any})
    for (k, v) in src
        if haskey(dest, k)
            dest[k] === v || return false
        else
            dest[k] = v
        end
    end
    return true
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PCall,
                       block::Block=driver.sci.entry)
    val isa SSAValue || return nothing
    entry = get(driver.defs, val.id, nothing)
    entry === nothing && return nothing

    if entry.func === pat.func
        ops = _def_operands(entry)
        if length(ops) == length(pat.operands)
            result = MatchResult(Dict{Symbol,Any}(), Int[val.id])
            for (op, sub) in zip(ops, pat.operands)
                m = pattern_match(driver, op, sub, entry.block)
                m === nothing && return nothing
                _merge_bindings!(result.bindings, m.bindings) || return nothing
                append!(result.matched_ssas, m.matched_ssas)
            end
            return result
        end
    end

    # Trace through single-use transparent ops to find the underlying operation
    if _is_transparent(entry.func)
        _use_count(driver, val) == 1 || return nothing
        ops = _def_operands(entry)
        isempty(ops) && return nothing
        if entry.func === Intrinsics.broadcast
            inner = ops[1]
            if inner isa SSAValue
                inner_entry = get(driver.defs, inner.id, nothing)
                if inner_entry !== nothing
                    it = value_type(entry.block, inner)
                    ot = value_type(entry.block, val)
                    it !== nothing && ot !== nothing || return nothing
                    CC.widenconst(it) <: Tile && CC.widenconst(ot) <: Tile || return nothing
                    size(CC.widenconst(it)) == size(CC.widenconst(ot)) || return nothing
                end
            end
        end
        result = pattern_match(driver, ops[1], pat, entry.block)
        result === nothing && return nothing
        push!(result.matched_ssas, val.id)
        return result
    end
    return nothing
end

pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PBind, block::Block=driver.sci.entry) =
    MatchResult(Dict{Symbol,Any}(pat.name => val), Int[])

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PTypedBind,
                       block::Block=driver.sci.entry)
    T = value_type(block, val)
    T === nothing && return nothing
    CC.widenconst(T) <: pat.type || return nothing
    MatchResult(Dict{Symbol,Any}(pat.name => val), Int[])
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::POneUse,
                       block::Block=driver.sci.entry)
    val isa SSAValue && _use_count(driver, val) == 1 || return nothing
    pattern_match(driver, val, pat.inner, block)
end

#=============================================================================
 Rewrite Application
=============================================================================#

"""Resolve an RHS operand, inserting sub-calls before `ref` as needed."""
_resolve_rhs(driver, block, ref, op::RBind, bindings, typ) = bindings[op.name]
_resolve_rhs(driver, block, ref, op::RConst, bindings, typ) = op.val
function _resolve_rhs(driver::RewriteDriver, block, ref, op::RCall, bindings, typ)
    operands = Any[_resolve_rhs(driver, block, ref, sub, bindings, typ) for sub in op.operands]
    inst = insert_before!(block, ref, Expr(:call, op.func, operands...), typ)
    _notify_insert!(driver, block, inst.ssa_idx, op.func)
    SSAValue(inst.ssa_idx)
end

function _apply_rewrite!(driver::RewriteDriver, block, inst_ssa, rule, match)
    entry = driver.defs[inst_ssa]
    if rule.rhs isa RFunc
        # Look up live instruction for RFunc interface
        pos = findfirst(==(inst_ssa), block.body.ssa_idxes)
        pos === nothing && return false
        inst = Instruction(inst_ssa, block.body.stmts[pos], block.body.types[pos])
        rule.rhs.func(driver.sci, block, inst, match, driver) || return false
        return true
    elseif rule.rhs isa RBind
        # Forwarding: replace all uses of root with the bound value, delete root.
        # Collect users BEFORE replace_uses! updates their operands.
        _add_users_to_worklist!(driver, inst_ssa)
        replace_uses!(driver.sci.entry, SSAValue(inst_ssa), match.bindings[rule.rhs.name])
        _erase_op!(driver, entry)
    else
        # Substitution: delete matched intermediates, replace root in-place.
        # Only delete intermediates that have no remaining uses.
        # Transparent-op tracing may have added intermediates to matched_ssas
        # that have uses outside the matched chain.
        for dead_ssa in match.matched_ssas
            dead_ssa == inst_ssa && continue
            dead_entry = get(driver.defs, dead_ssa, nothing)
            dead_entry === nothing && continue
            _use_count(driver, SSAValue(dead_ssa)) == 0 || continue
            _erase_op!(driver, dead_entry)
        end
        pos = findfirst(==(inst_ssa), block.body.ssa_idxes)
        typ = block.body.types[pos]
        operands = Any[_resolve_rhs(driver, block, SSAValue(inst_ssa), op, match.bindings, typ)
                       for op in rule.rhs.operands]
        block.body.stmts[pos] = Expr(:call, rule.rhs.func, operands...)
        # Update defs, re-add self and users to worklist (statement changed)
        driver.defs[inst_ssa] = DefEntry(block, inst_ssa, rule.rhs.func)
        push!(driver.worklist, inst_ssa)
        _add_users_to_worklist!(driver, inst_ssa)
    end
end

#=============================================================================
 Driver
=============================================================================#

"""
    rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule}; max_rewrites=10_000)

Apply rewrite rules to the structured IR using a worklist-based fixpoint driver.
Rules are tried until no more matches fire or `max_rewrites` is reached.
Dead code left behind is cleaned up by the pipeline's `dce_pass!`.
"""
function rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule};
                           max_rewrites::Int=10_000)
    # Build dispatch table
    dispatch = Dict{Any, Vector{RewriteRule}}()
    for rule in rules
        push!(get!(dispatch, root_func(rule), RewriteRule[]), rule)
    end

    # Build defs index
    defs = Dict{Int, DefEntry}()
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(inst)
            call === nothing && continue
            func, _ = call
            defs[inst.ssa_idx] = DefEntry(block, inst.ssa_idx, func)
        end
    end

    # Seed worklist (forward order → reversed by LIFO → processes top-down)
    wl = Worklist()
    for block in eachblock(sci)
        for inst in instructions(block)
            haskey(defs, inst.ssa_idx) && push!(wl, inst.ssa_idx)
        end
    end

    driver = RewriteDriver(sci, defs, dispatch, wl, max_rewrites)

    num_rewrites = 0
    while !isempty(driver.worklist) && num_rewrites < driver.max_rewrites
        ssa_idx = pop!(driver.worklist)::Int
        entry = get(driver.defs, ssa_idx, nothing)
        entry === nothing && continue

        # Verify instruction is still live in its block
        pos = findfirst(==(ssa_idx), entry.block.body.ssa_idxes)
        pos === nothing && begin
            delete!(driver.defs, ssa_idx)
            continue
        end

        # Trivial dead-op elimination: if this op has no uses and is pure,
        # erase it. This keeps use counts accurate for `one_use` patterns
        # (e.g., FMA fusion needs mulf's dead transparent-op users removed
        # so the mulf reads as single-use). Full DCE handles the rest.
        if _use_count(driver, SSAValue(ssa_idx)) == 0
            stmt = entry.block.body.stmts[pos]
            if !must_keep(stmt)
                _erase_op!(driver, entry)
                continue
            end
        end

        # Look up applicable rules by function
        applicable = get(driver.dispatch, entry.func, nothing)
        applicable === nothing && continue

        for rule in applicable
            m = pattern_match(driver, SSAValue(ssa_idx), rule.lhs)
            m === nothing && continue
            rule.guard !== nothing && !rule.guard(m, driver) && continue
            _apply_rewrite!(driver, entry.block, ssa_idx, rule, m)
            num_rewrites += 1
            break
        end
    end
end
