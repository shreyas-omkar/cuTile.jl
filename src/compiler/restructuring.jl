# Control flow restructuring - converts unstructured Julia IR to nested structured IR
#
# Adapted from SPIRV.jl's structural_analysis.jl

using Graphs: SimpleDiGraph, add_edge!, vertices, edges, nv, ne,
              inneighbors, outneighbors, Edge, strongly_connected_components

#=============================================================================
 Region Types
=============================================================================#

"""
Single-entry control-flow structure classified according to well-specified patterns.
"""
@enum RegionType begin
    REGION_BLOCK           # Sequence of blocks
    REGION_IF_THEN         # Conditional with one branch
    REGION_IF_THEN_ELSE    # Conditional with two branches merging
    REGION_SWITCH          # Multi-way branch
    REGION_TERMINATION     # Block ending with function termination
    REGION_PROPER          # Acyclic region not matching other patterns
    REGION_SELF_LOOP       # Single block with self-edge
    REGION_WHILE_LOOP      # Simple while loop pattern
    REGION_NATURAL_LOOP    # General single-entry loop
    REGION_IMPROPER        # Multiple-entry cyclic region
end

#=============================================================================
 Control Flow Graph from Julia IR
=============================================================================#

"""
    julia_cfg(code::CodeInfo) -> SimpleDiGraph

Build a CFG from Julia CodeInfo.
"""
function julia_cfg(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    cfg = SimpleDiGraph(n)

    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            add_edge!(cfg, i, stmt.label)
        elseif stmt isa GotoIfNot
            # Fallthrough edge
            add_edge!(cfg, i, i + 1)
            # Branch edge
            add_edge!(cfg, i, stmt.dest)
        elseif stmt isa ReturnNode
            # No outgoing edges
        else
            # Fallthrough to next statement
            if i < n
                add_edge!(cfg, i, i + 1)
            end
        end
    end

    return cfg
end

"""
    find_basic_blocks(code::CodeInfo) -> Vector{UnitRange{Int}}

Find basic block boundaries in Julia IR.
Returns ranges of statement indices for each basic block.
"""
function find_basic_blocks(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    n == 0 && return UnitRange{Int}[]

    # Find block starts (targets of jumps and start)
    is_block_start = falses(n)
    is_block_start[1] = true

    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            is_block_start[stmt.label] = true
        elseif stmt isa GotoIfNot
            is_block_start[stmt.dest] = true
            if i + 1 <= n
                is_block_start[i + 1] = true
            end
        end
    end

    # Also mark after GotoNode/ReturnNode as block starts
    for (i, stmt) in enumerate(stmts)
        if (stmt isa GotoNode || stmt isa ReturnNode) && i + 1 <= n
            is_block_start[i + 1] = true
        end
    end

    # Build block ranges
    blocks = UnitRange{Int}[]
    block_start = 1
    for i in 2:n
        if is_block_start[i]
            push!(blocks, block_start:i-1)
            block_start = i
        end
    end
    push!(blocks, block_start:n)

    return blocks
end

#=============================================================================
 Dominator Analysis using Julia's Core.Compiler
=============================================================================#

"""
    compute_dominators(cfg::SimpleDiGraph) -> Dict{Int, Set{Int}}

Compute dominator sets for each vertex.
"""
function compute_dominators(cfg::SimpleDiGraph)
    n = nv(cfg)
    n == 0 && return Dict{Int, Set{Int}}()

    # Initialize: entry dominated only by itself, others by all
    doms = Dict{Int, Set{Int}}()
    entry = 1  # Julia IR always starts at statement 1
    all_verts = Set(1:n)

    for v in 1:n
        if v == entry
            doms[v] = Set([entry])
        else
            doms[v] = copy(all_verts)
        end
    end

    # Iterate until fixed point
    changed = true
    while changed
        changed = false
        for v in 2:n
            preds = inneighbors(cfg, v)
            if isempty(preds)
                continue
            end

            # dom(v) = {v} ∪ ∩{dom(p) : p ∈ preds(v)}
            new_doms = copy(doms[first(preds)])
            for p in preds[2:end]
                intersect!(new_doms, doms[p])
            end
            push!(new_doms, v)

            if new_doms != doms[v]
                doms[v] = new_doms
                changed = true
            end
        end
    end

    return doms
end

"""
    compute_immediate_dominators(doms::Dict{Int, Set{Int}}) -> Dict{Int, Int}

Compute immediate dominator for each vertex.
"""
function compute_immediate_dominators(doms::Dict{Int, Set{Int}})
    idoms = Dict{Int, Int}()

    for (v, dom_set) in doms
        # Entry node has no idom
        if length(dom_set) == 1
            continue
        end

        # Find the immediate dominator: the dominator that is dominated by all others
        candidates = setdiff(dom_set, Set([v]))
        for c in candidates
            # c is idom if all other dominators (except v) dominate c
            is_idom = true
            for other in candidates
                if other != c && !(other in doms[c])
                    is_idom = false
                    break
                end
            end
            if is_idom
                idoms[v] = c
                break
            end
        end
    end

    return idoms
end

#=============================================================================
 Back Edge Detection
=============================================================================#

"""
    find_back_edges(cfg::SimpleDiGraph, doms::Dict{Int, Set{Int}}) -> Set{Edge{Int}}

Find back edges in the CFG (edges where target dominates source).
"""
function find_back_edges(cfg::SimpleDiGraph, doms::Dict{Int, Set{Int}})
    back_edges = Set{Edge{Int}}()

    for e in edges(cfg)
        src, dst = e.src, e.dst
        # Back edge if dst dominates src
        if haskey(doms, src) && dst in doms[src]
            push!(back_edges, e)
        end
    end

    return back_edges
end

#=============================================================================
 Control Tree Node
=============================================================================#

"""
    ControlNode

A node in the control tree, containing block index and region type.
"""
struct ControlNode
    index::Int           # Starting block/statement index
    region_type::RegionType
end

"""
    ControlTree

Tree structure representing hierarchical control flow regions.
Leaves are REGION_BLOCK nodes.
"""
mutable struct ControlTree
    node::ControlNode
    parent::Union{ControlTree, Nothing}
    children::Vector{ControlTree}
end

ControlTree(index::Int, region_type::RegionType) =
    ControlTree(ControlNode(index, region_type), nothing, ControlTree[])

node_index(tree::ControlTree) = tree.node.index
region_type(tree::ControlTree) = tree.node.region_type

function Base.show(io::IO, tree::ControlTree)
    print(io, "ControlTree(", tree.node.index, ", ", tree.node.region_type, ")")
end

#=============================================================================
 Helper Functions
=============================================================================#

"""
    is_single_entry_single_exit(cfg, v)

Check if vertex v has exactly one incoming and one outgoing edge.
"""
is_single_entry_single_exit(cfg, v) =
    length(inneighbors(cfg, v)) == 1 && length(outneighbors(cfg, v)) == 1

#=============================================================================
 Region Pattern Matching
=============================================================================#

"""
    detect_block_region(cfg, v, visited, consumed)

Detect a linear sequence of blocks starting at v.
Returns the sequence of vertices if found, nothing otherwise.
"""
function detect_block_region(cfg, v, visited, consumed)
    # Must have at most 1 in-edge and 1 out-edge to start
    length(outneighbors(cfg, v)) > 1 && return nothing
    length(inneighbors(cfg, v)) > 1 && return nothing

    # Walk forward
    vs = [v]
    curr = v
    while length(outneighbors(cfg, curr)) == 1
        next = only(outneighbors(cfg, curr))
        length(inneighbors(cfg, next)) == 1 || break
        length(outneighbors(cfg, next)) > 1 && break
        next in vs && break  # Avoid cycles
        next in visited && break
        next in consumed && break  # Don't include already consumed vertices
        push!(vs, next)
        curr = next
    end

    length(vs) == 1 && return nothing
    return vs
end

"""
    detect_if_then(cfg, v, consumed)

Detect if-then pattern at v (conditional with one branch).
Returns (condition_block, then_block, merge_block) or nothing.
"""
function detect_if_then(cfg, v, consumed)
    outs = outneighbors(cfg, v)
    length(outs) == 2 || return nothing

    a, b = outs

    # Check if a is the "then" block going to b (merge)
    if !(a in consumed) &&
       is_single_entry_single_exit(cfg, a) &&
       only(inneighbors(cfg, a)) == v &&
       only(outneighbors(cfg, a)) == b
        return (v, a, b)
    end

    # Check if b is the "then" block going to a (merge)
    if !(b in consumed) &&
       is_single_entry_single_exit(cfg, b) &&
       only(inneighbors(cfg, b)) == v &&
       only(outneighbors(cfg, b)) == a
        return (v, b, a)
    end

    return nothing
end

"""
    detect_if_then_else(cfg, v, consumed)

Detect if-then-else pattern at v.
Returns (condition_block, then_block, else_block, merge_block) or nothing.
"""
function detect_if_then_else(cfg, v, consumed)
    outs = outneighbors(cfg, v)
    length(outs) == 2 || return nothing

    a, b = outs

    # Neither branch can be consumed
    a in consumed && return nothing
    b in consumed && return nothing

    # Both branches must be SESE
    is_single_entry_single_exit(cfg, a) || return nothing
    is_single_entry_single_exit(cfg, b) || return nothing

    # Both must come from v
    only(inneighbors(cfg, a)) == v || return nothing
    only(inneighbors(cfg, b)) == v || return nothing

    # Both must go to same merge point
    merge_a = only(outneighbors(cfg, a))
    merge_b = only(outneighbors(cfg, b))
    merge_a == merge_b || return nothing

    # Merge can't be v (would be a loop)
    merge_a == v && return nothing

    return (v, a, b, merge_a)
end

"""
    detect_while_loop(cfg, v, back_edges, consumed)

Detect while loop pattern at v.
Returns (header, body, exit) or nothing.
"""
function detect_while_loop(cfg, v, back_edges, consumed)
    ins = inneighbors(cfg, v)
    outs = outneighbors(cfg, v)

    # Header needs 2 in-edges (entry + back) and 2 out-edges (body + exit)
    length(ins) == 2 || return nothing
    length(outs) == 2 || return nothing

    a, b = outs

    # Check if a is body looping back to v
    if !(a in consumed) &&
       is_single_entry_single_exit(cfg, a) &&
       only(inneighbors(cfg, a)) == v &&
       only(outneighbors(cfg, a)) == v &&
       Edge(a, v) in back_edges
        return (v, a, b)
    end

    # Check if b is body looping back to v
    if !(b in consumed) &&
       is_single_entry_single_exit(cfg, b) &&
       only(inneighbors(cfg, b)) == v &&
       only(outneighbors(cfg, b)) == v &&
       Edge(b, v) in back_edges
        return (v, b, a)
    end

    return nothing
end

"""
    detect_self_loop(cfg, v, back_edges)

Detect self-loop at v.
"""
function detect_self_loop(cfg, v, back_edges)
    return Edge(v, v) in back_edges
end

#=============================================================================
 Control Tree Construction
=============================================================================#

"""
    build_control_tree(cfg::SimpleDiGraph, code::CodeInfo; max_iterations=10000) -> ControlTree

Build a control tree from the CFG.
"""
function build_control_tree(cfg::SimpleDiGraph, code::CodeInfo; max_iterations::Int=10000)
    n = nv(cfg)
    n == 0 && return ControlTree(0, REGION_BLOCK)

    doms = compute_dominators(cfg)
    back_edges = find_back_edges(cfg, doms)

    # Initialize each vertex as a block region
    trees = Dict(v => ControlTree(v, REGION_BLOCK) for v in 1:n)
    visited = Set{Int}()
    consumed = Set{Int}()  # Vertices that have been merged into a region

    # Process in reverse post-order (approximated by forward order for Julia IR)
    worklist = collect(1:n)

    iterations = 0
    while !isempty(worklist)
        iterations += 1
        if iterations > max_iterations
            @warn "Control tree construction exceeded iteration limit ($max_iterations)"
            break
        end

        v = popfirst!(worklist)
        haskey(trees, v) || continue
        v in visited && continue

        # Try to match region patterns
        region = try_match_region(cfg, v, trees, back_edges, visited, consumed)

        if region !== nothing
            (region_type, included_verts) = region

            # Check if any vertex (other than v) has already been consumed
            non_v_verts = filter(w -> w != v, included_verts)
            if any(w -> w in consumed, non_v_verts)
                # Already consumed, mark as visited and move on
                push!(visited, v)
                continue
            end

            # Create new tree node
            children = [trees[w] for w in included_verts if haskey(trees, w)]
            new_tree = ControlTree(v, region_type)
            new_tree.children = children
            for child in children
                child.parent = new_tree
            end

            # Mark merged vertices as consumed
            for w in included_verts
                if w != v
                    push!(consumed, w)
                    delete!(trees, w)
                end
            end
            trees[v] = new_tree

            # Re-process this vertex for possible further reduction
            # but only if we actually made progress (merged something)
            if !isempty(non_v_verts)
                pushfirst!(worklist, v)
            else
                push!(visited, v)
            end
        else
            push!(visited, v)
        end
    end

    # Return the tree rooted at vertex 1, or construct one if needed
    if haskey(trees, 1)
        return trees[1]
    else
        # Fallback: return first available tree
        return first(values(trees))
    end
end

"""
    try_match_region(cfg, v, trees, back_edges, visited, consumed)

Try to match a region pattern at vertex v.
Returns (RegionType, included_vertices) or nothing.
"""
function try_match_region(cfg, v, trees, back_edges, visited, consumed)
    # Try acyclic patterns first
    result = detect_block_region(cfg, v, visited, consumed)
    if result !== nothing
        return (REGION_BLOCK, result)
    end

    result = detect_if_then_else(cfg, v, consumed)
    if result !== nothing
        (_, then_v, else_v, _) = result
        return (REGION_IF_THEN_ELSE, [v, then_v, else_v])
    end

    result = detect_if_then(cfg, v, consumed)
    if result !== nothing
        (_, then_v, _) = result
        return (REGION_IF_THEN, [v, then_v])
    end

    # Try cyclic patterns
    if detect_self_loop(cfg, v, back_edges)
        return (REGION_SELF_LOOP, [v])
    end

    result = detect_while_loop(cfg, v, back_edges, consumed)
    if result !== nothing
        (_, body_v, _) = result
        return (REGION_WHILE_LOOP, [v, body_v])
    end

    return nothing
end

#=============================================================================
 Control Tree to Structured IR Conversion
=============================================================================#

"""
    control_tree_to_structured_ir(tree::ControlTree, code::CodeInfo) -> StructuredCodeInfo

Convert a control tree to structured IR.
"""
function control_tree_to_structured_ir(tree::ControlTree, code::CodeInfo)
    entry_block = tree_to_block(tree, code, Ref(1))
    return StructuredCodeInfo(code, entry_block)
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, block_id::Ref{Int}) -> Block

Convert a control tree node to a Block.
"""
function tree_to_block(tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    idx = node_index(tree)
    rtype = region_type(tree)
    id = block_id[]
    block_id[] += 1

    block = Block(id)

    if rtype == REGION_BLOCK
        # Collect all statement indices from this block and its children
        collect_statements!(block, tree, code)
    elseif rtype == REGION_IF_THEN_ELSE
        # First child is condition block, then then_block, else_block
        handle_if_then_else!(block, tree, code, block_id)
    elseif rtype == REGION_IF_THEN
        # Condition block followed by then_block
        handle_if_then!(block, tree, code, block_id)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, block_id)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, block_id)
    else
        # For unsupported regions, collect statements sequentially
        collect_statements!(block, tree, code)
    end

    # Set terminator based on last statement
    set_terminator!(block, code)

    return block
end

"""
    collect_statements!(block::Block, tree::ControlTree, code::CodeInfo)

Collect statement indices from a control tree into a block.
"""
function collect_statements!(block::Block, tree::ControlTree, code::CodeInfo)
    idx = node_index(tree)

    if region_type(tree) == REGION_BLOCK && !isempty(tree.children)
        # REGION_BLOCK with children: process children in order
        for child in tree.children
            collect_statements!(block, child, code)
        end
    else
        # Leaf node or single statement
        if 1 <= idx <= length(code.code)
            stmt = code.code[idx]
            # Don't include control flow terminators as regular statements
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                push!(block.stmts, idx)
            elseif stmt isa ReturnNode
                # Mark as terminator, don't include in stmts
                block.terminator = stmt
            end
        end
    end
end

"""
    handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle if-then-else region.
"""
function handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    @assert length(tree.children) >= 3 "if-then-else needs at least 3 children"

    # First child is the condition block
    cond_tree = tree.children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements (up to but not including GotoIfNot)
    if 1 <= cond_idx <= length(code.code)
        stmt = code.code[cond_idx]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(block.stmts, cond_idx)
        end
    end

    # Find the GotoIfNot that determines the condition
    cond_value = find_condition_value(cond_idx, code)

    # Then and else blocks
    then_tree = tree.children[2]
    else_tree = tree.children[3]

    then_block = tree_to_block(then_tree, code, block_id)
    else_block = tree_to_block(else_tree, code, block_id)

    # Create IfOp
    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.nested, if_op)
end

"""
    handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle if-then region (no else branch).
"""
function handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    @assert length(tree.children) >= 2 "if-then needs at least 2 children"

    # First child is the condition block
    cond_tree = tree.children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(code.code)
        stmt = code.code[cond_idx]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(block.stmts, cond_idx)
        end
    end

    cond_value = find_condition_value(cond_idx, code)

    # Then block
    then_tree = tree.children[2]
    then_block = tree_to_block(then_tree, code, block_id)

    # Empty else block
    else_block = Block(block_id[])
    block_id[] += 1

    # Create IfOp
    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.nested, if_op)
end

"""
    handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle loop region.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    # First child is header, second is body
    if isempty(tree.children)
        # Single-block loop
        idx = node_index(tree)
        body_block = Block(block_id[])
        block_id[] += 1
        if 1 <= idx <= length(code.code)
            stmt = code.code[idx]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                push!(body_block.stmts, idx)
            end
        end
        loop_op = LoopOp(IRValue[], body_block, SSAValue[])
        push!(block.nested, loop_op)
    else
        # Header + body
        header_tree = tree.children[1]
        header_idx = node_index(header_tree)

        # Collect header statements
        if 1 <= header_idx <= length(code.code)
            stmt = code.code[header_idx]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                push!(block.stmts, header_idx)
            end
        end

        # Body block
        if length(tree.children) >= 2
            body_tree = tree.children[2]
            body_block = tree_to_block(body_tree, code, block_id)
        else
            body_block = Block(block_id[])
            block_id[] += 1
        end

        loop_op = LoopOp(IRValue[], body_block, SSAValue[])
        push!(block.nested, loop_op)
    end
end

"""
    handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle self-loop region.
"""
function handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    idx = node_index(tree)

    body_block = Block(block_id[])
    block_id[] += 1

    if 1 <= idx <= length(code.code)
        stmt = code.code[idx]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body_block.stmts, idx)
        end
    end

    loop_op = LoopOp(IRValue[], body_block, SSAValue[])
    push!(block.nested, loop_op)
end

"""
    find_condition_value(stmt_idx::Int, code::CodeInfo) -> IRValue

Find the condition value for a GotoIfNot at or near stmt_idx.
"""
function find_condition_value(stmt_idx::Int, code::CodeInfo)
    # Look for GotoIfNot at or after this statement
    for i in stmt_idx:length(code.code)
        stmt = code.code[i]
        if stmt isa GotoIfNot
            cond = stmt.cond
            if cond isa SSAValue
                return cond
            elseif cond isa SlotNumber
                return cond
            else
                # Fallback: condition must be evaluated before
                return SSAValue(i - 1)
            end
        end
        # Stop if we hit another control flow
        if stmt isa GotoNode || stmt isa ReturnNode
            break
        end
    end

    # Fallback: use previous SSA value as condition
    return SSAValue(max(1, stmt_idx - 1))
end

"""
    set_terminator!(block::Block, code::CodeInfo)

Set the block terminator based on statements.
"""
function set_terminator!(block::Block, code::CodeInfo)
    if block.terminator !== nothing
        return
    end

    # Check if the last collected statement leads to a return
    if !isempty(block.stmts)
        last_idx = block.stmts[end]
        if last_idx < length(code.code)
            next_stmt = code.code[last_idx + 1]
            if next_stmt isa ReturnNode
                block.terminator = next_stmt
            end
        end
    end
end
