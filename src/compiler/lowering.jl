# Lowering: Julia IR -> Structured IR
#
# Provides the main entry point for converting Julia CodeInfo to StructuredCodeInfo.

#=============================================================================
 Main Entry Point
=============================================================================#

"""
    lower_to_structured_ir(code::CodeInfo) -> StructuredCodeInfo

Convert Julia CodeInfo to StructuredCodeInfo.

For straight-line code (no control flow), this creates a single block containing
all statements. For code with control flow, it builds a control tree and converts
it to nested structured control flow.
"""
function lower_to_structured_ir(code::CodeInfo)
    stmts = code.code
    n = length(stmts)

    if n == 0
        # Empty function
        entry = Block(1)
        return StructuredCodeInfo(code, entry)
    end

    # Check if the code is straight-line (no control flow)
    if is_straight_line(code)
        return lower_straight_line(code)
    else
        return lower_with_control_flow(code)
    end
end

"""
    lower_to_structured_ir(target::TileTarget) -> StructuredCodeInfo

Convert a TileTarget to StructuredCodeInfo.
"""
function lower_to_structured_ir(target::TileTarget)
    lower_to_structured_ir(target.ci)
end

#=============================================================================
 Straight-Line Code
=============================================================================#

"""
    is_straight_line(code::CodeInfo) -> Bool

Check if the code contains no control flow (no branches, just return at end).
"""
function is_straight_line(code::CodeInfo)
    stmts = code.code
    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            return false
        elseif stmt isa GotoIfNot
            return false
        elseif stmt isa ReturnNode
            # Return is only allowed as the last statement for straight-line
            if i != length(stmts)
                return false
            end
        end
    end
    return true
end

"""
    lower_straight_line(code::CodeInfo) -> StructuredCodeInfo

Lower straight-line code (no control flow) to structured IR.
Simply places all non-control-flow statements in a single block.
"""
function lower_straight_line(code::CodeInfo)
    stmts = code.code
    n = length(stmts)

    entry = Block(1)

    for i in 1:n
        stmt = stmts[i]
        if stmt isa ReturnNode
            # Set as terminator, don't include in stmts
            entry.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot)
            push!(entry.stmts, i)
        end
    end

    return StructuredCodeInfo(code, entry)
end

#=============================================================================
 Control Flow Code
=============================================================================#

"""
    lower_with_control_flow(code::CodeInfo) -> StructuredCodeInfo

Lower code with control flow using the restructuring algorithm.
"""
function lower_with_control_flow(code::CodeInfo)
    # Build CFG
    cfg = julia_cfg(code)

    # Build control tree
    control_tree = build_control_tree(cfg, code)

    # Convert to structured IR
    return control_tree_to_structured_ir(control_tree, code)
end

#=============================================================================
 Validation
=============================================================================#

"""
    validate_structured_ir(sci::StructuredCodeInfo) -> Bool

Validate that the structured IR is well-formed.
Returns true if valid, throws an error otherwise.
"""
function validate_structured_ir(sci::StructuredCodeInfo)
    code = sci.code
    n = length(code.code)

    # Collect all referenced statement indices
    referenced = Set{Int}()
    each_stmt(sci.entry) do idx
        if idx < 1 || idx > n
            error("Invalid statement index $idx (code has $n statements)")
        end
        if idx in referenced
            error("Statement $idx referenced multiple times")
        end
        push!(referenced, idx)
    end

    # Note: Not all statements need to be referenced (e.g., control flow statements
    # become implicit in the structure)

    return true
end

#=============================================================================
 Debugging Utilities
=============================================================================#

"""
    dump_structured_ir(sci::StructuredCodeInfo)

Print the structured IR for debugging.
"""
function dump_structured_ir(sci::StructuredCodeInfo)
    print_structured_ir(stdout, sci)
end

"""
    dump_julia_ir(code::CodeInfo)

Print the Julia IR for debugging.
"""
function dump_julia_ir(code::CodeInfo)
    println("Julia IR:")
    for (i, stmt) in enumerate(code.code)
        ssatype = code.ssavaluetypes[i]
        println("  %$i = $stmt :: $ssatype")
    end
end

"""
    compare_ir(code::CodeInfo, sci::StructuredCodeInfo)

Compare Julia IR and structured IR side-by-side for debugging.
"""
function compare_ir(code::CodeInfo, sci::StructuredCodeInfo)
    println("=== Julia IR ===")
    dump_julia_ir(code)
    println()
    println("=== Structured IR ===")
    dump_structured_ir(sci)
end
