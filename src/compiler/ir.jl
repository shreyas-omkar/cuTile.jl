# Intermediate IR for control flow restructuring
#
# This is a minimal, generic IR for converting unstructured Julia control flow
# to nested structured control flow suitable for Tile IR emission.

#=============================================================================
 Block Arguments (for loop carried values)
=============================================================================#

"""
    BlockArg

Represents a block argument (similar to MLIR block arguments).
Used for loop carried values and condition branch results.
"""
struct BlockArg
    id::Int
    type::Any  # Julia type
end

#=============================================================================
 IR Values - references to SSA values or block arguments
=============================================================================#

"""
    IRValue

Union type for values that can be used in the IR.
Can reference original Julia SSA values or block arguments.
"""
const IRValue = Union{SSAValue, BlockArg, Argument, SlotNumber}

#=============================================================================
 Terminator Operations
=============================================================================#

"""
    YieldOp

Yields values from a structured control flow region (if/loop body).
The yielded values become the results of the containing IfOp/LoopOp.
"""
struct YieldOp
    values::Vector{IRValue}
end

YieldOp() = YieldOp(IRValue[])

"""
    ContinueOp

Continue to the next iteration of a loop with updated carried values.
"""
struct ContinueOp
    values::Vector{IRValue}
end

ContinueOp() = ContinueOp(IRValue[])

"""
    BreakOp

Break out of a loop, yielding values.
"""
struct BreakOp
    values::Vector{IRValue}
end

BreakOp() = BreakOp(IRValue[])

const Terminator = Union{ReturnNode, YieldOp, ContinueOp, BreakOp, Nothing}

#=============================================================================
 Structured Control Flow Operations
=============================================================================#

# Forward declaration for Block (needed for mutual recursion)
abstract type ControlFlowOp end

"""
    Block

A basic block containing statements and potentially nested control flow.
Statements are indices into the original CodeInfo.code array.
"""
mutable struct Block
    id::Int
    args::Vector{BlockArg}           # Block arguments (for loop carried values)
    stmts::Vector{Int}               # Indices into CodeInfo.code (original SSA positions)
    nested::Vector{ControlFlowOp}    # Structured control flow ops within this block
    terminator::Terminator           # ReturnNode, ContinueOp, YieldOp, BreakOp, or nothing
end

Block(id::Int) = Block(id, BlockArg[], Int[], ControlFlowOp[], nothing)

function Base.show(io::IO, block::Block)
    print(io, "Block(id=", block.id)
    if !isempty(block.args)
        print(io, ", args=", length(block.args))
    end
    print(io, ", stmts=", length(block.stmts))
    if !isempty(block.nested)
        print(io, ", nested=", length(block.nested))
    end
    print(io, ")")
end

"""
    IfOp <: ControlFlowOp

Structured if-then-else with nested blocks.
Both branches must yield values of the same types.
"""
struct IfOp <: ControlFlowOp
    condition::IRValue               # SSAValue or BlockArg for the condition
    then_block::Block
    else_block::Block
    result_vars::Vector{SSAValue}    # SSA values that receive the yielded results
end

function Base.show(io::IO, op::IfOp)
    print(io, "IfOp(cond=", op.condition,
          ", then=Block(", op.then_block.id, ")",
          ", else=Block(", op.else_block.id, ")",
          ", results=", length(op.result_vars), ")")
end

"""
    ForOp <: ControlFlowOp

Structured for loop with known bounds.
Used when loop bounds can be determined (e.g., from range iteration).
"""
struct ForOp <: ControlFlowOp
    lower::IRValue                   # Lower bound
    upper::IRValue                   # Upper bound (exclusive)
    step::IRValue                    # Step value
    init_values::Vector{IRValue}     # Initial values for loop-carried variables
    body::Block                      # Has induction var + carried vars as args
    result_vars::Vector{SSAValue}    # SSA values that receive final results
end

function Base.show(io::IO, op::ForOp)
    print(io, "ForOp(lower=", op.lower, ", upper=", op.upper, ", step=", op.step,
          ", init=", length(op.init_values),
          ", body=Block(", op.body.id, ")",
          ", results=", length(op.result_vars), ")")
end

"""
    LoopOp <: ControlFlowOp

General loop with dynamic exit condition.
Used for while loops and when bounds cannot be determined.
"""
struct LoopOp <: ControlFlowOp
    init_values::Vector{IRValue}     # Initial values for loop-carried variables
    body::Block                      # Has carried vars as block args
    result_vars::Vector{SSAValue}    # SSA values that receive final results
end

function Base.show(io::IO, op::LoopOp)
    print(io, "LoopOp(init=", length(op.init_values),
          ", body=Block(", op.body.id, ")",
          ", results=", length(op.result_vars), ")")
end

#=============================================================================
 StructuredCodeInfo - the structured IR for a function
=============================================================================#

"""
    StructuredCodeInfo

Represents a function's code after restructuring.
Keeps the original Julia IR intact while providing a structured view
with nested control flow.
"""
struct StructuredCodeInfo
    code::CodeInfo                   # Original Julia IR - statements accessed by index
    entry::Block                     # Structured view with nested control flow
end

#=============================================================================
 Iteration Utilities
=============================================================================#

"""
    each_block(f, block::Block)

Recursively iterate over all blocks, calling f on each.
"""
function each_block(f, block::Block)
    f(block)
    for op in block.nested
        each_block_in_op(f, op)
    end
end

function each_block_in_op(f, op::IfOp)
    each_block(f, op.then_block)
    each_block(f, op.else_block)
end

function each_block_in_op(f, op::ForOp)
    each_block(f, op.body)
end

function each_block_in_op(f, op::LoopOp)
    each_block(f, op.body)
end

"""
    each_stmt(f, block::Block)

Recursively iterate over all statement indices.
"""
function each_stmt(f, block::Block)
    for stmt_idx in block.stmts
        f(stmt_idx)
    end
    for op in block.nested
        each_stmt_in_op(f, op)
    end
end

function each_stmt_in_op(f, op::IfOp)
    each_stmt(f, op.then_block)
    each_stmt(f, op.else_block)
end

function each_stmt_in_op(f, op::ForOp)
    each_stmt(f, op.body)
end

function each_stmt_in_op(f, op::LoopOp)
    each_stmt(f, op.body)
end

#=============================================================================
 Pretty Printing (MLIR SCF-style)
=============================================================================#

"""
    IRPrinter

Context for printing structured IR with proper indentation and value formatting.
"""
struct IRPrinter
    io::IO
    code::CodeInfo
    indent::Int
end

IRPrinter(io::IO, code::CodeInfo) = IRPrinter(io, code, 0)
indent(p::IRPrinter, n::Int=1) = IRPrinter(p.io, p.code, p.indent + n)

function print_indent(p::IRPrinter)
    for _ in 1:p.indent
        print(p.io, "  ")
    end
end

# Format an IR value for printing
function format_value(p::IRPrinter, v::SSAValue)
    string("%", v.id)
end
function format_value(p::IRPrinter, v::BlockArg)
    string("%arg", v.id)
end
function format_value(p::IRPrinter, v::Argument)
    # Use slot names if available from CodeInfo
    if v.n <= length(p.code.slotnames)
        name = p.code.slotnames[v.n]
        return string(name)
    else
        return string("_", v.n)
    end
end
function format_value(p::IRPrinter, v::SlotNumber)
    string("slot#", v.id)
end
function format_value(p::IRPrinter, v::QuoteNode)
    repr(v.value)
end
function format_value(p::IRPrinter, v::GlobalRef)
    string(v.mod, ".", v.name)
end
function format_value(p::IRPrinter, v)
    repr(v)
end

# Format type for printing (compact form)
function format_type(t)
    if t isa Core.Const
        string("Const(", repr(t.val), ")")
    elseif t isa Type
        string(t)
    else
        string(t)
    end
end

# Format result variables
function format_results(p::IRPrinter, results::Vector{SSAValue})
    if isempty(results)
        ""
    elseif length(results) == 1
        r = results[1]
        typ = p.code.ssavaluetypes[r.id]
        string(format_value(p, r), " : ", format_type(typ))
    else
        parts = [string(format_value(p, r), " : ", format_type(p.code.ssavaluetypes[r.id]))
                 for r in results]
        string("(", join(parts, ", "), ")")
    end
end

# Print a statement from the original CodeInfo
function print_stmt(p::IRPrinter, stmt_idx::Int)
    stmt = p.code.code[stmt_idx]
    ssatype = p.code.ssavaluetypes[stmt_idx]

    print_indent(p)

    # Print as: %N = <expr> : Type
    print(p.io, "%", stmt_idx, " = ")
    print_expr(p, stmt)
    println(p.io, " : ", format_type(ssatype))
end

# Print an expression (RHS of a statement)
function print_expr(p::IRPrinter, expr::Expr)
    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        print(p.io, format_value(p, func), "(")
        print(p.io, join([format_value(p, a) for a in args], ", "))
        print(p.io, ")")
    elseif expr.head == :invoke
        mi = expr.args[1]
        func = expr.args[2]
        args = expr.args[3:end]
        fname = mi isa Core.MethodInstance ? mi.def.name : format_value(p, func)
        print(p.io, "invoke ", fname, "(")
        print(p.io, join([format_value(p, a) for a in args], ", "))
        print(p.io, ")")
    elseif expr.head == :new
        typ = expr.args[1]
        args = expr.args[2:end]
        print(p.io, "new ", typ, "(")
        print(p.io, join([format_value(p, a) for a in args], ", "))
        print(p.io, ")")
    elseif expr.head == :foreigncall
        print(p.io, "foreigncall ", repr(expr.args[1]))
    else
        print(p.io, expr.head, " ")
        print(p.io, join([format_value(p, a) for a in expr.args], ", "))
    end
end

function print_expr(p::IRPrinter, node::PhiNode)
    print(p.io, "φ (")
    parts = String[]
    for (edge, val) in zip(node.edges, node.values)
        if val isa SSAValue || val isa BlockArg || val isa Argument
            push!(parts, string("#", edge, " => ", format_value(p, val)))
        else
            push!(parts, string("#", edge, " => ", repr(val)))
        end
    end
    print(p.io, join(parts, ", "))
    print(p.io, ")")
end

function print_expr(p::IRPrinter, node::PiNode)
    print(p.io, "π (", format_value(p, node.val), ", ", node.typ, ")")
end

function print_expr(p::IRPrinter, node::GotoNode)
    print(p.io, "goto #", node.label)
end

function print_expr(p::IRPrinter, node::GotoIfNot)
    print(p.io, "goto #", node.dest, " if not ", format_value(p, node.cond))
end

function print_expr(p::IRPrinter, node::ReturnNode)
    if isdefined(node, :val)
        print(p.io, "return ", format_value(p, node.val))
    else
        print(p.io, "return")
    end
end

function print_expr(p::IRPrinter, v)
    print(p.io, format_value(p, v))
end

# Print block arguments (for loops and structured control flow)
function print_block_args(p::IRPrinter, args::Vector{BlockArg})
    if isempty(args)
        return
    end
    parts = [string("%arg", a.id, " : ", format_type(a.type)) for a in args]
    print(p.io, "(", join(parts, ", "), ")")
end

# Print iteration arguments with initial values
function print_iter_args(p::IRPrinter, args::Vector{BlockArg}, init_values::Vector{IRValue})
    if isempty(args)
        return
    end
    parts = String[]
    for (arg, init) in zip(args, init_values)
        push!(parts, string("%arg", arg.id, " = ", format_value(p, init), " : ", format_type(arg.type)))
    end
    print(p.io, " iter_args(", join(parts, ", "), ")")
end

# Print a terminator
function print_terminator(p::IRPrinter, term::ReturnNode)
    print_indent(p)
    if isdefined(term, :val)
        println(p.io, "return ", format_value(p, term.val))
    else
        println(p.io, "return")
    end
end

function print_terminator(p::IRPrinter, term::YieldOp)
    print_indent(p)
    if isempty(term.values)
        println(p.io, "scf.yield")
    else
        println(p.io, "scf.yield ", join([format_value(p, v) for v in term.values], ", "))
    end
end

function print_terminator(p::IRPrinter, term::ContinueOp)
    print_indent(p)
    if isempty(term.values)
        println(p.io, "scf.continue")
    else
        println(p.io, "scf.continue ", join([format_value(p, v) for v in term.values], ", "))
    end
end

function print_terminator(p::IRPrinter, term::BreakOp)
    print_indent(p)
    if isempty(term.values)
        println(p.io, "scf.break")
    else
        println(p.io, "scf.break ", join([format_value(p, v) for v in term.values], ", "))
    end
end

function print_terminator(p::IRPrinter, ::Nothing)
    # No terminator
end

# Print a block's contents (statements, nested ops, terminator)
function print_block_body(p::IRPrinter, block::Block)
    # Print statements
    for stmt_idx in block.stmts
        print_stmt(p, stmt_idx)
    end

    # Print nested control flow
    for op in block.nested
        print_control_flow(p, op)
    end

    # Print terminator
    print_terminator(p, block.terminator)
end

# Print IfOp (MLIR SCF style)
function print_control_flow(p::IRPrinter, op::IfOp)
    print_indent(p)

    # Print results assignment if any
    if !isempty(op.result_vars)
        print(p.io, format_results(p, op.result_vars), " = ")
    end

    # scf.if %cond {
    println(p.io, "scf.if ", format_value(p, op.condition), " {")

    # Then block body
    print_block_body(indent(p), op.then_block)

    # } else {
    print_indent(p)
    println(p.io, "} else {")

    # Else block body
    print_block_body(indent(p), op.else_block)

    # }
    print_indent(p)
    println(p.io, "}")
end

# Print ForOp (MLIR SCF style)
function print_control_flow(p::IRPrinter, op::ForOp)
    print_indent(p)

    # Print results assignment if any
    if !isempty(op.result_vars)
        print(p.io, format_results(p, op.result_vars), " = ")
    end

    # scf.for %iv = %lb to %ub step %step iter_args(...) {
    print(p.io, "scf.for ")

    # Induction variable (first block arg if present)
    if !isempty(op.body.args)
        iv = op.body.args[1]
        print(p.io, "%arg", iv.id, " = ")
    end

    print(p.io, format_value(p, op.lower), " to ", format_value(p, op.upper),
          " step ", format_value(p, op.step))

    # Print iteration arguments (remaining block args after induction var)
    if length(op.body.args) > 1
        carried_args = op.body.args[2:end]
        print_iter_args(p, carried_args, op.init_values)
    end

    println(p.io, " {")

    # Body
    print_block_body(indent(p), op.body)

    print_indent(p)
    println(p.io, "}")
end

# Print LoopOp (MLIR SCF while style)
function print_control_flow(p::IRPrinter, op::LoopOp)
    print_indent(p)

    # Print results assignment if any
    if !isempty(op.result_vars)
        print(p.io, format_results(p, op.result_vars), " = ")
    end

    # scf.while iter_args(...) {
    print(p.io, "scf.while")
    print_iter_args(p, op.body.args, op.init_values)
    println(p.io, " {")

    # Body
    print_block_body(indent(p), op.body)

    print_indent(p)
    println(p.io, "}")
end

# Main entry point: show for StructuredCodeInfo
function Base.show(io::IO, ::MIME"text/plain", sci::StructuredCodeInfo)
    # Print header
    println(io, "StructuredCodeInfo {")

    p = IRPrinter(io, sci.code, 1)

    # Print block arguments for entry (function parameters)
    if !isempty(sci.entry.args)
        print_indent(p)
        print(io, "^entry")
        print_block_args(p, sci.entry.args)
        println(io, ":")
    end

    # Print entry block body
    print_block_body(p, sci.entry)

    println(io, "}")
end

# Keep the simple show method for compact display
function Base.show(io::IO, sci::StructuredCodeInfo)
    print(io, "StructuredCodeInfo(", length(sci.code.code), " stmts, entry=Block#", sci.entry.id, ")")
end

# Legacy function - delegates to show
function print_structured_ir(io::IO, sci::StructuredCodeInfo)
    show(io, MIME"text/plain"(), sci)
end

print_structured_ir(sci::StructuredCodeInfo) = print_structured_ir(stdout, sci)
