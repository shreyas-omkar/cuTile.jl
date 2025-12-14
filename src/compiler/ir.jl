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

function Base.show(io::IO, sci::StructuredCodeInfo)
    print(io, "StructuredCodeInfo(entry=", sci.entry, ")")
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
 Pretty Printing
=============================================================================#

"""
    print_structured_ir(io::IO, sci::StructuredCodeInfo)

Pretty print the structured IR for debugging.
"""
function print_structured_ir(io::IO, sci::StructuredCodeInfo)
    println(io, "StructuredCodeInfo:")
    print_block(io, sci.entry, sci.code, 1)
end

print_structured_ir(sci::StructuredCodeInfo) = print_structured_ir(stdout, sci)

function print_block(io::IO, block::Block, code::CodeInfo, indent::Int)
    prefix = "  " ^ indent

    println(io, prefix, "Block ", block.id, ":")

    if !isempty(block.args)
        println(io, prefix, "  args: ", join(["$(a.id):$(a.type)" for a in block.args], ", "))
    end

    # Print statements
    for stmt_idx in block.stmts
        stmt = code.code[stmt_idx]
        ssatype = code.ssavaluetypes[stmt_idx]
        println(io, prefix, "  %", stmt_idx, " = ", stmt, " :: ", ssatype)
    end

    # Print nested control flow
    for op in block.nested
        print_control_flow_op(io, op, code, indent + 1)
    end

    # Print terminator
    if block.terminator !== nothing
        println(io, prefix, "  ", block.terminator)
    end
end

function print_control_flow_op(io::IO, op::IfOp, code::CodeInfo, indent::Int)
    prefix = "  " ^ indent
    println(io, prefix, "IfOp(", op.condition, ") -> ", op.result_vars)
    println(io, prefix, "  then:")
    print_block(io, op.then_block, code, indent + 2)
    println(io, prefix, "  else:")
    print_block(io, op.else_block, code, indent + 2)
end

function print_control_flow_op(io::IO, op::ForOp, code::CodeInfo, indent::Int)
    prefix = "  " ^ indent
    println(io, prefix, "ForOp(", op.lower, ":", op.step, ":", op.upper,
            ", init=", op.init_values, ") -> ", op.result_vars)
    println(io, prefix, "  body:")
    print_block(io, op.body, code, indent + 2)
end

function print_control_flow_op(io::IO, op::LoopOp, code::CodeInfo, indent::Int)
    prefix = "  " ^ indent
    println(io, prefix, "LoopOp(init=", op.init_values, ") -> ", op.result_vars)
    println(io, prefix, "  body:")
    print_block(io, op.body, code, indent + 2)
end
