using Test
import cuTile as ct

const sm_arch = "sm_120"

@testset "cuTile" begin

@testset "Tile type" begin
    @test eltype(ct.Tile{Float32, (16,)}) == Float32
    @test eltype(ct.Tile{Float64, (32, 32)}) == Float64
    @test ct.tile_shape(ct.Tile{Float32, (16,)}) == (16,)
    @test ct.tile_shape(ct.Tile{Float32, (32, 32)}) == (32, 32)
end

@testset "load/store 1D" begin
    function copy_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end
    ct.compile(copy_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "load/store 2D" begin
    function copy_2d_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    ct.compile(copy_2d_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "add" begin
    function add_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a + tile_b
        ct.store(c, pid, result)
        return
    end
    ct.compile(add_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "sub" begin
    function sub_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a - tile_b
        ct.store(c, pid, result)
        return
    end
    ct.compile(sub_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "mul" begin
    function mul_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a * tile_b
        ct.store(c, pid, result)
        return
    end
    ct.compile(mul_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "transpose" begin
    function transpose_kernel(x::Ptr{Float32}, y::Ptr{Float32})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(x, (bidx, bidy), (32, 32))
        transposed = ct.transpose(tile)
        ct.store(y, (bidy, bidx), transposed)
        return
    end
    ct.compile(transpose_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

end
