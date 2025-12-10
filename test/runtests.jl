using Test
import cuTile as ct

const sm_arch = "sm_120"

@testset "cuTile" verbose=true begin

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
    @test_nowarn ct.compile(copy_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "load/store 2D" begin
    function copy_2d_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    @test_nowarn ct.compile(copy_2d_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
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
    @test_nowarn ct.compile(add_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
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
    @test_nowarn ct.compile(sub_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
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
    @test_nowarn ct.compile(mul_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
end

@testset "load from TileArray 1D" begin
    function tilearray_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end
    # Use a concrete ArraySpec for the type
    spec = ct.ArraySpec{1}(16, true)
    argtypes = Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}
    @test_nowarn ct.compile(tilearray_kernel, argtypes; sm_arch)
end

@testset "load from TileArray 2D" begin
    function tilearray_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    spec = ct.ArraySpec{2}(16, true)
    argtypes = Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}}
    @test_nowarn ct.compile(tilearray_2d_kernel, argtypes; sm_arch)
end

@testset "examples" begin
    function find_sources(path::String, sources=String[])
        if isdir(path)
            for entry in readdir(path)
                find_sources(joinpath(path, entry), sources)
            end
        elseif endswith(path, ".jl")
            push!(sources, path)
        end
        sources
    end

    examples_dir = joinpath(@__DIR__, "..", "examples")
    examples = find_sources(examples_dir)

    cd(examples_dir) do
        @testset for example in examples
            mod = @eval module $(gensym()) end
            @eval mod begin
                redirect_stdout(devnull) do
                    include($example)
                end
            end
        end
    end
end

end
