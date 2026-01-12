// Second pass: compact into expected output format.

struct Uniforms {
    voxelResolution: u32,
    brickResolution: u32,
    voxelSize: f32,
    numTriangles: u32,
    meshMinBounds: vec3<f32>,
    _pad: u32,
    brickStart: u32,
    brickEnd: u32,
    _pad2: vec2<u32>,
}

struct BrickOutput {
    brickGridIndex: u32,
    lodColor: u32,
    dataOffset: u32,
    numOccupied: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> denseColors: array<u32>;
@group(0) @binding(3) var<storage, read_write> brickOutputs: array<BrickOutput>;
@group(0) @binding(4) var<storage, read_write> packedColors: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;

//================================//
// A popcount is used to count the number of set bits in a 32-bit integer.
fn popcount(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    v = v + (v >> 8u);
    v = v + (v >> 16u);
    return v & 0x3Fu;
}

//================================//
fn unpackColor(packed: u32) -> vec3<u32> 
{
    return vec3<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu
    );
}

//================================//
@compute @workgroup_size(64)
fn c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let localBrickIndex = gid.x;
    let bricksThisPass = uniforms.brickEnd - uniforms.brickStart;
    if (localBrickIndex >= bricksThisPass) { return; } // Out of scope
    
    // Count occupied voxels in this brick
    var occupiedCount = 0u;
    let occBase = localBrickIndex * 16u;
    for (var i = 0u; i < 16u; i++) 
    {
        occupiedCount += popcount(occupancy[occBase + i]);
    }
    if (occupiedCount == 0u) { return; } // Not occupied at all? Skip
    
    let outputIndex = atomicAdd(&counters[0], 1u);
    let colorOffset = atomicAdd(&counters[1], occupiedCount);
    
    // global to local brick index
    let globalBrickIndex = uniforms.brickStart + localBrickIndex;
    let brickZ = globalBrickIndex / (uniforms.brickResolution * uniforms.brickResolution);
    let brickY = (globalBrickIndex / uniforms.brickResolution) % uniforms.brickResolution;
    let brickX = globalBrickIndex % uniforms.brickResolution;
    
    // Voxel base in global voxel coordinates
    let voxelBase = vec3<u32>(brickX, brickY, brickZ) * 8u;
    
    // compute LOD average color
    var colorSum = vec3<u32>(0u);
    var colorIdx = 0u;
    
    for (var localZ = 0u; localZ < 8u; localZ++) {
        for (var localY = 0u; localY < 8u; localY++) {
            for (var localX = 0u; localX < 8u; localX++) 
            {
                let localVoxelIdx = localX + localY * 8u + localZ * 64u;
                let wordIdx = localVoxelIdx / 32u;
                let bitIdx = localVoxelIdx % 32u;
                
                // This just checks if the voxel is occupied
                if ((occupancy[occBase + wordIdx] & (1u << bitIdx)) != 0u) 
                {
                    let localDenseIdx = localBrickIndex * 512u + localVoxelIdx;
                    
                    let packedColor = denseColors[localDenseIdx];
                    packedColors[colorOffset + colorIdx] = packedColor;
                    colorIdx++;
                    
                    let rgb = unpackColor(packedColor);
                    colorSum += rgb;
                }
            }
        }
    }
    
    // Average all the sum to get one LOD color
    let avgR = colorSum.x / occupiedCount;
    let avgG = colorSum.y / occupiedCount;
    let avgB = colorSum.z / occupiedCount;
    
    // Store as local indices, since there will be multiple passes, 
    // In the cpu Side I retake them and transform to global
    // depending on the brickStart and brickEnd offsets.
    brickOutputs[outputIndex].brickGridIndex = localBrickIndex;
    brickOutputs[outputIndex].lodColor = (avgR & 0xFFu) | ((avgG & 0xFFu) << 8u) | ((avgB & 0xFFu) << 16u);
    brickOutputs[outputIndex].dataOffset = colorOffset;
    brickOutputs[outputIndex].numOccupied = occupiedCount;
}