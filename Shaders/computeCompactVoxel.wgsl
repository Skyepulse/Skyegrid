// computeCompactVoxel.wgsl - Pass 2: Compact colors and compute LOD

struct Uniforms {
    voxelResolution: u32,
    brickResolution: u32,
    voxelSize: f32,
    numTriangles: u32,
    meshMinBounds: vec3<f32>,
    _pad: u32,
}

struct BrickOutput {
    brickGridIndex: u32,
    lodColor: u32,      // packed RGB + flags
    dataOffset: u32,
    numOccupied: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> occupancy: array<u32>;
@group(0) @binding(2) var<storage, read> denseColors: array<u32>;
@group(0) @binding(3) var<storage, read_write> brickOutputs: array<BrickOutput>;
@group(0) @binding(4) var<storage, read_write> packedColors: array<u32>; // Output packed colors
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>; // [0]: brick counter, [1]: color counter

fn popcount(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    v = v + (v >> 8u);
    v = v + (v >> 16u);
    return v & 0x3Fu;
}

fn unpackColor(packed: u32) -> vec3<u32> {
    return vec3<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu
    );
}

@compute @workgroup_size(64)
fn c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let brickIndex = gid.x;
    let totalBricks = uniforms.brickResolution * uniforms.brickResolution * uniforms.brickResolution;
    if (brickIndex >= totalBricks) { return; }
    
    // Count occupied voxels in this brick
    var occupiedCount = 0u;
    let occBase = brickIndex * 16u;
    
    for (var i = 0u; i < 16u; i++) {
        occupiedCount += popcount(occupancy[occBase + i]);
    }
    
    if (occupiedCount == 0u) { return; }
    
    // Brick has voxels - allocate output slot
    let outputIndex = atomicAdd(&counters[0], 1u);
    let colorOffset = atomicAdd(&counters[1], occupiedCount);
    
    // Compute brick grid coordinates
    let brickZ = brickIndex / (uniforms.brickResolution * uniforms.brickResolution);
    let brickY = (brickIndex / uniforms.brickResolution) % uniforms.brickResolution;
    let brickX = brickIndex % uniforms.brickResolution;
    
    // Compute voxel base
    let voxelBase = vec3<u32>(brickX, brickY, brickZ) * 8u;
    
    // Gather colors and compute LOD average
    var colorSum = vec3<u32>(0u);
    var colorIdx = 0u;
    
    for (var localZ = 0u; localZ < 8u; localZ++) {
        for (var localY = 0u; localY < 8u; localY++) {
            for (var localX = 0u; localX < 8u; localX++) {
                let localIdx = localX + localY * 8u + localZ * 64u;
                let wordIdx = localIdx / 32u;
                let bitIdx = localIdx % 32u;
                
                if ((occupancy[occBase + wordIdx] & (1u << bitIdx)) != 0u) {
                    let voxel = voxelBase + vec3<u32>(localX, localY, localZ);
                    let linearIdx = voxel.x + voxel.y * uniforms.voxelResolution + voxel.z * uniforms.voxelResolution * uniforms.voxelResolution;
                    
                    let packedColor = denseColors[linearIdx];
                    packedColors[colorOffset + colorIdx] = packedColor;
                    colorIdx++;
                    
                    let rgb = unpackColor(packedColor);
                    colorSum += rgb;
                }
            }
        }
    }
    
    // Compute LOD color (average)
    let avgR = colorSum.x / occupiedCount;
    let avgG = colorSum.y / occupiedCount;
    let avgB = colorSum.z / occupiedCount;
    
    // Store brick output
    brickOutputs[outputIndex].brickGridIndex = brickIndex;
    brickOutputs[outputIndex].lodColor = (avgR & 0xFFu) | ((avgG & 0xFFu) << 8u) | ((avgB & 0xFFu) << 16u);
    brickOutputs[outputIndex].dataOffset = colorOffset;
    brickOutputs[outputIndex].numOccupied = occupiedCount;
}