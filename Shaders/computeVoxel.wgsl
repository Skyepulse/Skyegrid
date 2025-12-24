// Main compute shader for voxel traversal and rendering, will be writing output colors to a texture, and provide
// feedback to the CPU on which voxel bricks are inside the view, so that we upload and allocate only those bricks.

//================================//
//         VOXEL STRUCTS          //
//================================//
struct ComputeVoxelParams 
{
    pixelToRay: mat4x4<f32>,
    cameraOrigin: vec3<f32>,
    numToRender: u32,
    voxelResolution: u32,
    time: f32,
    _pad1: vec2<f32>,
};

//================================//
struct Brick
{
    occupancy: array<u32, 16>, // 8 x u64 = 16 x u32
};

// MAX FEEDBACK SIZE 8192
const MAX_FEEDBACK: u32 = 8192u;

//================================//
//           BINDINGS             //
//================================//
// Output color texture (will then be blitted to screen)
@group(0) @binding(0)
var targetTexture: texture_storage_2d<rgba8unorm, write>;
// Params
@group(0) @binding(1)
var<uniform> params: ComputeVoxelParams;
// Brick grid
@group(0) @binding(2)
var<storage, read_write> brickGrid: array<atomic<u32>>; // atomic write in order to see if already requested by GPU
// Brick pool
@group(0) @binding(3)
var<storage, read> brickPool: array<Brick>;
// Colors
@group(0) @binding(4)
var<storage, read> colorPool: array<u32>;
// Feedback count buffer
@group(0) @binding(5)
var<storage, read_write> feedbackCount: atomic<u32>;
// Feedback indices buffer
@group(0) @binding(6)
var<storage, read_write> feedbackIndices: array<u32>;

//================================//
//           HELPERS              //
//================================//
// Bit layout (example):
// [31]     : resident flag
// [30]     : requested flag (GPU feedback)
// [29]     : unloaded flag (has LOD color)
// [28:24]  : unused
// [23:0]   : brick pool index OR packed LOD color
fn isBrickLoaded(brickIndex: u32) -> bool
{
    let pointer = atomicLoad(&brickGrid[brickIndex]);
    return (pointer & 0x80000000u) != 0u; // Check resident flag at 31st bit
}

//================================//
fn isBrickUnloaded(brickIndex: u32) -> bool
{
    let pointer = atomicLoad(&brickGrid[brickIndex]);
    return (pointer & 0x20000000u) != 0u; // Check unloaded flag at 29th bit
}

//================================//
fn readVoxelData(worldPos: vec3<f32>) -> bool
{
    let voxelCoord: vec3<u32> = worldToVoxelCoord(worldPos);
    let brickCoord: vec3<u32> = voxelToBrickCoord(voxelCoord);
    let brickIndex: u32 = brickToIndex(brickCoord);
    let localCoord: vec3<u32> = voxelToLocalCoord(voxelCoord);
    let pointer = atomicLoad(&brickGrid[brickIndex]);

    // Check if the pointer, points to a valid brick
    if isBrickLoaded(brickIndex) == false
    {
        writeFeedback(brickIndex);
        return false;
    }

    // The adress in the pointer OR LOD COLOR but it is loaded so we know
    // it is a brick index, so it is in [23:0]
    let brickSlot: u32 = pointer & 0x00FFFFFFu;
    let brick: Brick = brickPool[brickSlot];
    return isVoxelSet(brick, localCoord.x, localCoord.y, localCoord.z);
}

//================================//
// We call this in case we get false in readVoxelData (brick not loaded)
// in the case this color is the empty one, we suppose it is an empty brick
fn loadLODColor(worldPos: vec3<f32>) -> vec3<f32>
{
    let voxelCoord: vec3<u32> = worldToVoxelCoord(worldPos);
    let brickCoord: vec3<u32> = voxelToBrickCoord(voxelCoord);
    let brickIndex: u32 = brickToIndex(brickCoord);
    
    return loadLODColorBrickIndex(brickIndex);
}

//================================//
fn loadLODColorBrickIndex(brickIndex: u32) -> vec3<f32>
{
    let pointer = atomicLoad(&brickGrid[brickIndex]);

    // The brick is not loaded, we therefore get the LOD color
    let packedLOD = pointer & 0x00FFFFFFu; // LOD color is in [23:0]

    // Decode r, g, b
    let r: f32 = f32(packedLOD & 255u) / 255.0; // red in 0-7 bits
    let g: f32 = f32((packedLOD >> 8u) & 255u) / 255.0; // green in 8 - 15 bits
    let b: f32 = f32((packedLOD >> 16u) & 255u) / 255.0; // blue in 16 - 23 bits

    return vec3<f32>(r, g, b);
}

//================================//
fn isVoxelSet(brick: Brick, x: u32, y: u32, z: u32) -> bool
{
    let bit = x + y * 8u;           // 0..63
    let word = z * 2u + (bit >> 5u); // Which slice
    let mask = 1u << (bit & 31u);
    return (brick.occupancy[word] & mask) != 0u;
}

//================================//
fn loadColor(brickSlot: u32, voxelIndex: u32) -> vec3<f32>
{
    let start = brickSlot * 512u; // We know each brick in the color pool is represented by 512u
    let packedColor = colorPool[start + voxelIndex];

    // Decode r, g, b
    let r: f32 = f32(packedColor & 255u) / 255.0; // red in 0-7 bits
    let g: f32 = f32((packedColor >> 8u) & 255u) / 255.0; // green in 8 - 15 bits
    let b: f32 = f32((packedColor >> 16u) & 255u) / 255.0; // blue in 16 - 23 bits
    // the rest is padding

    return vec3<f32>(r, g, b);
}

//================================//
fn writeFeedback(brickIndex: u32)
{
    // did we already request this brick?
    let old = atomicOr(&brickGrid[brickIndex], 0x40000000u); // we read and modify request flag at 30th bit,
                                            // if it was 0, we set it to 1 and request it. If it was 1, we do nothing (already requested)
    if (old & 0x40000000u) != 0u
    {
        return; // already requested
    }

    // If not already requested, we write to feedback buffer
    let index = atomicAdd(&feedbackCount, 1u);

    // we know there are at most N bricks, of 8x8x8 voxels, so also
    // check we are under max bricks which is 
    let brickResolution: u32 = params.voxelResolution / 8u;
    let maxBricks: u32 = brickResolution * brickResolution * brickResolution;
    if (index < MAX_FEEDBACK && brickIndex < maxBricks)
    {
        feedbackIndices[index] = brickIndex;
    }
}

//================================//
// Get voxel coord in [0, voxelResolution - 1]^3
fn worldToVoxelCoord(position: vec3<f32>) -> vec3<u32> 
{
    return vec3<u32>(clamp(floor(position), vec3<f32>(0.0), vec3<f32>(f32(params.voxelResolution) - 1.0)));
}

//================================//
fn voxelToBrickCoord(voxelCoord: vec3<u32>) -> vec3<u32>
{
    
    //return vec3<u32>(
    //    voxelCoord.x / 8u,
    //    voxelCoord.y / 8u,
    //    voxelCoord.z / 8u
    //);
    
    return voxelCoord >> vec3<u32>(3u, 3u, 3u); // equivalent to division by 8
}

//================================//
fn brickToIndex(brickCoord: vec3<u32>) -> u32
{
    let gridResolution: u32 = params.voxelResolution / 8u;
    return brickCoord.x + brickCoord.y * gridResolution + brickCoord.z * gridResolution * gridResolution;
}

//================================//
fn voxelToLocalCoord(voxelCoord: vec3<u32>) -> vec3<u32>
{
    return vec3<u32>(
        voxelCoord.x % 8u,  // local.x in [0, 7]
        voxelCoord.y % 8u,  // same for y
        voxelCoord.z % 8u   // same for z
    );
}

//================================//
// will be used in the color pool lookup
fn localCoordToVoxelIndex(localCoord: vec3<u32>) -> u32
{
    return localCoord.x + localCoord.y * 8u + localCoord.z * 64u; // 8 * 8 = 64
}

//================================//
fn brickIndexFromPointer(pointer: u32) -> u32
{
    return pointer & 0x00FFFFFFu; // brick index is in [23:0]
}

//================================//
@compute @workgroup_size(8, 8, 1)
fn c(@builtin(global_invocation_id) gid: vec3<u32>) 
{
    let size: vec2<u32> = textureDimensions(targetTexture);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    // Pixel to ray conversion
    let px = (f32(gid.x) + 0.5) / f32(size.x);
    let py = (f32(gid.y) + 0.5) / f32(size.y);

    let x_ndc = px * 2.0 - 1.0;
    let y_ndc = 1.0 - py * 2.0;

    let rayDir = normalize(
        (params.pixelToRay * vec4<f32>(x_ndc, y_ndc, 1.0, 0.0)).xyz
    );

    let rayOrigin = params.cameraOrigin;

    var steps: i32 = 0;
    var color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    //let hit = traverseGrid(rayOrigin, rayDir, &color, &steps);
    
    // Max num bricks
    color = vec3<f32>(0.0, 0.5, 0.0);
    let maxBricks = (params.voxelResolution / 8u) * (params.voxelResolution / 8u) * (params.voxelResolution / 8u);
    // Produce a unique brick index
    let linearID = gid.y * size.x + gid.x;
    if (linearID >= maxBricks)
    {
        textureStore(
            targetTexture,
            vec2<i32>(i32(gid.x), i32(gid.y)),
            vec4<f32>(color, 1.0)
        );
        return;
    }

    // check if the brick is loaded, if not get the LOD
    let brickIndex: u32 = linearID;
    let brickLoaded: bool = isBrickLoaded(brickIndex);
    let brickUnloaded: bool = isBrickUnloaded(brickIndex);

    if (brickUnloaded)
    {
        color = loadLODColorBrickIndex(brickIndex);
    }
    
    // if unloaded, and our index is less than numToRender, REQUEST IT
    if (!brickLoaded && (linearID < params.numToRender))
    {
        writeFeedback(brickIndex); // this will make it loaded next frame
    }

    if (brickLoaded)
    {
        // get brick slot
        let pointer = atomicLoad(&brickGrid[brickIndex]);
        let brickSlot: u32 = brickIndexFromPointer(pointer);
        
        let localCoord: vec3<u32> = vec3<u32>(0u, 0u, 0u); // start with first voxel
        let voxelIndex: u32 = localCoordToVoxelIndex(localCoord);
        color = loadColor(brickSlot, voxelIndex);
    }

    textureStore(
        targetTexture,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(color, 1.0)
    );
}