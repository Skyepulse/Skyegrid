// Main compute shader for voxel traversal and rendering, will be writing output colors to a texture, and provide
// feedback to the CPU on which voxel bricks are inside the view, so that we upload and allocate only those bricks.

//================================//
//         VOXEL STRUCTS          //
//================================//
struct ComputeVoxelParams 
{
    pixelToRay: mat4x4<f32>,
    cameraOrigin: vec3<f32>,
    maxColorBufferSize: u32,
    voxelResolution: u32,
    time: f32,
    hasColor: u32,
    _pad: u32,
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
// Brick grid (read-only for fast parallel access)
@group(0) @binding(2)
var<storage, read> brickGrid: array<u32>;
// Brick pool
@group(0) @binding(3)
var<storage, read> brickPool: array<Brick>;
// Feedback count buffer
@group(0) @binding(4)
var<storage, read_write> feedbackCount: atomic<u32>;
// Feedback indices buffer
@group(0) @binding(5)
var<storage, read_write> feedbackIndices: array<u32>;

// Colors (max 3 buffers)
@group(0) @binding(6)
var<storage, read> colorPool: array<u32>;
@group(0) @binding(7)
var<storage, read> colorPool2: array<u32>;
@group(0) @binding(8)
var<storage, read> colorPool3: array<u32>;

// Separate atomic buffer for request flags (avoids contention on brickGrid reads)
@group(0) @binding(9)
var<storage, read_write> brickRequestFlags: array<atomic<u32>>;

//================================//
//           HELPERS              //
//================================//
// Bit layout (example):
// [31]     : resident flag
// [30]     : requested flag (GPU feedback) - NOW IN SEPARATE BUFFER
// [29]     : unloaded flag (has LOD color)
// [28:24]  : unused
// [23:0]   : brick pool index OR packed LOD color
fn isBrickLoaded(pointer: u32) -> bool
{
    return (pointer & 0x80000000u) != 0u; // Check resident flag at 31st bit
}

//================================//
fn isBrickUnloaded(pointer: u32) -> bool
{
    return (pointer & 0x20000000u) != 0u; // Check unloaded flag at 29th bit
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
fn readColorFromPool(globalOffset: u32) -> u32
{
    let bufferIdx   = globalOffset / params.maxColorBufferSize;
    let localOffset = globalOffset % params.maxColorBufferSize;

    switch (bufferIdx)
    {
        case 0u: { return colorPool[localOffset]; }
        case 1u: { return colorPool2[localOffset]; }
        case 2u: { return colorPool3[localOffset]; }
        default: { return 0u; }
    }
}

//================================//
fn loadColor(brickSlot: u32, voxelIndex: u32) -> vec3<f32>
{
    let globalOffset: u32 = brickSlot * 512u + voxelIndex;  
    let packedColor: u32 = readColorFromPool(globalOffset);

    let r: f32 = f32(packedColor & 255u) / 255.0;
    let g: f32 = f32((packedColor >> 8u) & 255u) / 255.0;
    let b: f32 = f32((packedColor >> 16u) & 255u) / 255.0;

    return vec3<f32>(r, g, b);
}

//================================//
fn loadLODColorFromPointer(pointer: u32) -> vec3<f32>
{
    let packedLOD = pointer & 0x00FFFFFFu;

    let r: f32 = f32(packedLOD & 255u) / 255.0;
    let g: f32 = f32((packedLOD >> 8u) & 255u) / 255.0;
    let b: f32 = f32((packedLOD >> 16u) & 255u) / 255.0;

    return vec3<f32>(r, g, b);
}

//================================//
fn writeFeedback(brickIndex: u32)
{
    // Atomic on SEPARATE buffer - no contention with reads
    let old = atomicOr(&brickRequestFlags[brickIndex], 1u);
    if (old != 0u)
    {
        return; // Already requested this frame
    }

    let index = atomicAdd(&feedbackCount, 1u);

    let brickResolution: u32 = params.voxelResolution / 8u;
    let maxBricks: u32 = brickResolution * brickResolution * brickResolution;
    if (index < MAX_FEEDBACK && brickIndex < maxBricks)
    {
        feedbackIndices[index] = brickIndex;
    }
}

//================================//
fn brickToIndex(brickCoord: vec3<u32>) -> u32
{
    let gridResolution: u32 = params.voxelResolution / 8u;
    return brickCoord.x + brickCoord.y * gridResolution + brickCoord.z * gridResolution * gridResolution;
}

//================================//
fn localCoordToVoxelIndex(localCoord: vec3<u32>) -> u32
{
    return localCoord.x + localCoord.y * 8u + localCoord.z * 64u;
}

//================================//
fn brickSlotFromPointer(pointer: u32) -> u32
{
    return pointer & 0x00FFFFFFu;
}

//================================//
@compute @workgroup_size(8, 8, 1)
fn c(@builtin(global_invocation_id) gid: vec3<u32>) 
{
    let size: vec2<u32> = textureDimensions(targetTexture);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let px = (f32(gid.x) + 0.5) / f32(size.x);
    let py = (f32(gid.y) + 0.5) / f32(size.y);

    let x_ndc = px * 2.0 - 1.0;
    let y_ndc = 1.0 - py * 2.0;

    let rayDir = normalize(
        (params.pixelToRay * vec4<f32>(x_ndc, y_ndc, 1.0, 0.0)).xyz
    );

    let rayOrigin = params.cameraOrigin;

    var color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    let hit = traverseGrid(rayOrigin, rayDir, &color);

    textureStore(
        targetTexture,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(color, 1.0)
    );
}

//================================//
fn traverseGrid(rayOrigin: vec3<f32>, rayDir: vec3<f32>, color: ptr<function, vec3<f32>>) -> bool
{
    let brickSize: f32 = 8.0;
    let brickResolution: i32 = i32(params.voxelResolution) / 8;
    let gridMax: f32 = f32(params.voxelResolution);

    // Compute ray inverse and step direction
    let rayDirInv: vec3<f32> = 1.0 / rayDir;
    let rayDirSign: vec3<f32> = sign(rayDir);
    let stepBrick: vec3<i32> = vec3<i32>(rayDirSign);

    // Compute slab intersection
    let t1: vec3<f32> = (vec3<f32>(0.0) - rayOrigin) * rayDirInv;
    let t2: vec3<f32> = (vec3<f32>(gridMax) - rayOrigin) * rayDirInv;

    let tMin: vec3<f32> = min(t1, t2);
    let tMax: vec3<f32> = max(t1, t2);

    var tNear: f32 = max(max(tMin.x, tMin.y), tMin.z);
    let tFar: f32 = min(min(tMax.x, tMax.y), tMax.z);

    // Miss check
    if (tNear > tFar || tFar < 0.0)
    {
        return false;
    }

    // Clamp tNear to 0 - we don't want to start behind the camera
    tNear = max(tNear, 0.0);

    // Small epsilon to ensure we're inside
    let epsilon: f32 = 0.001;
    tNear = tNear + epsilon;

    // Entry point and starting brick
    let entryPoint: vec3<f32> = rayOrigin + rayDir * tNear;
    var brickCoord: vec3<i32> = vec3<i32>(floor(entryPoint / brickSize));
    brickCoord = clamp(brickCoord, vec3<i32>(0), vec3<i32>(brickResolution - 1));

    let nextBoundary: vec3<f32> = (vec3<f32>(brickCoord) + max(rayDirSign, vec3<f32>(0.0))) * brickSize;
    var tBrick: vec3<f32> = (nextBoundary - rayOrigin) * rayDirInv;
    
    // Ensure tBrick values are not behind us
    tBrick = max(tBrick, vec3<f32>(tNear));
    
    let deltaBrick: vec3<f32> = abs(rayDirInv) * brickSize;

    // Main DDA loop
    for (var i: i32 = 0; i < 512; i = i + 1) // Safety limit
    {
        // Bounds check
        if (brickCoord.x < 0 || brickCoord.y < 0 || brickCoord.z < 0 ||
            brickCoord.x >= brickResolution ||
            brickCoord.y >= brickResolution ||
            brickCoord.z >= brickResolution)
        {
            break;
        }

        let brickIndex: u32 = brickToIndex(vec3<u32>(brickCoord));
        let brickPointer: u32 = brickGrid[brickIndex]; // Non-atomic read to try and speed up

        let brickLoaded: bool = isBrickLoaded(brickPointer);
        let brickUnloaded: bool = isBrickUnloaded(brickPointer);

        if (brickUnloaded)
        {
            (*color) = loadLODColorFromPointer(brickPointer);
            writeFeedback(brickIndex);
            return true;
        }
        else if (brickLoaded)
        {
            let brickSlot: u32 = brickSlotFromPointer(brickPointer);
            let brickMin: vec3<f32> = vec3<f32>(brickCoord) * brickSize;

            // Compute entry/exit for this brick
            let t1_local: vec3<f32> = (brickMin - rayOrigin) * rayDirInv;
            let t2_local: vec3<f32> = (brickMin + brickSize - rayOrigin) * rayDirInv;

            let tEnterVec: vec3<f32> = min(t1_local, t2_local);
            let tExitVec: vec3<f32> = max(t1_local, t2_local);

            var tEnter: f32 = max(max(tEnterVec.x, tEnterVec.y), tEnterVec.z);
            let tExit: f32 = min(min(tExitVec.x, tExitVec.y), tExitVec.z);

            // Don't enter brick from behind camera
            tEnter = max(tEnter, 0.0);

            if (tEnter <= tExit)
            {
                let hit = traverseBrick(rayOrigin, rayDir, tEnter, tExit, brickMin, brickSlot, color);
                if (hit)
                {
                    return true;
                }
            }
        }

        // DDA advance, step to next brick
        if (tBrick.x < tBrick.y)
        {
            if (tBrick.x < tBrick.z)
            {
                brickCoord.x = brickCoord.x + stepBrick.x;
                tBrick.x = tBrick.x + deltaBrick.x;
            }
            else
            {
                brickCoord.z = brickCoord.z + stepBrick.z;
                tBrick.z = tBrick.z + deltaBrick.z;
            }
        }
        else
        {
            if (tBrick.y < tBrick.z)
            {
                brickCoord.y = brickCoord.y + stepBrick.y;
                tBrick.y = tBrick.y + deltaBrick.y;
            }
            else
            {
                brickCoord.z = brickCoord.z + stepBrick.z;
                tBrick.z = tBrick.z + deltaBrick.z;
            }
        }
    }

    return false;
}

//================================//
fn traverseBrick(rayOrigin: vec3<f32>, rayDir: vec3<f32>, tEnter: f32, tExit: f32, brickMin: vec3<f32>, brickSlot: u32, color: ptr<function, vec3<f32>>) -> bool
{
    let rayDirInv: vec3<f32> = 1.0 / rayDir;
    let rayDirSign: vec3<f32> = sign(rayDir);
    let stepVoxel: vec3<i32> = vec3<i32>(rayDirSign);

    // Entry point in local brick space
    let entryPoint: vec3<f32> = rayOrigin + rayDir * tEnter;
    let localEntry: vec3<f32> = entryPoint - brickMin;

    // Starting voxel (clamped to [0,7])
    var voxel: vec3<i32> = vec3<i32>(clamp(floor(localEntry), vec3<f32>(0.0), vec3<f32>(7.0)));

    // DDA setup for voxels
    let nextBoundary: vec3<f32> = vec3<f32>(voxel) + max(rayDirSign, vec3<f32>(0.0));
    var tVoxel: vec3<f32> = (nextBoundary + brickMin - rayOrigin) * rayDirInv;
    
    // Ensure we don't step backwards
    tVoxel = max(tVoxel, vec3<f32>(tEnter));
    
    let deltaVoxel: vec3<f32> = abs(rayDirInv);

    // Voxel DDA loop
    for (var i: i32 = 0; i < 24; i = i + 1) // Max ~24 steps through 8x8x8
    {
        // Bounds check
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= 8 || voxel.y >= 8 || voxel.z >= 8)
        {
            break;
        }

        // Check occupancy
        if (isVoxelSet(brickPool[brickSlot], u32(voxel.x), u32(voxel.y), u32(voxel.z)))
        {
            if (params.hasColor == 0u)
            {
                (*color) = vec3<f32>(1.0, 1.0, 1.0);
            }
            else
            {
                let voxelIndex: u32 = localCoordToVoxelIndex(vec3<u32>(voxel));
                (*color) = loadColor(brickSlot, voxelIndex);
            }
            return true;
        }

        // Find smallest t and advance
        let tNext: f32 = min(min(tVoxel.x, tVoxel.y), tVoxel.z);
        
        if (tNext > tExit)
        {
            break;
        }

        // Advance along the axis with smallest t
        if (tVoxel.x <= tVoxel.y && tVoxel.x <= tVoxel.z)
        {
            voxel.x = voxel.x + stepVoxel.x;
            tVoxel.x = tVoxel.x + deltaVoxel.x;
        }
        else if (tVoxel.y <= tVoxel.z)
        {
            voxel.y = voxel.y + stepVoxel.y;
            tVoxel.y = tVoxel.y + deltaVoxel.y;
        }
        else
        {
            voxel.z = voxel.z + stepVoxel.z;
            tVoxel.z = tVoxel.z + deltaVoxel.z;
        }
    }

    return false;
}