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
// Brick grid
@group(0) @binding(2)
var<storage, read_write> brickGrid: array<atomic<u32>>; // atomic write in order to see if already requested by GPU
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

//================================//
//           HELPERS              //
//================================//
// Bit layout (example):
// [31]     : resident flag
// [30]     : requested flag (GPU feedback)
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
fn readVoxelData(worldPos: vec3<f32>) -> bool
{
    let voxelCoord: vec3<u32> = worldToVoxelCoord(worldPos);
    let brickCoord: vec3<u32> = voxelToBrickCoord(voxelCoord);
    let brickIndex: u32 = brickToIndex(brickCoord);
    let localCoord: vec3<u32> = voxelToLocalCoord(voxelCoord);
    let pointer = atomicLoad(&brickGrid[brickIndex]);

    // Check if the pointer, points to a valid brick
    if isBrickLoaded(pointer) == false
    {
        return false;
    }

    // The adress in the pointer OR LOD COLOR but it is loaded so we know
    // it is a brick index, so it is in [23:0]
    let brickSlot: u32 = pointer & 0x00FFFFFFu;
    let brick: Brick = brickPool[brickSlot];
    return isVoxelSet(brick, localCoord.x, localCoord.y, localCoord.z);
}

//================================//
// We call this in case we get an unloaded brick
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
fn readColorFromPool(globalOffset: u32) -> u32
{
    let bufferIdx   = globalOffset / params.maxColorBufferSize;
    let localOffset = globalOffset % params.maxColorBufferSize;

    switch (bufferIdx)
    {
        case 0u: { return colorPool[localOffset]; }
        case 1u: { return colorPool2[localOffset]; }
        case 2u: { return colorPool3[localOffset]; }
        default: { return 0u; } // out of bounds, should never happen
    }
}

//================================//
fn loadColor(brickSlot: u32, voxelIndex: u32) -> vec3<f32>
{
    let globalOffset: u32 = brickSlot * 512u + voxelIndex;  
    let packedColor: u32 = readColorFromPool(globalOffset);

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
fn worldToBrickIndex(position: vec3<f32>) -> u32
{
    let voxelCoord: vec3<u32> = worldToVoxelCoord(position);
    let brickCoord: vec3<u32> = voxelToBrickCoord(voxelCoord);
    return brickToIndex(brickCoord);
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
fn brickSlotFromPointer(pointer: u32) -> u32
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
    let hit = traverseGrid(rayOrigin, rayDir, &color, &steps);
    //let hit = traverseDEBUG(rayOrigin, rayDir, &color, &steps);

    textureStore(
        targetTexture,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(color, 1.0)
    );
}

//================================//
fn traverseGrid(rayOrigin: vec3<f32>, rayDir: vec3<f32>, color: ptr<function, vec3<f32>>, steps: ptr<function, i32>) -> bool
{
    // Ray sign and inverse
    let rayDirInv:          vec3<f32>   = 1.0 / rayDir;
    let signDir:            vec3<f32>   = sign(rayDirInv);
    let clampedRayDirInv:   vec3<f32>   = clamp(rayDirInv, vec3<f32>(-1e8), vec3<f32>(1e8)); // avoid infinities

    let stepBrick:          vec3<i32>   = vec3<i32>(signDir);
    let brickSize:          f32         = 8.0;

    // Try to find entry and out of slab
    let t1:     vec3<f32>   = -rayOrigin * rayDirInv;
    let t2:     vec3<f32>   = (vec3<f32>(f32(params.voxelResolution)) - rayOrigin) * rayDirInv;

    let tmin:   vec3<f32>   = min(t1, t2);
    let tmax:   vec3<f32>   = max(t1, t2);
    var tNear:  f32         = 0.0;
    var tFar:   f32         = 1e30;

    for (var i: i32 = 0; i < 3; i = i + 1) 
    {
        tNear = max(tNear, tmin[i]);
        tFar = min(tFar, tmax[i]);
    }

    // Return early if we are sure to miss the slab
    if (tNear > tFar) 
    {
        return false;
    }

    // FIX: Add small epsilon to ensure we're inside the grid
    let epsilon: f32 = 0.0001;
    tNear = tNear + epsilon;

    // start at entry point t = tNear
    let entryPoint:         vec3<f32> = rayOrigin + rayDir * tNear;
    var brickCoord:         vec3<i32> = vec3<i32>(floor(entryPoint / brickSize)); // which brick we are in

    // Clamp brick coord to valid range
    brickCoord = clamp(brickCoord, vec3<i32>(0), vec3<i32>(i32(params.voxelResolution) / 8 - 1));

    // (Brickcoord + 0.5 * (signDir + 1))) * 8 gives us the coordinate of the next brick boundary in the ray direction
    let nextBrickBoundary:  vec3<f32> = (vec3<f32>(brickCoord) + 0.5 * (signDir + 1.0)) * brickSize;

    var tBrick:             vec3<f32> = (nextBrickBoundary - rayOrigin) * rayDirInv; // parametric distance to next brick boundary
    let deltaBrick:         vec3<f32> = abs(clampedRayDirInv) * brickSize;
    let brickResolution:    i32       = i32(params.voxelResolution) / 8;

    // INIT LOOP
    while(true)
    {
        (*steps) = (*steps) + 1;

        // Check if we are outside the grid
        if (brickCoord.x < 0 || brickCoord.y < 0 || brickCoord.z < 0 ||
            brickCoord.x >= brickResolution ||
            brickCoord.y >= brickResolution ||
            brickCoord.z >= brickResolution)
        {
            break; // outside grid, return without setting color
        }

        let brickIndex:     u32 = brickToIndex(vec3<u32>(brickCoord));
        let brickPointer:   u32 = atomicLoad(&brickGrid[brickIndex]);
        let brickSlot:      u32 = brickSlotFromPointer(brickPointer);

        let brickLoaded:    bool = isBrickLoaded(brickPointer);
        let brickUnloaded:  bool = isBrickUnloaded(brickPointer);
        let brickEmpty:     bool = (!brickLoaded && !brickUnloaded);

        if (!brickEmpty) // Not empty
        {
            if (brickUnloaded) // easy, load LOD color
            {
                (*color) = loadLODColorBrickIndex(brickIndex);
                writeFeedback(brickIndex);
                return true;
            }
            else if (brickLoaded) // full brick, do voxel traversal
            {
                let brickMin: vec3<f32> = vec3<f32>(brickCoord) * brickSize;
                let brickMax: vec3<f32> = brickMin + vec3<f32>(brickSize);

                // local slab
                let t1_local: vec3<f32> = (brickMin - rayOrigin) * clampedRayDirInv;
                let t2_local: vec3<f32> = (brickMax - rayOrigin) * clampedRayDirInv;

                let tmin_local: vec3<f32> = min(t1_local, t2_local);
                let tmax_local: vec3<f32> = max(t1_local, t2_local);

                let tEnter = max(max(tmin_local.x, tmin_local.y), tmin_local.z);
                let tExit  = min(min(tmax_local.x, tmax_local.y), tmax_local.z);

                if (tEnter <= tExit)
                {
                    let hit = traverseBrick(rayOrigin, rayDir, tEnter, tExit, brickCoord, brickPointer, brickSlot, color);
                    if(hit) // if not we just continue
                    {
                        return true;
                    }
                }
            }
        }

        // Advance
        if (tBrick.x < tBrick.y) 
        {
            if (tBrick.x < tBrick.z) 
            {
                brickCoord.x    = brickCoord.x + stepBrick.x;
                tBrick.x        = tBrick.x + deltaBrick.x;
            } 
            else 
            {
                brickCoord.z    = brickCoord.z + stepBrick.z;
                tBrick.z        = tBrick.z + deltaBrick.z;
            }
        } 
        else 
        {
            if (tBrick.y < tBrick.z) 
            {
                brickCoord.y    = brickCoord.y + stepBrick.y;
                tBrick.y        = tBrick.y + deltaBrick.y;
            } 
            else 
            {
                brickCoord.z    = brickCoord.z + stepBrick.z;
                tBrick.z        = tBrick.z + deltaBrick.z;
            }
        }
    }

    return false;
}

//================================//
fn traverseBrick(rayOrigin: vec3<f32>, rayDir: vec3<f32>, tEnter: f32, tExit: f32, brickCoord: vec3<i32>, brickPointer: u32, brickSlot: u32, color: ptr<function, vec3<f32>>) -> bool
{
    let brickSize:          f32         = 8.0;
    let rayDirInv:          vec3<f32>   = 1.0 / rayDir;
    let signDir:            vec3<f32>   = sign(rayDirInv);
    let clampedRayDirInv:   vec3<f32>   = clamp(rayDirInv, vec3<f32>(-1e8), vec3<f32>(1e8)); // avoid infinities

    let brickMin:           vec3<f32>   = vec3<f32>(brickCoord) * brickSize;

    let entryPoint:         vec3<f32>   = rayOrigin + rayDir * tEnter;

    var voxel:              vec3<i32>   = vec3<i32>(clamp(floor(entryPoint - brickMin), vec3<f32>(0.0), vec3<f32>(7.0))); // local voxel coord in brick, known to be in [0, 7]^3
    let nextVoxelBoundary:  vec3<f32>   = (vec3<f32>(voxel) + 0.5 * (signDir + 1.0)); // next voxel boundary in ray direction

    var tVoxel:             vec3<f32>   = (nextVoxelBoundary + vec3<f32>(brickMin) - entryPoint) * rayDirInv;
    let deltaVoxel:         vec3<f32>   = abs(clampedRayDirInv);

    var tCurrent:           f32         = tEnter;  

    // DDA loop
    while(true)
    {
        if (voxel.x >= 8 || voxel.y >= 8 || voxel.z >= 8
            || voxel.x < 0 || voxel.y < 0 || voxel.z < 0)
        {
            break;
        }

        let voxelIndex: u32 = localCoordToVoxelIndex(vec3<u32>(voxel));

        // is it occupied:
        let hit = isVoxelSet(brickPool[brickSlot], vec3<u32>(voxel).x, vec3<u32>(voxel).y, vec3<u32>(voxel).z);
        if (hit)
        {
            // load color

            if (params.hasColor == 0u)
            {
                (*color) = vec3<f32>(1.0, 1.0, 1.0);
            }
            else
            {
                (*color) = loadColor(brickSlot, voxelIndex);
            }
            return true;
        }

        // Advance
        if (tVoxel.x < tVoxel.y) 
        {
            if (tVoxel.x < tVoxel.z) 
            {
                voxel.x     = voxel.x + i32(signDir.x);
                tCurrent    = tVoxel.x;
                tVoxel.x    = tVoxel.x + deltaVoxel.x;
            } 
            else 
            {
                voxel.z     = voxel.z + i32(signDir.z);
                tCurrent    = tVoxel.z;
                tVoxel.z    = tVoxel.z + deltaVoxel.z;
            }
        } 
        else 
        {
            if (tVoxel.y < tVoxel.z) 
            {
                voxel.y     = voxel.y + i32(signDir.y);
                tCurrent    = tVoxel.y;
                tVoxel.y    = tVoxel.y + deltaVoxel.y;
            } 
            else 
            {
                voxel.z     = voxel.z + i32(signDir.z);
                tCurrent    = tVoxel.z;
                tVoxel.z    = tVoxel.z + deltaVoxel.z;
            }
        }

        if (tCurrent > tExit)
        {
            break;
        }
    }

    return false;
}

//================================//
fn traverseDEBUG(rayOrigin: vec3<f32>, rayDir: vec3<f32>, color: ptr<function, vec3<f32>>, steps: ptr<function, i32>) -> bool
{
    // Ray sign and inverse
    let rayDirInv: vec3<f32> = 1.0 / rayDir;
    let signDir: vec3<f32> = sign(rayDirInv);
    let clampedRayDirInv: vec3<f32> = clamp(rayDirInv, vec3<f32>(-1e8), vec3<f32>(1e8));

    // Try to find entry and out of slab
    let t1: vec3<f32> = -rayOrigin * rayDirInv;
    let t2: vec3<f32> = (vec3<f32>(f32(params.voxelResolution)) - rayOrigin) * rayDirInv;

    let tmin: vec3<f32> = min(t1, t2);
    let tmax: vec3<f32> = max(t1, t2);

    var tNear: f32 = 0.0;
    var tFar: f32 = 1e30;
    for (var i: i32 = 0; i < 3; i = i + 1) 
    {
        tNear = max(tNear, tmin[i]);
        tFar = min(tFar, tmax[i]);
    }

    // Meaning we don't hit the slab
    if (tNear > tFar) 
    {
        return false;
    }

    // Find the axis we hit the slab from AABB
    // 0 -> X, 1 -> Y, 2 -> Z
    var slabAxis: i32 = 0;
    if (tmin.x < tmin.y) 
    {
        if (tmin.x < tmin.z) 
        {
            slabAxis = 0;
        } 
        else 
        {
            slabAxis = 2;
        }
    }
    else 
    {
        if (tmin.y < tmin.z) 
        {
            slabAxis = 1;
        } 
        else 
        {
            slabAxis = 2;
        }
    }

    // start at entry point t = tNear
    let entryPoint: vec3<f32> = rayOrigin + rayDir * tNear;
    var voxelCoord: vec3<u32> = vec3<u32>(clamp(floor(entryPoint), vec3<f32>(0.0), vec3<f32>(f32(params.voxelResolution) - 1.0)));

    // (coord + 0.5 * (signDir + 1))) gives us the coordinate of the next voxel boundary in the ray direction
    let nextVoxelBoundary: vec3<f32> = (vec3<f32>(voxelCoord) + 0.5 * (signDir + 1.0));
    var t: vec3<f32> = (nextVoxelBoundary - entryPoint) * rayDirInv; // parametric distance to next voxel boundary
    let step: vec3<i32> = vec3<i32>(signDir);
    let delta: vec3<f32> = rayDirInv * signDir;

    var mask: vec3<f32> = vec3<f32>(0.0);
    mask[slabAxis] = 1.0;

    // t[mask] holds the exact distance we hit the voxel boundary
    // We subtract delta[mask] to get the entry point into the voxel
    var t_hit: f32 = dot(t, mask) - dot(delta, mask);

    let denom = f32(params.voxelResolution - 1u);
    (*color) = vec3<f32>(
        f32(voxelCoord.x),
        f32(voxelCoord.y),
        f32(voxelCoord.z)
    ) / denom;
    return true;
}