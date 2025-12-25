//================================//
struct UploadEntry
{
    gpuBrickSlot: u32, // the allocated brick slot for the position in the brick pool, should be written inside brick grid at the position brickGridIndex
    occupancy: array<u32, 16>, // 8 x u64 = 16 x u32
    // for the colors, its r,g,b (uint8 each) per voxel, so 8x8x8 = 512 colors
    colors: array<u32, 512>,
};

//================================//
struct Params
{
    uploadCount: u32,
    maxColorBufferSize: u32,
};

//================================//
struct Brick
{
    occupancy: array<u32, 16>, // 8 x u64 = 16 x u32
};

@group(0) @binding(0) 
var<storage, read> uploadEntries: array<UploadEntry>;
@group(0) @binding(1)
var<uniform> params: Params;
@group(0) @binding(2)
var<storage, read_write> brickPool: array<Brick>;

@group(0) @binding(3)
var<storage, read_write> colorPool1: array<u32>;
@group(0) @binding(4)
var<storage, read_write> colorPool2: array<u32>;
@group(0) @binding(5)
var<storage, read_write> colorPool3: array<u32>;

//================================//
fn writeColorToPool(globalOffset: u32, color: u32)
{
    let bufferIdx = globalOffset / params.maxColorBufferSize;
    let localOffset = globalOffset % params.maxColorBufferSize;
    
    switch (bufferIdx)
    {
        case 0u: { colorPool1[localOffset] = color; }
        case 1u: { colorPool2[localOffset] = color; }
        case 2u: { colorPool3[localOffset] = color; }
        default: { /* Out of bounds */ } // SHOULD NOT HAPPEN
    }
}

//================================//
fn writeBrick(brickSlot: u32, entry: UploadEntry)
{
    // Write occupancy
    for (var i: u32 = 0u; i < 16u; i = i + 1u)
    {
        brickPool[brickSlot].occupancy[i] = entry.occupancy[i];
    }

    // Write colors
    let globalStart = brickSlot * 512u;
    for (var i: u32 = 0u; i < 512u; i = i + 1u)
    {
        let globalOffset = globalStart + i;
        writeColorToPool(globalOffset, entry.colors[i]);
    }
}

//================================//
@compute @workgroup_size(128, 1, 1)
fn c(@builtin(global_invocation_id) gid: vec3<u32>) 
{
    let uploadIndex = gid.x;
    
    // Bounds check using your uniform
    if (uploadIndex >= params.uploadCount) 
    { 
        return; 
    }
    
    let entry = uploadEntries[uploadIndex];
    writeBrick(entry.gpuBrickSlot, entry);
}