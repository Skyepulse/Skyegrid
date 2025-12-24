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
};

//================================//
struct Brick
{
    occupancy: array<u32, 16>, // 8 x u64 = 16 x u32
};

@group(0) @binding(0) 
var<storage, read> uploadEntries: array<UploadEntry>;
@group(0) @binding(1)
var<storage, read_write> brickPool: array<Brick>;
@group(0) @binding(2)
var<storage, read_write> colorPool: array<u32>;
@group(0) @binding(3)
var<uniform> params: Params;

//================================//
fn writeBrick(brickSlot: u32, entry: UploadEntry)
{
    // Write occupancy
    for (var i: u32 = 0u; i < 16u; i = i + 1u)
    {
        brickPool[brickSlot].occupancy[i] = entry.occupancy[i];
    }

    // Write colors
    let start = brickSlot * 512u; // We know each brick in the color pool is represented by 512u
    for (var i: u32 = 0u; i < 512u; i = i + 1u)
    {
        colorPool[start + i] = entry.colors[i]; // already packed
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