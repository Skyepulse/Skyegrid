struct ComputeVoxelParams 
{
    voxelDimensions: vec3<u32>,
    _pad: u32,
};

//  2D output texture
@group(0) @binding(0) var targetTexture: texture_storage_2d<rgba8unorm, write>;
//  Voxel data buffer as texels
@group(0) @binding(1) var voxelImage: texture_storage_3d<rgba32uint, read>;
//  Voxel parameters uniform buffer
@group(0) @binding(2) var<uniform> voxelParams: ComputeVoxelParams;

@compute @workgroup_size(8, 8, 1)
fn c(@builtin(global_invocation_id) id: vec3<u32>) 
{
    // do stuff
}