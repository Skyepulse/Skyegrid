struct ComputeVoxelParams 
{
    voxelDimensions: vec3<u32>,
    time: f32,
};

//  2D output texture
@group(0) @binding(0) var targetTexture: texture_storage_2d<rgba8unorm, write>;
//  Voxel data buffer as texels
@group(0) @binding(1) var voxelImage: texture_storage_3d<rgba32uint, read>;
//  Voxel parameters uniform buffer
@group(0) @binding(2) var<uniform> voxelParams: ComputeVoxelParams;

@compute @workgroup_size(8, 8, 1)
fn c(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
    )
{
    let t = voxelParams.time;

    let r = 0.5 + 0.5 * sin(t + f32(wid.x));
    let g = 0.5 + 0.5 * sin(t + f32(wid.y));
    let b = 0.5 + 0.5 * sin(t + f32(wid.x + wid.y));

    textureStore(
        targetTexture,
        vec2<i32>(gid.xy),
        vec4<f32>(r, g, b, 1.0)
    );
}