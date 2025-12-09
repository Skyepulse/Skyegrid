//================================//
struct ComputeVoxelParams {
    pixelToRay: mat4x4<f32>,
    cameraOrigin: vec3<f32>,
    _pad0: f32,
    voxelResolution: u32,
    time: f32,
    _pad1: vec2<f32>,
};

//================================//
@group(0) @binding(0)
var targetTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) 
var VoxelImage: texture_storage_3d<rgba32uint, read>;
@group(0) @binding(2)
var<uniform> params: ComputeVoxelParams;

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
    var voxelCoord: vec3<u32> = worldToVoxelCoord(entryPoint);

    // (coord + 0.5 * (signDir + 1))) gives us the coordinate of the next voxel boundary in the ray direction
    let nextVoxelBoundary: vec3<f32> = (vec3<f32>(voxelCoord) + 0.5 * (signDir + 1.0));
    var t: vec3<f32> = (nextVoxelBoundary - entryPoint) * rayDirInv; // parametric distance to next voxel boundary
    let step: vec3<i32> = vec3<i32>(signDir);
    let delta: vec3<f32> = rayDirInv * signDir;

    // First hit
    var texelCoord: vec3<i32> = coordToTexelCoord(voxelCoord);
    var voxelData: vec4<u32> = textureLoad(VoxelImage, texelCoord);
    var hit: bool = readVoxelData(voxelData, voxelCoord);
    if (!hit)
    {
        // Traversal
        while (true)
        {
            // We find the smalles t amongst t.x, ty, tz
            // We step along that axis
            // We increment voxel coord by step[axis]
            // We increment t[axis] by delta[axis]
            // We always record the stepped into axis

            (*steps) = (*steps) + 1;

            if (t.x < t.y)
            {
                if(t.x < t.z)
                {
                    // Step X
                    voxelCoord.x = u32(i32(voxelCoord.x) + step.x);
                    t.x = t.x + delta.x;
                    slabAxis = 0;
                }
                else
                {
                    // Step Z
                    voxelCoord.z = u32(i32(voxelCoord.z) + step.z);
                    t.z = t.z + delta.z;
                    slabAxis = 2;
                }
            }
            else
            {
                if(t.y < t.z)
                {
                    // Step Y
                    voxelCoord.y = u32(i32(voxelCoord.y) + step.y);
                    t.y = t.y + delta.y;
                    slabAxis = 1;
                }
                else
                {
                    // Step Z
                    voxelCoord.z = u32(i32(voxelCoord.z) + step.z);
                    t.z = t.z + delta.z;
                    slabAxis = 2;
                }
            }

            // If coord is out of bounds on any axis, we exit
            if (voxelCoord.x >= params.voxelResolution ||
                voxelCoord.y >= params.voxelResolution ||
                voxelCoord.z >= params.voxelResolution) 
            {
                return false;
            }

            // Read voxel data
            texelCoord = coordToTexelCoord(voxelCoord);
            voxelData = textureLoad(VoxelImage, texelCoord);
            hit = readVoxelData(voxelData, voxelCoord);

            if (hit)
            {
                break;
            }
        }
    }

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

//================================//
fn worldToVoxelCoord(position: vec3<f32>) -> vec3<u32> 
{
    return vec3<u32>(clamp(floor(position), vec3<f32>(0.0), vec3<f32>(f32(params.voxelResolution) - 1.0)));
}

//================================//
fn coordToTexelCoord(voxelCoord: vec3<u32>) -> vec3<i32> 
{
    return vec3<i32>(
        i32(voxelCoord.x / 4),
        i32(voxelCoord.y / 4),
        i32(voxelCoord.z / 8)
    );
}

//================================//
fn readVoxelData(texel: vec4<u32>, coord: vec3<u32>) -> bool 
{
    return bool((texel[coord.x % 4u] >> ((coord.y % 4u) + (coord.z % 8u) * 4u)) & 1u);
}