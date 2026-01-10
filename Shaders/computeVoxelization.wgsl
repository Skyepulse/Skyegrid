// computeVoxelization.wgsl - Pass 1: Voxelize triangles

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

struct Vertex {
    position: vec3<f32>,
    _pad: f32,
    uv: vec2<f32>,
    _pad2: vec2<f32>,
    normal: vec3<f32>,
    _pad3: f32,
}

struct Triangle {
    indices: vec3<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var meshTexture: texture_2d<f32>;
@group(0) @binding(4) var meshSampler: sampler;
@group(0) @binding(5) var<storage, read_write> occupancy: array<atomic<u32>>; // 16 u32s per brick
@group(0) @binding(6) var<storage, read_write> denseColors: array<atomic<u32>>; // RGB packed per voxel

// Convert voxel coords to linear index
fn voxelToLinear(voxel: vec3<u32>) -> u32 {
    return voxel.x + voxel.y * uniforms.voxelResolution + voxel.z * uniforms.voxelResolution * uniforms.voxelResolution;
}

// Convert voxel coords to brick index and local index within brick
fn voxelToBrickIndices(voxel: vec3<u32>) -> vec2<u32> {
    let brickCoord = voxel / 8u;
    let localCoord = voxel % 8u;

    let globalBrickIndex = brickCoord.x + brickCoord.y * uniforms.brickResolution + brickCoord.z * uniforms.brickResolution * uniforms.brickResolution;
    if (globalBrickIndex < uniforms.brickStart || globalBrickIndex >= uniforms.brickEnd) {
        // Out of range, return invalid indices
        return vec2<u32>(0xFFFFFFFFu, 0u);
    }
    
    let localBrickIndex = globalBrickIndex - uniforms.brickStart;
    let localVoxelIndex = localCoord.x + localCoord.y * 8u + localCoord.z * 64u; // 0-511
    
    return vec2<u32>(localBrickIndex, localVoxelIndex);
}

// Set occupancy bit atomically
fn setOccupancy(brickIndex: u32, localIndex: u32) {
    // 16 u32s per brick, each u32 has 32 bits
    let wordIndex = brickIndex * 16u + (localIndex / 32u);
    let bitIndex = localIndex % 32u;
    atomicOr(&occupancy[wordIndex], 1u << bitIndex);
}

// Pack RGB color
fn packColor(r: u32, g: u32, b: u32) -> u32 {
    return (r & 0xFFu) | ((g & 0xFFu) << 8u) | ((b & 0xFFu) << 16u);
}

// Triangle-AABB intersection test
fn triangleAABBIntersect(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, boxCenter: vec3<f32>, boxHalfSize: vec3<f32>) -> bool {
    // Translate triangle to origin
    let t0 = v0 - boxCenter;
    let t1 = v1 - boxCenter;
    let t2 = v2 - boxCenter;
    
    // Test AABB axes
    let minP = min(min(t0, t1), t2);
    let maxP = max(max(t0, t1), t2);
    
    if (maxP.x < -boxHalfSize.x || minP.x > boxHalfSize.x) { return false; }
    if (maxP.y < -boxHalfSize.y || minP.y > boxHalfSize.y) { return false; }
    if (maxP.z < -boxHalfSize.z || minP.z > boxHalfSize.z) { return false; }
    
    // Triangle normal test
    let edge0 = t1 - t0;
    let edge1 = t2 - t1;
    let normal = cross(edge0, edge1);
    let d = dot(normal, t0);
    let r = boxHalfSize.x * abs(normal.x) + boxHalfSize.y * abs(normal.y) + boxHalfSize.z * abs(normal.z);
    if (abs(d) > r) { return false; }
    
    // Edge cross-product tests (9 tests)
    let edge2 = t0 - t2;
    let axes = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let edges = array<vec3<f32>, 3>(edge0, edge1, edge2);
    
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            let axis = cross(axes[i], edges[j]);
            let len2 = dot(axis, axis);
            if (len2 < 1e-10) { continue; }
            
            let p0 = dot(axis, t0);
            let p1 = dot(axis, t1);
            let p2 = dot(axis, t2);
            let minProj = min(min(p0, p1), p2);
            let maxProj = max(max(p0, p1), p2);
            let rr = boxHalfSize.x * abs(axis.x) + boxHalfSize.y * abs(axis.y) + boxHalfSize.z * abs(axis.z);
            if (minProj > rr || maxProj < -rr) { return false; }
        }
    }
    
    return true;
}

// Compute barycentric coordinates for a point projected onto triangle
fn barycentric(p: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let e2 = p - v0;
    
    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d11 = dot(e1, e1);
    let d20 = dot(e2, e0);
    let d21 = dot(e2, e1);
    
    let denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-10) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    
    return vec3<f32>(u, v, w);
}

@compute @workgroup_size(64)
fn c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let localBrickIndex = gid.x;
    let bricksThisPass = uniforms.brickEnd - uniforms.brickStart;
    if (localBrickIndex >= bricksThisPass) {
        return;
    }

    // Convert local brick index to global brick coordinates
    let globalBrickIndex = uniforms.brickStart + localBrickIndex;

    let bricksPerRow = uniforms.brickResolution;
    let brickZ = globalBrickIndex / (bricksPerRow * bricksPerRow);
    let brickY = (globalBrickIndex / bricksPerRow) % bricksPerRow;
    let brickX = globalBrickIndex % bricksPerRow;

    // Brick bounds in voxel space
    let brickVoxelBase = vec3<u32>(brickX, brickY, brickZ) * 8u;
    let brickVoxelMax  = brickVoxelBase + vec3<u32>(7u);

    // Brick bounds in world space
    let brickMinWorld =
        uniforms.meshMinBounds +
        vec3<f32>(brickVoxelBase) * uniforms.voxelSize;

    let brickMaxWorld =
        uniforms.meshMinBounds +
        (vec3<f32>(brickVoxelMax) + 1.0) * uniforms.voxelSize;

    let halfVoxel = uniforms.voxelSize * 0.5;

    // Iterate triangles
    for (var triIndex = 0u; triIndex < uniforms.numTriangles; triIndex++) {
        let tri = triangles[triIndex];

        let v0 = vertices[tri.indices.x];
        let v1 = vertices[tri.indices.y];
        let v2 = vertices[tri.indices.z];

        let p0 = v0.position;
        let p1 = v1.position;
        let p2 = v2.position;

        // Triangle AABB in world space
        let triMin = min(min(p0, p1), p2);
        let triMax = max(max(p0, p1), p2);

        // Early reject: triangle does not touch this brick
        if (triMax.x < brickMinWorld.x || triMin.x > brickMaxWorld.x ||
            triMax.y < brickMinWorld.y || triMin.y > brickMaxWorld.y ||
            triMax.z < brickMinWorld.z || triMin.z > brickMaxWorld.z) {
            continue;
        }

        // Iterate voxels inside this brick only
        for (var z = 0u; z < 8u; z++) {
            for (var y = 0u; y < 8u; y++) {
                for (var x = 0u; x < 8u; x++) {
                    let voxel = brickVoxelBase + vec3<u32>(x, y, z);
                    let voxelCenter =
                        uniforms.meshMinBounds +
                        (vec3<f32>(voxel) + 0.5) * uniforms.voxelSize;

                    if (!triangleAABBIntersect(
                            p0, p1, p2,
                            voxelCenter,
                            vec3<f32>(halfVoxel))) {
                        continue;
                    }

                    let localVoxelIndex = x + y * 8u + z * 64u;

                    // Occupancy
                    let wordIndex =
                        localBrickIndex * 16u + (localVoxelIndex / 32u);
                    let bitIndex = localVoxelIndex % 32u;
                    atomicOr(&occupancy[wordIndex], 1u << bitIndex);

                    // Sample color
                    let bary = barycentric(voxelCenter, p0, p1, p2);
                    let uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
                    let texColor =
                        textureSampleLevel(meshTexture, meshSampler, uv, 0.0);

                    let r = u32(clamp(texColor.r * 255.0, 0.0, 255.0));
                    let g = u32(clamp(texColor.g * 255.0, 0.0, 255.0));
                    let b = u32(clamp(texColor.b * 255.0, 0.0, 255.0));

                    let localDenseIdx =
                        localBrickIndex * 512u + localVoxelIndex;
                    atomicStore(
                        &denseColors[localDenseIdx],
                        packColor(r, g, b));
                }
            }
        }
    }
}
