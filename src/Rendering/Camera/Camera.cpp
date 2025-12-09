#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include "../../../includes/Rendering/Camera/Camera.hpp"

//================================//
void Camera::LookAtPoint(const Eigen::Vector3f& point, const Eigen::Vector3f& up)
{
    Eigen::Vector3f dir = (point - this->position).normalized();
    this->LookAtDirection(dir, up);
}

//================================//
void Camera::LookAtDirection(const Eigen::Vector3f& direction, const Eigen::Vector3f& up)
{
    Eigen::Vector3f f = direction.normalized();              // +Z forward
    Eigen::Vector3f r = up.cross(f).normalized();             // +X right
    Eigen::Vector3f u = f.cross(r);                            // +Y up (recomputed)

    Eigen::Matrix3f R;
    R.col(0) = r;
    R.col(1) = u;
    R.col(2) = f;

    this->orientation = Eigen::Quaternionf(R);
    this->orientation.normalize();
}

//================================//
Eigen::Matrix4f Camera::RotationMatrix() const
{
    Eigen::Matrix3f R = this->orientation.toRotationMatrix();
    Eigen::Matrix4f rotation;
    rotation << R(0,0), R(0,1), R(0,2), 0,
                R(1,0), R(1,1), R(1,2), 0,
                R(2,0), R(2,1), R(2,2), 0,
                     0,      0,      0, 1;
    return rotation;
}


//================================//
Eigen::Matrix4f Camera::TranslationMatrix() const
{
    Eigen::Matrix4f translation;
    translation << 1, 0, 0, this->position.x(),
                   0, 1, 0, this->position.y(),
                   0, 0, 1, this->position.z(),
                   0, 0, 0,        1;
    return translation;
}

//================================//
void Camera::SetRotation(const Eigen::Vector3f& eulerAngles)
{
    Eigen::AngleAxisf rx(eulerAngles.x(), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf ry(eulerAngles.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rz(eulerAngles.z(), Eigen::Vector3f::UnitZ());

    // Z * Y * X (intrinsic rotations)
    orientation = rz * ry * rx;
    orientation.normalize();
}

//================================//
void Camera::Rotate(const Eigen::Vector3f& deltaRotation)
{
    Eigen::AngleAxisf rx(deltaRotation.x(), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf ry(deltaRotation.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rz(deltaRotation.z(), Eigen::Vector3f::UnitZ());

    Eigen::Quaternionf dq = rz * ry * rx;
    orientation = (dq * orientation).normalized();
}

//================================//
void Camera::Move(const Eigen::Vector3f& deltaPosition)
{
    // x is forward, y is up, z is right everytime depending on the orientation
    Eigen::Vector3f forward = (RotationMatrix() * Eigen::Vector4f(0, 0, 1, 0)).head<3>().normalized();
    Eigen::Vector3f right = (RotationMatrix() * Eigen::Vector4f(1, 0, 0, 0)).head<3>().normalized();
    Eigen::Vector3f up = (RotationMatrix() * Eigen::Vector4f(0, 1, 0, 0)).head<3>().normalized();

    // Move the camera in the direction of the delta position
    this->position += forward * deltaPosition.z() + right * deltaPosition.x() + up * deltaPosition.y();
}

//================================//
Eigen::Matrix4f Camera::PixelToRayMatrix() const
{
    float aspectRatio = extent.x() / extent.y();
    float fovRad = fov * static_cast<float>(M_PI) / 180.0f;
    float tanHalfFov = std::tan(fovRad / 2.0f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
    P(0,0) = aspectRatio * tanHalfFov;  // scale x
    P(1,1) = tanHalfFov;                // scale y
    P(2,2) = 1.0f;                      // z forward

    Eigen::Matrix4f R = RotationMatrix();
    return R * P;
}

//================================//
void Camera::ValidatePixelToRayMatrix()
{
    Eigen::Matrix4f M = PixelToRayMatrix();

    // Explicit center-pixel check
    Eigen::Vector4f centerNDC(0.0f, 0.0f, 1.0f, 0.0f);
    Eigen::Vector3f centerRay =
        (M * centerNDC).head<3>().normalized();

    // Compute from rotation and translation the forward direction of the camera
    Eigen::Vector3f forward = (RotationMatrix() * Eigen::Vector4f(0, 0, 1, 0)).head<3>().normalized();

    assert(centerRay.isApprox(forward, 1e-5) && "PixelToRayMatrix validation failed: center ray does not match forward direction");
}