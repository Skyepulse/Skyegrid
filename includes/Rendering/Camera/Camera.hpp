#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

//================================//
class Camera
{
public:
    Camera(Eigen::Vector2f extent, float fov = 20.0f, Eigen::Vector3f position = Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f rotation = Eigen::Vector3f(0.0f, 0.0f, 0.0f))
        : extent(extent), fov(fov), position(position) 
        {
            SetRotation(rotation);
        };

    Camera() = default;
    ~Camera() = default;

    Eigen::Vector2f GetExtent() const { return this->extent; }
    void SetExtent(const Eigen::Vector2f& extent) { this->extent = extent; }

    float GetFov() const { return this->fov; }
    void SetFov(float fov) { this->fov = fov; }

    Eigen::Vector3f GetPosition() const { return this->position; }
    void SetPosition(const Eigen::Vector3f& position) { this->position = position; }

    // Methods
    void LookAtDirection(const Eigen::Vector3f& direction, const Eigen::Vector3f& up = Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    void LookAtPoint(const Eigen::Vector3f& point, const Eigen::Vector3f& up = Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    Eigen::Matrix4f RotationMatrix() const;
    Eigen::Matrix4f TranslationMatrix() const;

    // Modify the Camera
    void SetRotation(const Eigen::Vector3f& eulerAngles);
    void Rotate(const Eigen::Vector3f& deltaRotation);
    void Move(const Eigen::Vector3f& deltaPosition);

    Eigen::Matrix4f PixelToRayMatrix() const;
    void ValidatePixelToRayMatrix();

private:

    Eigen::Vector2f extent;
    float fov = 45.0f;
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
};

#endif // CAMERA_HPP