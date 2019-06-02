#include <iostream>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_in_mesh.h>
#include <igl/Hit.h>
#include <igl/rotate_by_quat.h>


int pick_face(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
int pick_vertex(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

Eigen::Vector3f computePosition(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y, Eigen::RowVector3d pt3D);
Eigen::Vector3f computeTranslation(igl::opengl::glfw::Viewer& viewer,int mouse_x,int from_x,int mouse_y,int from_y,Eigen::RowVector3d pt3D);
Eigen::Vector4f computeRotation(igl::opengl::glfw::Viewer& viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);

void update_display(igl::opengl::glfw::Viewer& viewer,const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
