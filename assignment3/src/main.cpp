#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>


/*** insert any necessary libigl headers here ***/
#include <igl/dijkstra.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

// vertex array, #V x3
Eigen::MatrixXd V;

// face array, #F x3
Eigen::MatrixXi F;

// UV coordinates, #V x2
Eigen::MatrixXd UV;

bool showingUV = false;
bool freeBoundary = false;
double TextureResolution = 10;
igl::opengl::ViewerCore temp3D;
igl::opengl::ViewerCore temp2D;
VectorXd x;
int showAngleDistortion = 0;
bool Visualize=0;
Eigen::MatrixXd Colors;

double Colormin = 0;
double Colormax = 150.0/255.0;


void Redraw()
{
	viewer.data().clear();

	if (!showingUV)
	{
		viewer.data().set_mesh(V, F);
		viewer.data().set_face_based(false);

		if (UV.size() != 0)
		{
			viewer.data().set_uv(TextureResolution*UV);
			viewer.data().show_texture = true;
			if(Visualize)
				viewer.data().set_colors(Colors);

		}
	}
	else
	{
		viewer.data().show_texture = false;
		viewer.data().set_mesh(UV, F);
		if (Visualize)
			viewer.data().set_colors(Colors);
	}
}

bool callback_mouse_move(Viewer &viewer, int mouse_x, int mouse_y)
{
	if (showingUV)
		viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Translation;
	return false;
}

static void computeSurfaceGradientMatrix(SparseMatrix<double> & D1, SparseMatrix<double> & D2)
{
	MatrixXd F1, F2, F3;
	SparseMatrix<double> DD, Dx, Dy, Dz;

	igl::local_basis(V, F, F1, F2, F3);
	igl::grad(V, F, DD);

	Dx = DD.topLeftCorner(F.rows(), V.rows());
	Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
	Dz = DD.bottomRightCorner(F.rows(), V.rows());

	D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
	D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
}
static inline void SSVD2x2(const Eigen::Matrix2d& J, Eigen::Matrix2d& U, Eigen::Matrix2d& S, Eigen::Matrix2d& V)
{
	double e = (J(0) + J(3))*0.5;
	double f = (J(0) - J(3))*0.5;
	double g = (J(1) + J(2))*0.5;
	double h = (J(1) - J(2))*0.5;
	double q = sqrt((e*e) + (h*h));
	double r = sqrt((f*f) + (g*g));
	double a1 = atan2(g, f);
	double a2 = atan2(h, e);
	double rho = (a2 - a1)*0.5;
	double phi = (a2 + a1)*0.5;

	S(0) = q + r;
	S(1) = 0;
	S(2) = 0;
	S(3) = q - r;

	double c = cos(phi);
	double s = sin(phi);
	U(0) = c;
	U(1) = s;
	U(2) = -s;
	U(3) = c;

	c = cos(rho);
	s = sin(rho);
	V(0) = c;
	V(1) = -s;
	V(2) = s;
	V(3) = c;
}
void Visualization()
{

	Eigen::SparseMatrix<double> Dx;
	Eigen::SparseMatrix<double> Dy;

	Eigen::SparseMatrix<double> Jac;

	Eigen::SparseMatrix<double> UpLeft;
	Eigen::SparseMatrix<double> UpRight;
	Eigen::SparseMatrix<double> DownLeft;
	Eigen::SparseMatrix<double> DownRight;

	computeSurfaceGradientMatrix(Dx, Dy);

	Eigen::VectorXd uVec = UV.col(0);
	Eigen::VectorXd vVec = UV.col(1);
	Eigen::SparseMatrix<double> uVecTemp;
	Eigen::SparseMatrix<double> vVecTemp;
	uVecTemp = uVec.sparseView();
	vVecTemp = vVec.sparseView();

	UpLeft = (Dx * uVecTemp);
	UpRight = (Dy * uVecTemp);
	DownLeft = (Dx * vVecTemp);
	DownRight = (Dy * vVecTemp);

	Eigen::SparseMatrix<double> SMUp;
	Eigen::SparseMatrix<double> SMDown;

	igl::cat(2, UpLeft, UpRight, SMUp);
	igl::cat(2, DownLeft, DownRight, SMDown);
	igl::cat(1, SMUp, SMDown, Jac);

	int tailleV = V.rows();

	Eigen::MatrixXd distPerFace;
	distPerFace.resize(F.rows(), 1);
	distPerFace.setZero();


	for (int i = 0; i < F.rows(); i++)
	{
		double DxU = 0;
		double DxV = 0;
		double DyU = 0;
		double DyV = 0;

		for (int j = 0; j < 3; j++)
		{

			int indicePtcur = F.row(i)[j];

			DxU += Jac.coeff(indicePtcur, 0);
			DxV += Jac.coeff(tailleV + indicePtcur, 0);
			DyU += Jac.coeff(indicePtcur, 1);
			DyV += Jac.coeff(tailleV + indicePtcur, 1);
		}

		DxU /= 3;
		DxV /= 3;
		DyU /= 3;
		DyV /= 3;


		Eigen::Matrix2d J;
		J.setZero();

		J.row(0)[0] = DxU;
		J.row(0)[1] = DyU;
		J.row(1)[0] = DxV;
		J.row(1)[1] = DyV;

		Eigen::Matrix2d Utmp;
		Eigen::Matrix2d Stmp;
		Eigen::Matrix2d Vtmp;

		SSVD2x2(J, Utmp, Stmp, Vtmp);

		double sigmaUn = Stmp.row(0)[0];
		double sigmaDeux = Stmp.row(1)[1];


		if (showAngleDistortion < 0)
		{ 
			distPerFace.row(i)[0] = ((sigmaUn - sigmaDeux) * (sigmaUn - sigmaDeux));
		}
		else if (showAngleDistortion == 0)
		{ 
			distPerFace.row(i)[0] = ((sigmaUn - sigmaDeux) * (sigmaUn - sigmaDeux));
		}
		else if (showAngleDistortion == 1)
		{
			distPerFace.row(i)[0] = ((sigmaUn - 1) * (sigmaUn - 1) + (sigmaDeux - 1) * (sigmaDeux - 1));
		}
		else if (showAngleDistortion == 2)
		{ 
			distPerFace.row(i)[0] = ((sigmaUn * sigmaDeux - 1) * (sigmaUn * sigmaDeux - 1));
		}
		else {
			distPerFace.row(i)[0] = ((sigmaUn * sigmaDeux - 1) * (sigmaUn * sigmaDeux - 1));
		}
	}

	double maxCoef = distPerFace.maxCoeff();
	double minCoef = distPerFace.minCoeff();


	assert(Colors.rows() == distPerFace.rows());
	for (int i = 0; i < Colors.rows(); i++)
	{
		Colors(i, 0) = 1;
		if (distPerFace.row(i)[0] < Colormin)
		{
			Colors(i, 1) = 1;
			Colors(i, 2) = 1;

		}
		else if (distPerFace.row(i)[0] > Colormax)
		{
			Colors(i, 1) = 0;
			Colors(i, 2) = 0;

		}
		else
		{ 
			double teinte = 1 - ((distPerFace.row(i)[0] - Colormin) / ((double)(Colormax)));
			Colors(i, 1) = teinte;
			Colors(i, 2) = teinte;

		}
	}
}


void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should contain the positions in the correct order.
	d.resize(indices.rows() * 2);
	C.resize(indices.rows() * 2, V.rows() * 2);
	int sizeP = positions.rows(), sizeV = V.rows();
	for (int i = 0; i < indices.rows(); i++)
	{
		d(i) = positions(i, 0);
		d(i + sizeP) = positions(i , 1);
		C.insert(i, indices(i)) = 1;
		C.insert(i + sizeP, indices(i) + sizeV) = 1;
		//C.insert(i, 0) = positions(i, 0);
		//C.insert(i, 1) = positions(i, 1);
	}
}

void computeParameterization(int type)
{
	VectorXi fixed_UV_indices;
	VectorXi two_fixed_UV_indices;
	MatrixXd fixed_UV_positions;
	SparseMatrix<double> A;
	VectorXd b;

	Eigen::SparseMatrix<double> C;
	VectorXd d;
	// Find the indices of the boundary vertices of the mesh and put them in fixed_UV_indices
	if (!freeBoundary)
	{
		//std::vector<std::vector<int>> loop;
		igl::boundary_loop(F, fixed_UV_indices);
		/*
		int k = 0;
		fixed_UV_positions.resize(fixed_UV_positions.rows() + loop.at(0).size(), 3);
		fixed_UV_indices.resize(loop.at(0).size(), 1);
		for (int i = 0; i < loop.at(0).size(); i++)
		{
			fixed_UV_indices(i) = loop.at(0).at(i);
		}
		*/
		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

		// The boundary vertices should be fixed to positions on the unit disc. Find these position and
		// save them in the #V x 2 matrix fixed_UV_position.
	}
	else
	{


		std::vector<std::vector<int> > VV;
		std::vector<double> Weights;
		igl::adjacency_list(F, VV);
		Eigen::VectorXd MinDis;
		Eigen::VectorXi Prev;

		igl::boundary_loop(F, fixed_UV_indices);
		/*
		RowVector3d V1, V2;
		float MaxDistance = 0;
		int ind1 = -1, ind2 = -1;
		igl::boundary_loop(F, fixed_UV_indices);
		for (int i = 0; i < fixed_UV_indices.rows(); i++)
		{
			for (int c = 0; c < fixed_UV_indices.rows(); c++)
			{
				if (i != c)
				{
					float v1X = V.row(fixed_UV_indices(i))(0);
					float v1Y = V.row(fixed_UV_indices(i))(1);
					float v1Z = V.row(fixed_UV_indices(i))(2);

					float v2X = V.row(fixed_UV_indices(c))(0);
					float v2Y = V.row(fixed_UV_indices(c))(1);
					float v2Z = V.row(fixed_UV_indices(c))(2);
					float dis = sqrtf(pow(v1X - v2X, 2) + pow(v1Y - v2Y, 2) + pow(v1Z - v2Z, 2));
					if (dis > MaxDistance)
					{
						MaxDistance = dis;
						V1 = V.row(fixed_UV_indices(i));
						V2 = V.row(fixed_UV_indices(c));
						ind1 = fixed_UV_indices(i);
						ind2 = fixed_UV_indices(c);
					}
				}

			}
		}

		*/
		int I1 = -1, I2 = -1;
		if (V.rows() < 1000)
		{
			
			for (int i = 0; i < V.rows() - 1; i++)
			{
				Weights.insert(Weights.end(), 1);
			}
			double MaxDis = 0;
			for (int i = 0; i < V.rows(); i++)
			{
				Eigen::VectorXd MinDis;
				Eigen::VectorXi Prev;
				igl::dijkstra(i, { V.rows() }, VV, MinDis, Prev);

				for (int c = 0; c < MinDis.rows(); c++)
				{
					if (MaxDis < MinDis(c))
					{
						for (int j = 0; j < fixed_UV_indices.rows(); j++)
						{
							for (int j1 = 0; j1 < fixed_UV_indices.rows(); j1++)
							{
								if (fixed_UV_indices(j1) == c && fixed_UV_indices(j) == i)
								{
									MaxDis = MinDis(c);
									I1 = i;
									I2 = c;
								}
							}
						}
					}
				}
			}

		}
		else
		{
			RowVector3d V1, V2;
			float MaxDistance = 0;

			for (int i = 0; i < fixed_UV_indices.rows(); i++)
			{
				for (int c = 0; c < fixed_UV_indices.rows(); c++)
				{
					if (i != c)
					{
						float v1X = V.row(fixed_UV_indices(i))(0);
						float v1Y = V.row(fixed_UV_indices(i))(1);
						float v1Z = V.row(fixed_UV_indices(i))(2);

						float v2X = V.row(fixed_UV_indices(c))(0);
						float v2Y = V.row(fixed_UV_indices(c))(1);
						float v2Z = V.row(fixed_UV_indices(c))(2);
						float dis = sqrtf(pow(v1X - v2X, 2) + pow(v1Y - v2Y, 2) + pow(v1Z - v2Z, 2));
						if (dis > MaxDistance)
						{
							MaxDistance = dis;
							V1 = V.row(fixed_UV_indices(i));
							V2 = V.row(fixed_UV_indices(c));
							I1 = fixed_UV_indices(i);
							I2 = fixed_UV_indices(c);
						}
					}

				}
			}
		}


		two_fixed_UV_indices.conservativeResize(2, 1);
		two_fixed_UV_indices(0) = I1;
		two_fixed_UV_indices(1) = I2;
		fixed_UV_indices = two_fixed_UV_indices;
		//fixed_UV_positions.conservativeResize(2, 2);
		//fixed_UV_positions(0, 0) = V(I1, 0);
		//fixed_UV_positions(0, 1) = V(I1, 1);
		//fixed_UV_positions(1, 0) = V(I2, 0);
		//fixed_UV_positions(1, 1) = V(I2, 1);

		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);
		// it is not important that is mapping to circle because there is just 2 fixed points
		// and I need a way to map 3d point to 2d point



		// Fix two UV vertices. This should be done in an intelligent way. Hint: The two fixed vertices should be the two most distant one on the mesh.
	}

	ConvertConstraintsToMatrixForm(fixed_UV_indices, fixed_UV_positions, C, d);

	// Find the linear system for the parameterization (1- Tutte, 2- Harmonic, 3- LSCM, 4- ARAP)
	// and put it in the matrix A.
	// The dimensions of A should be 2#V x 2#V.
	if (type == '1') {
		// Add your code for computing uniform Laplacian for Tutte parameterization
		// Hint: use the adjacency matrix of the mesh

		// Mesh in (V,F)
		Eigen::SparseMatrix<double> Adj;
		igl::adjacency_matrix(F, Adj);
		//   // sum each row 
		SparseVector<double> Asum;
		igl::sum(Adj, 1, Asum);
		//   // Convert row sums into diagonal of sparse matrix
		SparseMatrix<double> Adiag;
		igl::diag(Asum, Adiag);
		//   Build uniform laplacian
		SparseMatrix<double> L;
		L = Adj - Adiag;
		SparseMatrix<double> L0;
		SparseMatrix<double> L1;
		SparseMatrix<double> Zero(L.rows(), L.cols());

		L0 = igl::cat(2, L, Zero);
		L1 = igl::cat(2, Zero, L);
		A = igl::cat(1, L0, L1);
		b.resize(A.rows());
		for (int i = 0; i < b.rows(); i++)
		{
			b(i) = 0;
		}

		/*

		A.resize(2 * L.rows(), 2 * L.cols());

		for (int i = 0; i < 2 * L.rows(); i++)
		{
			for (int c = 0; c <2 * L.cols(); c++)
			{
				A.insert(i, c) = 0;

			}
		}
		for (int i = 0; i < L.rows(); i++)
		{
			for (int c = 0; c < L.cols(); c++)
			{
				A.coeffRef(i, c) = L.coeff(i, c);

			}
		}
		int k = 0;
		for (int i = L.rows(); i < 2 * L.rows(); i++, k++)
		{
			int m = 0;
			for (int c = L.cols(); c < 2 * L.cols(); c++, m++)
			{
				A.coeffRef(i, c) = L.coeff(k, m);
				cout << c << endl;
			}
		}

		*/
	}

	if (type == '2') {
		SparseMatrix<double> L;
		igl::cotmatrix(V, F, L);
		
		SparseMatrix<double> L0;
		SparseMatrix<double> L1;
		SparseMatrix<double> Zero(L.rows(), L.cols());

		L0 = igl::cat(2, L, Zero);
		L1 = igl::cat(2, Zero, L);
		A = igl::cat(1, L0, L1);
		b.resize(A.rows());
		for (int i = 0; i < b.rows(); i++)
		{
			b(i) = 0;
		}


		// Add your code for computing cotangent Laplacian for Harmonic parameterization
		// Use can use a function "cotmatrix" from libIGL, but ~~~~***READ THE DOCUMENTATION***~~~~
	}

	if (type == '3') {
		// Add your code for computing the system for LSCM parameterization
		// Note that the libIGL implementation is different than what taught in the tutorial! Do not rely on it!!
		SparseMatrix<double> D1, D2;

		VectorXd Delta;
		igl::doublearea(V, F, Delta);

		computeSurfaceGradientMatrix(D1, D2);
		SparseMatrix<double> a, bb, c, d, p1, p2, p3, p4;
		SparseMatrix<double> area;
		area.resize(F.rows(), F.rows());

		for (int i = 0; i < area.rows(); i++)
		{
			area.insert(i, i) = Delta(i);
		}
		a = D1.transpose() * area * D1;
		bb = D2.transpose() * area * D2;
		c = D2.transpose() * area * D1;
		d = D1.transpose() * area * D2;
		p1 = a + bb;
		p2 = c - d;
		p3 = d - c;
		SparseMatrix<double> M1, M2, M;
		M1 = igl::cat(2, p1, p2);
		M2 = igl::cat(2, p3, p1);
		M = igl::cat(1, M1, M2);
		A = M;
		b.resize(A.rows());
		for (int i = 0; i < b.rows(); i++)
		{
			b(i) = 0;
		}
	}

	if (type == '4') {

		A.conservativeResize(F.rows() * 2, F.rows() * 2);
		Eigen::SparseMatrix<double> Dx;
		Eigen::SparseMatrix<double> Dy;
		Eigen::SparseMatrix<double> Jac;

		Eigen::SparseMatrix<double> UpLeft;
		Eigen::SparseMatrix<double> UpRight;
		Eigen::SparseMatrix<double> DownLeft;
		Eigen::SparseMatrix<double> DownRight;

		computeSurfaceGradientMatrix(Dx, Dy);

		Eigen::VectorXd uVec = UV.col(0);
		Eigen::VectorXd vVec = UV.col(1);
		Eigen::SparseMatrix<double> uVecTemp;
		Eigen::SparseMatrix<double> vVecTemp;
		uVecTemp = uVec.sparseView();
		vVecTemp = vVec.sparseView();

		UpLeft = (Dx * uVecTemp);
		UpRight = (Dy * uVecTemp);
		DownLeft = (Dx * vVecTemp);
		DownRight = (Dy * vVecTemp);

		Eigen::SparseMatrix<double> SMUp;
		Eigen::SparseMatrix<double> SMDown;

		igl::cat(2, UpLeft, UpRight, SMUp);
		igl::cat(2, DownLeft, DownRight, SMDown);
		igl::cat(1, SMUp, SMDown, Jac);

		int tailleV = V.rows();

		Eigen::MatrixXd distorstionPerFace;
		distorstionPerFace.resize(F.rows(), 1);
		distorstionPerFace.setZero();


		for (int i = 0; i < F.rows(); i++)
		{
			double DxU = 0;
			double DxV = 0;
			double DyU = 0;
			double DyV = 0;

			for (int j = 0; j < 3; j++)
			{

				int indicePtcur = F.row(i)[j];
				DxU += Jac.coeff(indicePtcur, 0);
				DxV += Jac.coeff(tailleV + indicePtcur, 0);
				DyU += Jac.coeff(indicePtcur, 1);
				DyV += Jac.coeff(tailleV + indicePtcur, 1);
			}


			Eigen::Matrix2d J;
			J.setZero();

			J.row(0)[0] = DxU;
			J.row(0)[1] = DyU;
			J.row(1)[0] = DxV;
			J.row(1)[1] = DyV;

			Eigen::Matrix2d Utmp;
			Eigen::Matrix2d Stmp;
			Eigen::Matrix2d Vtmp;

			SSVD2x2(J, Utmp, Stmp, Vtmp);
			Eigen::Matrix2d R = Utmp * Vtmp.transpose();
			double alpha = ((DxU - DyU) + (DxV - DyV)) / 2;
			double beta = ((DxU - DyU) + (DyV - DxV)) / 2;

			double sigmaUn = Stmp.row(0)[0];
			double sigmaDeux = Stmp.row(1)[1];
			Eigen::Matrix2d Arap = J - R;


			double maxCoef = distorstionPerFace.maxCoeff();
			double minCoef = distorstionPerFace.minCoeff();

			A.insert(i * 2, i * 2) = Arap.row(0)[0];
			A.insert(i * 2, i * 2 + 1) = Arap.row(0)[1];
			A.insert(i * 2 + 1, i * 2) = Arap.row(1)[0];
			A.insert(i * 2 + 1, i * 2 + 1) = Arap.row(1)[1];
		}
	
		b.resize(A.rows());
		for (int i = 0; i < b.rows(); i++)
		{
			b(i) = 0;
		}

		// Add your code for computing ARAP system and right-hand side
		// Implement a function that computes the local step first
		// Then construct the matrix with the given rotation matrices		
		
		//I could not solve the arap, did not know what to do after :((
		return;
	}


	// Solve the linear system.
	// Construct the system as discussed in class and the assignment sheet
	// Use igl::cat to concatenate matrices
	// Use Eigen::SparseLU to solve the system. Refer to tutorial 3 for more detail
	SparseMatrix<double> _A;
	SparseMatrix<double> _A1;
	SparseMatrix<double> _A2;
	VectorXd _b;


	SparseMatrix<double> Zero(C.rows(), C.rows());

	SparseMatrix<double> Ctrans = C.transpose();
	//SparseMatrix<double> Zero1(A.rows() - Ctrans.rows(), A.cols());
	//Ctrans = igl::cat(1, Ctrans, Zero1);
	_A1 = igl::cat(2, A, Ctrans);
	_A2 = igl::cat(2, C, Zero);
	_A = igl::cat(1, _A1, _A2);

	_b = igl::cat(1, b, d);
	Eigen::SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > solver;
	solver.analyzePattern(_A);
	solver.factorize(_A);
	x = solver.solve(_b);
	cout << x << endl;
	// The solver will output a vector
	//UV.resize(V.rows(), 2);
	//UV.col(0) = x.block(0, 0, V.rows(), 1);
	//UV.col(1) = x.block(V.rows(), 0, V.rows(), 1);
	UV.resize(V.rows(), 2);
	UV.col(0) = x.block(0, 0, V.rows(), 1);
	UV.col(1) = x.block(V.rows(), 0, V.rows(), 1);
	Colors.conservativeResize(F.rows(), 3);
	Visualization();

}
void Calculate_UV()
{

}
bool callback_key_pressed(Viewer &viewer, unsigned char key, int modifiers) {

	switch (key) {
	case '1':
	case '2':
	case '3':
	case '4':
		computeParameterization(key);
		break;
	case '5':
		Calculate_UV();
			// Add your code for detecting and displaying flipped triangles in the
			// UV domain here
		break;
	case '+':
		TextureResolution /= 2;
		break;
	case '-':
		TextureResolution *= 2;
		break;
	case ' ': // space bar -  switches view between mesh and parameterization
    if(showingUV)
    {
      temp2D = viewer.core;
      viewer.core = temp3D;
      showingUV = false;
    }
    else
    {
      if(UV.rows() > 0)
      {
        temp3D = viewer.core;
        viewer.core = temp2D;
        showingUV = true;
      }
      else { std::cout << "ERROR ! No valid parameterization\n"; }
    }
    break;
	}
	Redraw();
	return true;
}

bool load_mesh(string filename)
{
	igl::read_triangle_mesh(filename,V,F);
	Redraw();
	viewer.core.align_camera_center(V);
	showingUV = false;

	return true;
}

bool callback_init(Viewer &viewer)
{
	temp3D = viewer.core;
	temp2D = viewer.core;
	temp2D.orthographic = true;

	return false;
}

int main(int argc,char *argv[]) {
	if(argc != 2) {
		cout << "Usage ex3_bin <mesh.off/obj>" << endl;
		load_mesh("../data/Octo_cut2.obj");
	}
	else
	{
		// Read points and normals
		load_mesh(argv[1]);
	}

	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("Parmaterization", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::Checkbox("Free boundary", &freeBoundary);
			if (ImGui::Checkbox("Visualization", &Visualize))
			{
				Redraw();
			}
			ImGui::InputInt("Angle Distortion", &showAngleDistortion);
			ImGui::InputDouble("Min Color Value", &Colormin);
			ImGui::InputDouble("Max Color Value", &Colormax);

			// TODO: Add more parameters to tweak here...
		}
	};

	viewer.callback_key_pressed = callback_key_pressed;
	viewer.callback_mouse_move = callback_mouse_move;
	viewer.callback_init = callback_init;

	viewer.launch();
}
