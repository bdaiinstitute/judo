/* Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved. */
#include <mujoco/mujoco.h>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>
#include <functional>
#include <ranges>
#include <unsupported/Eigen/CXX11/Tensor>

namespace mujoco_extensions::pybind::jacobian_smoothing {

namespace py = pybind11;
/**
 * Setup openmp environment.
 */
constexpr std::size_t OMP_NUM_THREADS{8};

/**
 * Define vector, matrix, and tensor types that will be used throughout this file.
 */
using VectorT = Eigen::VectorXd;
using VectorTView = Eigen::Map<VectorT>;
using VectorTConstView = Eigen::Map<const VectorT>;
using MatrixT = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixTView = Eigen::Map<MatrixT>;
using MatrixTConstView = Eigen::Map<const MatrixT>;
using Tensor1T = Eigen::Tensor<double, 1, Eigen::RowMajor>;
using Tensor1TView = Eigen::TensorMap<Tensor1T>;
using Tensor1TConstView = Eigen::TensorMap<const Tensor1T>;
using Tensor2T = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using Tensor2TView = Eigen::TensorMap<Tensor2T>;
using Tensor2TConstView = Eigen::TensorMap<const Tensor2T>;
using Tensor3T = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using Tensor3TView = Eigen::TensorMap<Tensor3T>;
using Tensor3TConstView = Eigen::TensorMap<const Tensor3T>;

/**
 * Multiply a 3D ltensor by a vector. The result is a matrix.
 */
auto multiplyTensorVector(const auto& tensor, const auto& vector) {
  const auto tensorVectorProductDimensions = Eigen::array<Eigen::IndexPair<int>, 1>({Eigen::IndexPair<int>(2, 0)});
  const auto vectorView = Tensor1TConstView(vector.data(), vector.size());
  const auto result = Tensor2T(tensor.contract(vectorView, tensorVectorProductDimensions));
  return MatrixTConstView(result.data(), result.dimensions()[0], result.dimensions()[1]);
}

/**
 * Get model and data pointers from plant object of `Plant` python type. We need this function because C++ has no
 * control over this python object, and hence we cannot simply access its storage. This function returns two pointers to
 * objects of `mjModel` and `mjData` types.
 *
 * The design follows this suggestion https://github.com/google-deepmind/mujoco/issues/983#issuecomment-1643732152
 */
std::tuple<const mjModel*, const mjData*> getModelAndData(const py::object& plantObject) {
  const auto modelPointer = plantObject.attr("model").attr("_address").cast<std::uintptr_t>();
  const mjModel* model = reinterpret_cast<mjModel*>(modelPointer);
  const auto dataPointer = plantObject.attr("data").attr("_address").cast<std::uintptr_t>();
  const mjData* data = reinterpret_cast<mjData*>(dataPointer);
  return {model, data};
}

/**
 * Get model pointer from python model object.
 *
 * The design follows this suggestion https://github.com/google-deepmind/mujoco/issues/983#issuecomment-1643732152
 */
const mjModel* getModel(const py::object& python_model) {
  const auto modelPointer = python_model.attr("_address").cast<std::uintptr_t>();
  return reinterpret_cast<const mjModel*>(modelPointer);
}

/**
 * Get data pointer from python data object.
 *
 * The design follows this suggestion https://github.com/google-deepmind/mujoco/issues/983#issuecomment-1643732152
 */
const mjData* getData(const py::object& python_data) {
  const auto dataPointer = python_data.attr("_address").cast<std::uintptr_t>();
  return reinterpret_cast<const mjData*>(dataPointer);
}

/**
 * A simple projector for slack and dual variables.
 */
void simpleProjector(VectorTView dualV, VectorTView slackV, double kappa) {
  for (const auto index : std::views::iota(0, dualV.size())) {
    auto& dual = dualV(index);
    auto& slack = slackV(index);
    if (dual * dual < kappa && slack * slack < kappa) {
      dual = sqrt(kappa);
      slack = sqrt(kappa);
    } else if (dual < slack) {
      dual = kappa / slack;
    } else {
      slack = kappa / dual;
    }
  }
}

/**
 * A polynomial projector for slack and dual variables.
 */
void polynomialProjector(VectorTView dualV, VectorTView slackV, double kappa) {
  for (const auto index : std::views::iota(0, dualV.size())) {
    auto& dual = dualV(index);
    auto& slack = slackV(index);
    const double b{slack + dual};
    const double c{slack * dual - kappa};
    const double delta{b * b - 4. * c};
    const double x{0.5 * (-b + sqrt(delta))};
    dual += x;
    slack += x;
  }
}

/**
 * Retrieve the proportional gain used by the PD controller from MuJoCo model.
 */
auto getProportionalGains(const mjModel* model) {
  const auto actuationNumber = model->nu;
  return Eigen::Map<VectorT, 0, Eigen::InnerStride<mjNGAIN>>(model->actuator_gainprm, actuationNumber);
}

/**
 * Retrieve the mass matrix.
 */
auto getMassMatrix(const mjModel* model, const mjData* data) {
  const auto variableNumber = model->nv;
  auto massMatrix = MatrixT(variableNumber, variableNumber);
  mj_fullM(model, massMatrix.data(), data->qM);
  return massMatrix;
}

/**
 * Retrieve the signed distance function for all contact locations.
 */
auto getSignedDistanceFunction(const mjData* data) {
  const auto contactNumber = data->ncon;
  auto signedDistanceFunction = VectorT(contactNumber);
  for (const auto contactIndex : std::views::iota(0, contactNumber)) {
    signedDistanceFunction(contactIndex) = data->contact[contactIndex].dist;
  }
  return signedDistanceFunction;
}

/**
 * Types of contact loctions.
 */
enum ContactLocation { LEFT = 0b1, RIGHT = 0b10, RELATIVE = 0b11 };

/**
 * Compute contact Jacobian from the model and data given a contact location type.
 */
Tensor3T getContactJacobian(const mjModel* model, const mjData* data, const ContactLocation contactLocation) {
  const auto variablesNumber = model->nv;
  const auto contactNumber = data->ncon;
  const auto allContacts = data->contact;
  auto contactJacobian = Tensor3T(contactNumber, 3, variablesNumber);
  auto jacp = MatrixT(3, variablesNumber);
  double offset[3] = {0, 0, 0};
  for (const auto contactIndex : std::views::iota(0, contactNumber)) {
    const auto contact = allContacts[contactIndex];
    const auto geom1 = contact.geom[0];
    const auto geom2 = contact.geom[1];
    const auto body1 = model->geom_bodyid[geom1];
    const auto body2 = model->geom_bodyid[geom2];
    switch (contactLocation) {
      case ContactLocation::LEFT:
        jacp.setZero();
        for (const auto index : std::views::iota(0, 3)) {
          offset[index] = contact.pos[index] - 0.5 * contact.dist * contact.frame[index];
        }
        mj_jac(model, data, jacp.data(), NULL, offset, body1);
        contactJacobian.chip(contactIndex, 0) = Tensor2TView(jacp.data(), jacp.rows(), jacp.cols());
        break;
      case ContactLocation::RIGHT:
        jacp.setZero();
        for (const auto index : std::views::iota(0, 3)) {
          offset[index] = contact.pos[index] + 0.5 * contact.dist * contact.frame[index];
        }
        mj_jac(model, data, jacp.data(), NULL, offset, body2);
        contactJacobian.chip(contactIndex, 0) = Tensor2TView(jacp.data(), jacp.rows(), jacp.cols());
        break;
      case ContactLocation::RELATIVE:
        jacp.setZero();
        for (const auto index : std::views::iota(0, 3)) {
          offset[index] = contact.pos[index] + 0.5 * contact.dist * contact.frame[index];
        }
        mj_jac(model, data, jacp.data(), NULL, offset, body2);
        contactJacobian.chip(contactIndex, 0) = Tensor2TView(jacp.data(), jacp.rows(), jacp.cols());
        jacp.setZero();
        for (const auto index : std::views::iota(0, 3)) {
          offset[index] = contact.pos[index] - 0.5 * contact.dist * contact.frame[index];
        }
        mj_jac(model, data, jacp.data(), NULL, offset, body1);
        contactJacobian.chip(contactIndex, 0) -= Tensor2TView(jacp.data(), jacp.rows(), jacp.cols());
        break;
    }
  }

  return contactJacobian;
}
/**
 * Compute contact velocity.
 */
MatrixT getContactVelocity(const mjData* data, const Tensor3T& contactJacobian) {
  const auto variableNumber = contactJacobian.dimensions()[2];
  const auto qvel = VectorTConstView(data->qvel, variableNumber);
  return multiplyTensorVector(contactJacobian, qvel);
}

/**
 * Compute contact wrench.
 */
MatrixT getContactWrench(const mjModel* model, const mjData* data) {
  const auto contactNumber = data->ncon;
  auto contactWrench = MatrixT(contactNumber, 6);
  for (const auto contactIndex : std::views::iota(0, contactNumber)) {
    mj_contactForce(model, data, contactIndex, contactWrench.data() + contactIndex * 6);
  }
  return contactWrench;
}

/**
 * Compute contact force---the first coordinate of the contact wrench.
 */
VectorT getNormalForce(const MatrixT& contactWrench) {
  return contactWrench(Eigen::indexing::all, 0);
}

/**
 * Compute tangential force---the second and third coordinates of the contact wrench.
 */
MatrixT getTangentialForce(const MatrixT& contactWrench) {
  return contactWrench(Eigen::indexing::all, {1, 2});
}

/**
 * Compute pyramidal force---stacked tandential and negative tangential forces.
 */
MatrixT getPyramidalForce(const MatrixT& tangentialForce) {
  const auto contactNumber = tangentialForce.rows();
  auto pyramidalForce = MatrixT(contactNumber, 4);
  pyramidalForce(Eigen::indexing::all, {0, 1}) = tangentialForce.cwiseMax(0.0);
  pyramidalForce(Eigen::indexing::all, {2, 3}) = -tangentialForce.cwiseMin(0.0);
  return pyramidalForce;
}

/**
 * Retrieve friction coefficient at all contact locations.
 */
VectorT getFrictionCoefficient(const mjData* data) {
  const auto contactNumber = data->ncon;
  auto frictionCoefficient = VectorT(contactNumber);
  for (const auto contactIndex : std::views::iota(0, contactNumber)) {
    frictionCoefficient(contactIndex) = data->contact[contactIndex].friction[0];
  }
  return frictionCoefficient;
}

/**
 * Retrive contact frame of reference for all contact locations.
 */
Tensor3T getContactFrame(const mjData* data) {
  const auto contactNumber = data->ncon;
  auto contactFrame = Tensor3T(contactNumber, 3, 3);
  for (const auto contactIndex : std::views::iota(0, contactNumber)) {
    const auto frame = Tensor2TConstView(data->contact[contactIndex].frame, 3, 3);
    contactFrame.chip(contactIndex, 0) = frame;
  }
  return contactFrame;
}

/**
 * Compute contact Jacobian in contact frame of reference for all contact locations.
 */
void getContactJacobianInContactFrame(Tensor3TView contactJacobian, const Tensor3T& contactFrame) {
  Eigen::array<Eigen::IndexPair<int>, 1> productDimensions = {Eigen::IndexPair<int>(1, 0)};
  const auto dimensions = contactJacobian.dimensions();
  auto jacf = Tensor2T(dimensions[1], dimensions[2]);
  for (const auto index : std::views::iota(0, dimensions[0])) {
    jacf = contactFrame.chip(index, 0).contract(contactJacobian.chip(index, 0), productDimensions);
    contactJacobian.chip(index, 0) = jacf;
  }
}

/**
 * Compute linearized contact Jacobian---stacked contact jacobian in the contact tangent plane and its negative.
 */
MatrixT getLinearizedContactJacobian(const Tensor3T& contactJacobianInContactFrame) {
  const auto dimensions = contactJacobianInContactFrame.dimensions();
  auto linearizedContactJacobian = Tensor3T(dimensions[0], 4, dimensions[2]);
  const auto dataBegin = Eigen::array<Eigen::Index, 3>({0, 1, 0});
  const auto dataShift = Eigen::array<Eigen::Index, 3>({dimensions[0], 2, dimensions[2]});
  {
    const auto resultBegin = Eigen::array<Eigen::Index, 3>({0, 0, 0});
    const auto resultShift = Eigen::array<Eigen::Index, 3>({dimensions[0], 2, dimensions[2]});
    linearizedContactJacobian.slice(resultBegin, resultShift) =
        contactJacobianInContactFrame.slice(dataBegin, dataShift);
  }
  {
    const auto resultBegin = Eigen::array<Eigen::Index, 3>({0, 2, 0});
    const auto resultShift = Eigen::array<Eigen::Index, 3>({dimensions[0], 2, dimensions[2]});
    linearizedContactJacobian.slice(resultBegin, resultShift) =
        -contactJacobianInContactFrame.slice(dataBegin, dataShift);
  }
  return MatrixTConstView(linearizedContactJacobian.data(), 4 * dimensions[0], dimensions[2]);
}

/**
 * Compute signed distance Jacobian---the contact jacobian in the normal contact direction.
 */
auto getApproximateSignedDistanceJacobian(const auto& contactJacobianInContactFrame) {
  const auto dimensions = contactJacobianInContactFrame.dimensions();
  return Eigen::Map<const MatrixT, 0, Eigen::OuterStride<>>(contactJacobianInContactFrame.data(), dimensions[0],
                                                            dimensions[2], Eigen::OuterStride<>(3 * dimensions[2]));
}

std::tuple<VectorT, VectorT> getSlackAndDualVariables(const VectorT& normalForce, const VectorT& signedDistanceFunction,
                                                      double timeStep, double floor) {
  const auto gamma = (normalForce * timeStep).cwiseMax(1.e-4);
  const auto s_gamma = signedDistanceFunction.cwiseMax(floor);
  return {gamma, s_gamma};
}

/**
 * Compute resudual system state
 */
MatrixT getResidualJacobianVariable(const mjModel* model, const mjData* data, double timeStep, double kappa,
                                    double floor) {
  const auto contactNumber = data->ncon;
  const auto variableNumber = model->nv;
  const auto nbeta = 4 * contactNumber;
  const auto nz = variableNumber + 4 * contactNumber + 2 * nbeta;

  // Retrieve physical quantities
  const auto massMatrix = getMassMatrix(model, data);
  auto contactJacobian = getContactJacobian(model, data, ContactLocation::RELATIVE);
  const auto contactFrame = getContactFrame(data);
  getContactJacobianInContactFrame(contactJacobian, contactFrame);
  const MatrixT approximateSignedDistanceJacobian = getApproximateSignedDistanceJacobian(contactJacobian);
  const auto linearizedContactJacobian = getLinearizedContactJacobian(contactJacobian);
  const auto signedDistanceFunction = getSignedDistanceFunction(data);
  const auto frictionCoefficient = getFrictionCoefficient(data);
  const auto qvel = VectorTConstView(data->qvel, variableNumber);
  const auto pyramidTangentialVelocity = linearizedContactJacobian * qvel;
  // stretched identity matrix
  auto stretchedI = MatrixT(contactNumber, nbeta);
  stretchedI.setZero();
  for (const auto contactIndex : std::views::iota(0, contactNumber)) {
    stretchedI(contactIndex, 4 * contactIndex) = 1.;
    stretchedI(contactIndex, 4 * contactIndex + 1) = 1.;
    stretchedI(contactIndex, 4 * contactIndex + 2) = 1.;
    stretchedI(contactIndex, 4 * contactIndex + 3) = 1.;
  }

  // get slack and dual variable
  // TODO(slecleach): we might need to change the way we project onto the relaxed complementatirty constraint
  const auto contactWrench = getContactWrench(model, data);
  const auto normalForce = getNormalForce(contactWrench);
  auto [gamma, s_gamma] = getSlackAndDualVariables(normalForce, signedDistanceFunction, timeStep, floor);

  // this is very important to keep, without this the gradients throught IMPACT
  // are not visible on the unactuated object (do not exist?)
  // polynomialProjector(gamma, s_gamma, kappa);
  simpleProjector(VectorTView(gamma.data(), gamma.size()), VectorTView(s_gamma.data(), s_gamma.size()), kappa);

  // TODO(slecleach): need to check the time step scaling, we can use the scaling gamma/measured impact force
  const auto tangentialForce = getTangentialForce(contactWrench);
  auto beta = VectorT(nbeta);
  MatrixTView(beta.data(), contactNumber, 4) = getPyramidalForce(tangentialForce) * timeStep;

  const VectorT s_c = (frictionCoefficient.cwiseProduct(gamma) - stretchedI * beta).cwiseMax(floor);
  const auto c = kappa * s_c.cwiseInverse();

  auto one_c = VectorT(4 * c.size());
  for (const auto index : std::views::iota(0, c.size())) {
    one_c(4 * index) = c(index);
    one_c(4 * index + 1) = c(index);
    one_c(4 * index + 2) = c(index);
    one_c(4 * index + 3) = c(index);
  }
  VectorT s_beta = pyramidTangentialVelocity + one_c;

  // this is very important to keep, without this the gradients throught FRICTION
  // are not visible on the unactuated object (do not exist?)
  // to reduce the gradient at a distance for friction
  polynomialProjector(VectorTView(beta.data(), beta.size()), VectorTView(s_beta.data(), s_beta.size()), kappa * 0.05);

  // slices
  std::size_t offset{0};
  const auto i0 = Eigen::seq(offset, variableNumber - 1);
  offset += variableNumber;
  const auto i1 = Eigen::seq(offset, offset + contactNumber - 1);
  offset += contactNumber;
  const auto i2 = Eigen::seq(offset, offset + nbeta - 1);
  offset += nbeta;
  const auto i3 = Eigen::seq(offset, offset + contactNumber - 1);
  offset += contactNumber;
  const auto i4 = Eigen::seq(offset, offset + contactNumber - 1);
  offset += contactNumber;
  const auto i5 = Eigen::seq(offset, offset + nbeta - 1);
  offset += nbeta;
  const auto i6 = Eigen::seq(offset, offset + contactNumber - 1);
  offset += contactNumber;

  // build and populate jacobian matrix
  auto residualJacobianVariable = MatrixT(nz, nz);
  residualJacobianVariable.setZero();

  residualJacobianVariable(i0, i0) = 1. / timeStep * massMatrix;
  residualJacobianVariable(i0, i1) = -approximateSignedDistanceJacobian.transpose();
  residualJacobianVariable(i1, i0) = -approximateSignedDistanceJacobian;
  residualJacobianVariable(i0, i2) = -linearizedContactJacobian.transpose();
  residualJacobianVariable(i2, i0) = -1. / timeStep * linearizedContactJacobian;

  residualJacobianVariable(i3, i1) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(-frictionCoefficient);
  residualJacobianVariable(i3, i2) = stretchedI;
  residualJacobianVariable(i2, i3) = stretchedI.transpose();

  residualJacobianVariable(i1, i4) = MatrixT::Identity(contactNumber, contactNumber);
  residualJacobianVariable(i2, i5) = MatrixT::Identity(nbeta, nbeta);
  residualJacobianVariable(i3, i6) = MatrixT::Identity(contactNumber, contactNumber);

  residualJacobianVariable(i4, i1) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(s_gamma);
  residualJacobianVariable(i4, i4) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(gamma);

  residualJacobianVariable(i5, i2) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(s_beta);
  residualJacobianVariable(i5, i5) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(beta);

  residualJacobianVariable(i6, i3) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(s_c);
  residualJacobianVariable(i6, i6) = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(c);
  return residualJacobianVariable;
}

/**
 * Compute a mapping betwen actuation and position coordinates.
 */
Eigen::VectorXi getActuatedPos(const mjModel* model) {
  const auto nq = model->nq;
  const auto nu = model->nu;
  auto joints = Eigen::VectorXi(nu);
  for (const auto iu : std::views::iota(0, nu)) {
    joints(iu) = model->actuator_trnid[iu * 2];
  }
  const auto posAddress = Eigen::Map<Eigen::VectorXi>(model->jnt_qposadr, model->njnt);
  auto actuatedPosV = std::vector<int>();
  for (const auto ijnt : joints) {
    const auto start = posAddress(ijnt);
    const auto end = ijnt + 1 < model->njnt ? posAddress[ijnt + 1] : nq;
    for (const auto index : std::views::iota(start, end)) {
      actuatedPosV.emplace_back(index);
    }
  }
  return Eigen::Map<Eigen::VectorXi>(actuatedPosV.data(), actuatedPosV.size());
}

/**
 * TODO(slecleach): Document this function.
 */
MatrixT getResidualJacobianAction2(const mjModel* model, const mjData* data, double timeStep) {
  const auto variableNumber = model->nv;
  const auto contactNumber = data->ncon;
  const auto nu = model->nu;
  const auto nz = variableNumber + 2 * contactNumber;

  auto jacobian = MatrixT(nz, nu);
  jacobian.setZero();
  const auto Kp = getProportionalGains(model);
  const auto actuatedPos = getActuatedPos(model);
  for (const auto i : std::views::iota(0, actuatedPos.size())) {
    const auto idx = actuatedPos(i);
    jacobian(idx, i) = -timeStep * Kp(i);
  }
  return jacobian;
}

/**
 * TODO (slecleach): Document this function.
 */
MatrixT getResidualJacobianAction(const mjModel* model, const mjData* data, double timeStep) {
  const auto variableNumber = model->nv;
  const auto controlNumber = model->nu;
  const auto contactNumber = data->ncon;
  const auto nbeta = 4 * contactNumber;
  const auto nz = variableNumber + 4 * contactNumber + 2 * nbeta;

  auto jacobian = MatrixT(nz, controlNumber);
  jacobian.setZero();
  const auto Kp = getProportionalGains(model);
  const auto actuatedPos = getActuatedPos(model);
  for (const auto i : std::views::iota(0, actuatedPos.size())) {
    const auto idx = actuatedPos(i);
    jacobian(idx, i) = -timeStep * Kp(i);
  }
  return jacobian;
}

/**
 * TODO (slecleach): Document this function.
 */
MatrixT getVariableJacobianAction(const mjModel* model, const mjData* data, double timeStep, double kappa,
                                  double floor) {
  const auto jacobianVariable = getResidualJacobianVariable(model, data, timeStep, kappa, floor);
  const auto jacobianAction = getResidualJacobianAction(model, data, timeStep);
  const auto jacobianVariableInverse = jacobianVariable.partialPivLu();
  const auto jacobian = -jacobianVariableInverse.solve(jacobianAction);

  return jacobian;
}

/**
 * TODO (slecleach): Document this function.
 */
MatrixT getGammaJacobianAction(const mjModel* model, const mjData* data, double timeStep, double kappa, double floor) {
  const auto nv = model->nv;
  const auto nc = data->ncon;
  const auto variableJacobianAction = getVariableJacobianAction(model, data, timeStep, kappa, floor);
  const auto jacobian = variableJacobianAction(Eigen::seq(nv, nv + nc - 1), Eigen::indexing::all);
  return jacobian;
}

/**
 * TODO (slecleach): Document this function.
 */
MatrixT getSGammaJacobianAction(const mjModel* model, const mjData* data, double timeStep, double kappa, double floor) {
  const auto nv = model->nv;
  const auto nc = data->ncon;
  const auto nbeta = 4 * nc;
  const auto variableJacobianAction = getVariableJacobianAction(model, data, timeStep, kappa, floor);
  const auto jacobian =
      variableJacobianAction(Eigen::seq(nv + nc + nbeta + nc, nv + nc + nbeta + nc + nc - 1), Eigen::indexing::all);
  return jacobian;
}

/**
 * TODO (slecleach): Document this function.
 */
MatrixT getStateJacobianAction(const mjModel* model, const mjData* data, double timeStep, double kappa, double floor) {
  const auto nv = model->nv;
  const auto variableJacobianAction = getVariableJacobianAction(model, data, timeStep, kappa, floor);
  const auto posJacobianAction = variableJacobianAction(Eigen::seq(0, nv - 1), Eigen::indexing::all);
  const auto velJacobianAction = posJacobianAction / timeStep;
  auto jacobian = MatrixT(2 * nv, variableJacobianAction.cols());
  jacobian(Eigen::seq(0, nv - 1), Eigen::indexing::all) = posJacobianAction;
  jacobian(Eigen::seq(nv, 2 * nv - 1), Eigen::indexing::all) = velJacobianAction;
  return jacobian;
}

/**
 * Bindings for `jacobian_smoothing` submodule.
 */
void bindJacobianSmoothing(const std::reference_wrapper<py::module>& root) {
  Eigen::setNbThreads(OMP_NUM_THREADS);
  // Create `jacobian_smoothing` submodule.
  auto python_module = root.get().def_submodule("jacobian_smoothing");

  // Bind ContactLocation emum.
  py::enum_<ContactLocation>(python_module, "ContactLocation")
      .value("LEFT", ContactLocation::LEFT)
      .value("RIGHT", ContactLocation::RIGHT)
      .value("RELATIVE", ContactLocation::RELATIVE);

  // Bind functions for computing smoothed Jacobians after MuJoCo step.
  python_module.def(
      "get_proportional_gains",
      [](const py::object& python_model, const py::object& python_data) -> VectorT {
        const auto model = getModel(python_model);
        return getProportionalGains(model);
      },
      "Retrieve the proportional gain used by the PD controller from the mujoco model.");

  python_module.def(
      "get_mass_matrix",
      [](const py::object& python_model, const py::object& python_data) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getMassMatrix(model, data);
      },
      "Retrieve the mass matrix");

  python_module.def(
      "get_sdf",
      [](const py::object& python_data) -> VectorT {
        const auto data = getData(python_data);
        return getSignedDistanceFunction(data);
      },
      "Computes the signed distance function for the nc contact points.");

  python_module.def(
      "get_contact_jacobian",
      [](const py::object& python_model, const py::object& python_data,
         const ContactLocation contactLocation) -> Tensor3T {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getContactJacobian(model, data, contactLocation);
      },
      "Compute the velocity of the contact point wrt the **world** frame, expressed in the **world** frame.\nThis is "
      "the velocity of the contact point **attached** to the geometry it belongs to.",
      py::arg("model"), py::arg("data"), py::arg("mode") = ContactLocation::RELATIVE);

  python_module.def(
      "get_contact_velocity",
      [](const py::object& python_model, const py::object& python_data,
         const ContactLocation contactLocation) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        const auto contactJacobian = getContactJacobian(model, data, contactLocation);
        return getContactVelocity(data, contactJacobian);
      },
      "Compute the velocity of the contact point wrt the **world** frame, expressed in the **world** frame.\nThis is "
      "the velocity of the contact point **attached** to the geometry it belongs to.",
      py::arg("model"), py::arg("data"), py::arg("mode") = ContactLocation::RELATIVE);

  python_module.def(
      "get_contact_wrench",
      [](const py::object& python_model, const py::object& python_data) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getContactWrench(model, data);
      },
      "https://mujoco.readthedocs.io/en/stable/programming/simulation.html#contacts");

  python_module.def(
      "get_normal_force",
      [](const py::object& python_model, const py::object& python_data) -> VectorT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getNormalForce(getContactWrench(model, data));
      },
      "https://mujoco.readthedocs.io/en/stable/programming/simulation.html#contacts");

  python_module.def(
      "get_tangential_force",
      [](const py::object& python_model, const py::object& python_data, const bool pyramidalContact) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        const auto contactWrench = getContactWrench(model, data);
        const auto tangentialForce = getTangentialForce(contactWrench);
        if (!pyramidalContact) {
          return tangentialForce;
        }
        return getPyramidalForce(tangentialForce);
      },
      "https://mujoco.readthedocs.io/en/stable/programming/simulation.html#contacts\nFor pyramidal contact we have the "
      "force expressed as b > 0, b in R^4 each\ncomponent is aligned with the X, Y, -X, -Y axes\n",
      py::arg("model"), py::arg("data"), py::arg("pyramidal_contact") = false);

  python_module.def(
      "get_friction_coefficient",
      [](const py::object& python_data) -> VectorT {
        const auto data = getData(python_data);
        return getFrictionCoefficient(data);
      },
      "Retrieves the tangential friction coefficient for each active contact point.\nTODO (slecleach):\nThis is "
      "returning an scalar value for each contact point. There are actually\n2 tangential friction coefficients for "
      "both sliding directions. We ignore this for now\nand only consider the first sliding dimension.");

  python_module.def(
      "linearized_contact_jacobian",
      [](const py::object& python_model, const py::object& python_data) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        auto contactJacobian = getContactJacobian(model, data, ContactLocation::RELATIVE);
        const auto contactFrame = getContactFrame(data);
        getContactJacobianInContactFrame(contactJacobian, contactFrame);
        return getLinearizedContactJacobian(contactJacobian);
      },
      "Jacobian mapping linearized friction forces to generalized forces.");

  python_module.def(
      "approximate_sdf_jacobian",
      [](const py::object& python_model, const py::object& python_data) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        auto contactJacobian = getContactJacobian(model, data, ContactLocation::RELATIVE);
        const auto contactFrame = getContactFrame(data);
        getContactJacobianInContactFrame(contactJacobian, contactFrame);
        return getApproximateSignedDistanceJacobian(contactJacobian);
      },
      "Jacobian mapping generalized velocities to sdf gradients.\nWe make the approximation that the contact point are "
      "fixed to the geometries they belong to\nthis is not true but simplifies the computation significantly.\n");

  python_module.def(
      "get_complementarity_variables",
      [](const py::object& python_model, const py::object& python_data, double timeStep, double kappa,
         double floor) -> std::tuple<VectorT, VectorT> {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        const auto normalForce = getNormalForce(getContactWrench(model, data));
        const auto signedDistanceFunction = getSignedDistanceFunction(data);
        auto [gamma, s_gamma] = getSlackAndDualVariables(normalForce, signedDistanceFunction, timeStep, floor);
        polynomialProjector(VectorTView(gamma.data(), gamma.size()), VectorTView(s_gamma.data(), s_gamma.size()),
                            kappa);
        return std::make_tuple(gamma, s_gamma);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1, py::arg("kappa") = 3.e-3,
      py::arg("floor") = 1.e-10);

  python_module.def(
      "residual_jacobian_variable",
      [](const py::object& python_model, const py::object& python_data, double timeStep, double kappa,
         double floor) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getResidualJacobianVariable(model, data, timeStep, kappa, floor);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1, py::arg("kappa") = 3.e-3,
      py::arg("floor") = 1.e-10);

  python_module.def(
      "residual_jacobian_action_2",
      [](const py::object& python_model, const py::object& python_data, double timeStep) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getResidualJacobianAction2(model, data, timeStep);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1);

  python_module.def(
      "residual_jacobian_action",
      [](const py::object& python_model, const py::object& python_data, double timeStep) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getResidualJacobianAction(model, data, timeStep);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1);

  python_module.def(
      "variable_jacobian_action",
      [](const py::object& python_model, const py::object& python_data, double timeStep, double kappa,
         double floor) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getVariableJacobianAction(model, data, timeStep, kappa, floor);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1, py::arg("kappa") = 3.e-3,
      py::arg("floor") = 1.e-10);

  python_module.def(
      "gamma_jacobian_action",
      [](const py::object& python_model, const py::object& python_data, double timeStep, double kappa,
         double floor) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getGammaJacobianAction(model, data, timeStep, kappa, floor);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1, py::arg("kappa") = 3.e-3,
      py::arg("floor") = 1.e-10);

  python_module.def(
      "s_gamma_jacobian_action",
      [](const py::object& python_model, const py::object& python_data, double timeStep, double kappa,
         double floor) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getSGammaJacobianAction(model, data, timeStep, kappa, floor);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1, py::arg("kappa") = 3.e-3,
      py::arg("floor") = 1.e-10);

  python_module.def(
      "state_jacobian_action",
      [](const py::object& python_model, const py::object& python_data, double timeStep, double kappa,
         double floor) -> MatrixT {
        const auto model = getModel(python_model);
        const auto data = getData(python_data);
        return getStateJacobianAction(model, data, timeStep, kappa, floor);
      },
      py::arg("model"), py::arg("data"), py::arg("time_step") = 1.e-1, py::arg("kappa") = 3.e-3,
      py::arg("floor") = 1.e-10);
}

}  // namespace mujoco_extensions::pybind::jacobian_smoothing
