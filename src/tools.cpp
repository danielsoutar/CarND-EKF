#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, 
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if(estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout << "CalculateRMSE() - Error - invalid vector sizes\n";
    return rmse;
  }

	for (int i = 0; i < estimations.size(); ++i) {
    VectorXd delta = estimations[i] - ground_truth[i];
    delta = delta.array() * delta.array();
    rmse += delta;
	}

  // Divide by n to get mean
  rmse /= estimations.size();

  // Get square root with Eigen sqrt function
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  float px = x_state[0];
  float py = x_state[1];
  float vx = x_state[2];
  float vy = x_state[3];

  float pxpysq = px*px + py*py;

  if(pxpysq < 0.0001) { // Division by 0, or virtually by 0
    cout << "CalculateJacobian() - Error - division by 0\n";
    return Hj;
  }

  float rtpxpysq = std::sqrt(pxpysq);
  float fracpxpysq = rtpxpysq * rtpxpysq * rtpxpysq;

  float vxpy = vx * py;
  float vypx = vy * px;

  float px_rtpxpysq = px / rtpxpysq;
  float py_rtpxpysq = py / rtpxpysq;

  Hj << px_rtpxpysq, py_rtpxpysq, 0, 0,
        -py / pxpysq, px / pxpysq, 0, 0,
        (py * (vxpy - vypx)) / fracpxpysq, (px * (vypx - vxpy)) / fracpxpysq, px_rtpxpysq, py_rtpxpysq;

  return Hj;
}






