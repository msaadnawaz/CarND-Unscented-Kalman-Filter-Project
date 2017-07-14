#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2 //incorrectly set by Udacity
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2 //incorrectly set by Udacity
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;

  previous_timestamp_ = 0;

  time_us_ = 0;

  lambda_ = 3 - n_aug_;

  x_aug_ = VectorXd(n_aug_);

  weights_ = VectorXd(2*n_aug_+1);

  // initial augmented covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  //initial sigma point augmented matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for(int i=1; i<2*n_aug_+1; i++){
	  weights_(i) = 1/(2*(lambda_+n_aug_));
  }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
	  cout<< endl << "initialization started" <<endl;
	  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		  double rho = meas_package.raw_measurements_[0];
		  double phi = meas_package.raw_measurements_[1];
		  double rhod = meas_package.raw_measurements_[2];

		  x_.fill(0);
		  x_(0) = rho * cos(phi); //px
		  x_(1) = rho * sin(phi); //py
		  x_(2) = rhod; //v
	  }
	  else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
		  x_.fill(0);
		  x_(0) = meas_package.raw_measurements_[0]; //px
		  x_(1) = meas_package.raw_measurements_[1]; //py
	  }

	  P_ = MatrixXd::Identity(n_x_, n_x_);

	  previous_timestamp_ = meas_package.timestamp_;
	  is_initialized_ = true;
	  cout<< endl << "initialization done" <<endl;
	  return;
  }

  //////////////////////
  /* ***PREDICTION*** */
  //////////////////////
  cout<< endl << "processing prediction" <<endl;
  time_us_ = (meas_package.timestamp_ - previous_timestamp_);
  Prediction(time_us_);
  cout<< endl << "prediction done" <<endl;

  //////////////////
  /* ***UPDATE*** */
  //////////////////

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	  cout<< endl << "updating radar" <<endl;
	  UpdateRadar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
	  cout<< endl << "updating lidar" <<endl;
	  UpdateLidar(meas_package);
  }
  previous_timestamp_ = meas_package.timestamp_;
  cout<< endl << "update done" <<endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /////////////////////////////////
  /* ***GENERATE SIGMA POINTS*** */
  /////////////////////////////////
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  P_aug_.fill(0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_;
  P_aug_(6,6) = std_yawdd_;

  //calculate square root of P
  MatrixXd A = P_aug_.llt().matrixL();

  //first column gets augmented state vector
  Xsig_aug_.col(0) = x_aug_;

  //sigma points generation formula
  for(int i=0; i<n_aug_; i++){
	  Xsig_aug_.col(i+1) = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);
	  Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  cout<< endl << "sigma points generation done" <<endl;
  ////////////////////////////////
  /* ***PREDICT SIGMA POINTS*** */
  ////////////////////////////////

  //predict sigma points
  double p_x, p_y, v, yaw, yawd, nu_a, nu_yawdd;
  VectorXd curr_x = VectorXd(n_x_);
  VectorXd integ = VectorXd(n_x_);
  VectorXd proc_noise = VectorXd(n_x_);

  for(int i=0; i<(2*n_aug_+1); i++){
	  p_x = Xsig_aug_.col(i)(0);
	  p_y = Xsig_aug_.col(i)(1);
	  v = Xsig_aug_.col(i)(2);
	  yaw = Xsig_aug_.col(i)(3);
	  yawd = Xsig_aug_.col(i)(4);
	  nu_a = Xsig_aug_.col(i)(5);
	  nu_yawdd = Xsig_aug_.col(i)(6);

	  curr_x << p_x, p_y, v, yaw, yawd;
	  //avoid division by zero
	  if (fabs(yawd) > 0.001) {
		  integ << v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw)),
		  		   v/yawd * (-cos(yaw + yawd * delta_t) + cos(yaw)),
				   0,
				   yawd * delta_t,
				   0;
	  }
	  else {
		  integ << v * cos(yaw) * delta_t,
				   v * sin(yaw) * delta_t,
				   0,
				   0,
				   0;
	  }

	  proc_noise << 0.5 * delta_t * delta_t * cos(yaw) * nu_a,
			  	  	0.5 * delta_t * delta_t * sin(yaw) * nu_a,
					delta_t * nu_a,
					0.5 * delta_t * delta_t * nu_yawdd,
					delta_t * nu_yawdd;

	  //write predicted sigma points into right column
	  Xsig_pred_.col(i) = curr_x + integ + proc_noise;
  }

  cout<< endl << "sigma points prediction done" <<endl;
  ///////////////////////////////////////
  /* ***PREDICT MEAN AND COVARIANCE*** */
  ///////////////////////////////////////

  //predict state mean
  x_.fill(0);
  for(int i=0; i<2*n_aug_+1; i++){
	  x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //may need to normalize yaw

  //predict state covariance matrix
  P_.fill(0);
  for(int i=0; i<2*n_aug_+1; i++){
	  cout<< endl << "Xsig_pred_.col(i): " << endl << Xsig_pred_.col(i) <<endl ;
	  cout<< endl << "x_: " << endl << x_ <<endl ;
	  VectorXd x_diff = Xsig_pred_.col(i) - x_;
	  while(x_diff(3) > M_PI)
		  x_diff(3) -= 2*M_PI;
	  while(x_diff(3) < -M_PI)
		  x_diff(3) += 2*M_PI;
	  P_ += weights_(i) * x_diff * x_diff.transpose();
	  cout << endl<< "P from loop " << i << ":" << endl << P_ << endl;
  }

  cout << "Prediction x: \n" << x_ << endl;
  cout << "Prediction P: \n" << P_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  //////////////////////////////////
  /* ***MEASUREMENT PREDICTION*** */
  //////////////////////////////////

  //measurement state dimension
  int n_z_ = 2;

  //radar measurement vector
  VectorXd z_ = VectorXd(n_z_);
  z_ << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1];

  double p_x, p_y;

  //create transformed sigma point matrix
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_x_ + 1);

  //mean predicted state
  VectorXd z_pred_ = VectorXd(n_z_);

  //transform sigma points into measurement space
  for(int i=0; i<2*n_aug_+1; i++){
      p_x = Xsig_pred_(0,i);
	  p_y = Xsig_pred_(1,i);

	  Zsig_.col(i) << p_x,
					  p_y;
  }

  //calculate mean predicted measurement
  z_pred_.fill(0);

  for(int i=0; i<2*n_aug_+1; i++){
	  z_pred_ += weights_(i) *Zsig_.col(i);
  }

  //create measurement noise covariance matrix
  MatrixXd R_ = MatrixXd(n_z_,n_z_);
  R_ << std_laspx_*std_laspx_,                     0,
						  	0, std_laspy_*std_laspy_;

  //create measurement noise covariance matrix
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  S_.fill(0);
  for(int i=0;i<2*n_aug_+1;i++){
	  S_ += weights_(i) * (Zsig_.col(i)-z_pred_) * (Zsig_.col(i)-z_pred_).transpose();
  }

  S_ += R_;

  /////////////////////////
  /* ***UPDATE STATE*** */
  /////////////////////////

  //initial Kalman gain
  MatrixXd K_ = MatrixXd(n_x_, n_z_);

  //initial cross-correlation matrix
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc_.fill(0);
  for(int i=0; i<2*n_aug_+1; i++){
	  VectorXd z_diff = Zsig_.col(i) - z_pred_;

  	  VectorXd x_diff = Xsig_pred_.col(i) - x_;

	  while(x_diff(3) > M_PI)
		  x_diff(3) -= 2*M_PI;
	  while(x_diff(3) < -M_PI)
		  x_diff(3) += 2*M_PI;

	  Tc_ += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  K_ = Tc_ * S_.inverse();

  VectorXd z_diff = z_ - z_pred_;

  //update state mean and covariance matrix
  x_ += K_ * z_diff;

  P_ -= K_ * S_ * K_.transpose();

  cout << "Update lidar x: \n" << x_ << endl;
  cout << "Update lidar P: \n" << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //measurement state dimension
  int n_z_ = 3;

  //radar measurement vector
  VectorXd z_ = VectorXd(n_z_);
  z_ << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1],
		meas_package.raw_measurements_[2];

  double rho, phi, rhod, p_x, p_y, v, yaw;

  //create transformed sigma point matrix
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);;

  //mean predicted state
  VectorXd z_pred_ = VectorXd(n_z_);

  //////////////////////////////////
  /* ***MEASUREMENT PREDICTION*** */
  //////////////////////////////////

  //transform sigma points into measurement space
  for(int i=0; i<2*n_aug_+1; i++){
	  p_x = Xsig_pred_(0,i);
	  p_y = Xsig_pred_(1,i);
	  v = Xsig_pred_(2,i);
	  yaw = Xsig_pred_(3,i);

	  rho = sqrt(p_x*p_x + p_y*p_y);
	  phi = atan2(p_y,p_x);
	  rhod = (p_x*cos(yaw)*v + p_y*sin(yaw)*v)/sqrt(p_x*p_x + p_y*p_y);

	  Zsig_.col(i) << rho,
			  	  	 phi,
					 rhod;
  }

  //calculate mean predicted measurement
  z_pred_.fill(0);

  for(int i=0; i<2*n_aug_+1; i++){
	  z_pred_ += weights_(i) *Zsig_.col(i);
  }

  //create measurement noise covariance matrix
  MatrixXd R_;

  R_ << std_radr_*std_radr_,                       0,                     0,
		  	  	  	   	  0, std_radphi_*std_radphi_,                     0,
						  0,                       0, std_radrd_*std_radrd_;

  //create measurement noise covariance matrix
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  S_.fill(0);
  for(int i=0; i<2*n_aug_+1; i++){
	  VectorXd z_diff = Zsig_.col(i) - z_pred_;
	  while(z_diff(1) > M_PI)
		  z_diff(1) -= 2*M_PI;
	  while(z_diff(1) < -M_PI)
		  z_diff(1) += 2*M_PI;

	  S_ += weights_(i) * z_diff * z_diff.transpose();
  }

  S_ += R_;
  cout<< endl << "measurement prediction done" <<endl;
  /////////////////////////
  /* ***UPDATE STATE*** */
  /////////////////////////

  //initial Kalman gain
  MatrixXd K_ = MatrixXd(n_x_, n_z_);

  //initial cross-correlation matrix
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc_.fill(0);
  for(int i=0; i<2*n_aug_+1; i++){
	  VectorXd z_diff = Zsig_.col(i) - z_pred_;
	  while(z_diff(1) > M_PI)
		  z_diff(1) -= 2. * M_PI;
	  while(z_diff(1) < -M_PI)
		  z_diff(1) += 2. * M_PI;

	  VectorXd x_diff = Xsig_pred_.col(i) - x_;
	  while(x_diff(3) > M_PI)
		  x_diff(3) -= 2. * M_PI;
	  while(x_diff(3) < -M_PI)
		  x_diff(3) += 2. * M_PI;

	  Tc_ += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  K_ = Tc_ * S_.inverse();

  VectorXd z_diff = z_ - z_pred_;
  while(z_diff(1) > M_PI)
	  z_diff(1) -= 2. * M_PI;
  while(z_diff(1) < -M_PI)
	  z_diff(1) += 2. * M_PI;
  //update state mean and covariance matrix
  x_ += K_ * z_diff;

  P_ -= K_ * S_ * K_.transpose();

  cout << "Update radar x: \n" << x_ << endl;
  cout << "Update radar P: \n" << P_ << endl;
}
