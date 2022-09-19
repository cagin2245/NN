#ifndef FUNCTIONS_H
#define FUNCTIONS_H


#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <string>
#include <cmath>

using namespace std;

// Sigmoid Function
double sigmoid(double x);
double sigmoid_derivative(double x );

// ReLu Function
double relu(double x);
double relu_derivative(double x );

// Random number generator
double random(double low, double high);




#endif