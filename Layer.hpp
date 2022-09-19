#ifndef LAYER_H
#define LAYER_H


#include "Neuron.hpp"
#include "main_header.hpp"
#include <unordered_map>

enum LayerType
{
    STANDARD = 0, // Standard Layer: Fully connnected perceptrons
    OUTPUT,       // Output: No bias neuron
    INPUT,        // Input: Standard Input (output of neuron is outputRaw())
    SOFTMAX       // K-Class Classification Layer
};

enum ActivationFunction;

class Layer
{

};



#endif