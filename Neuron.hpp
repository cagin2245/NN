#ifndef NEURON_H
#define NEURON_H

#include "main_header.hpp"
#include "Edge.hpp"
#include "functions.hpp"
#include <string>


enum ActivationFunction
{
    LINEAR,
    SIGMOID,
    RELU
};

class Neuron
{
    public:

    Neuron(int _id_neuron, Layer * layer, ActivationFunction function = LINEAR, bool isBiased = false);

    ~Neuron();
    void trigger();
    double in();
    double out();
        double outDerivative();
        double outRaw();
    void clean();
    void addAccumulated(double v);
        void addNext(Neuron * n);
        void addPrevious(Edge * e);
    int getNeuronId() const;

    void setAccumulated(double v);
    void alterWeights(const vector<double>& weights);
        vector<double> getWeights();
        void randomizeAllWeights(double abs_value);
    string toString();
        void shiftWeights(float range);
        void shiftBackWeights(const vector<double>& range);
        vector<double> getBackpropagationShifts(const vector<double>& target);
        bool isBiased() const;
    private:
        Layer * _layer = NULL;
        int _id_neuron = 0;
        double _accumulated = 0.0;

            double _treshold = 0.0;
            vector<Edge *> _next;
            vector<Edge * > _previous;
            ActivationFunction _activation_function;
            bool _is_bias = false;
            




};






#endif