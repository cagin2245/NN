#ifndef EDGE_H
#define EDGE_H

#include "main_header.hpp"

extern double LEARNING_RATE;
extern double SHIFT_LIMIT;


// İki nöron arası köşe
class Edge
{
    public:
    Edge(Neuron * n,Neuron * start, double w); // -> w = weight
        Neuron * neuron() const;
        Neuron * neuronb() const;
    double weight() const;
    void propogate(double neuron_output);
    void alterWeight(double w);
        void shiftWeight(double dw);
        double getLastShift()const;
        double backPropogationMemory()const;
        void setBackPropogationMemory(double value);

    private:
        Neuron * _n = nullptr;
        Neuron * _nb = nullptr;
        double _w = 0.0;
        double _last_shift = 0;
        double _backpropagation_memory;
};


#endif