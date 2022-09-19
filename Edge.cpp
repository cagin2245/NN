#include "Edge.hpp"


Edge::Edge(Neuron * n, Neuron * nb, double w): _n(n), _nb(nb), _w(w)
{

}

Neuron * Edge::neuron() const
{
    return _n;
}
Neuron * Edge::neuronb() const
{
    return _nb;
}
double Edge::weight() const
{
    return _w;
}

void Edge::propogate(double neuron_output)
{
    neuron()->addAccumulated(neuron_output * weight());
}

void Edge::alterWeight(double w)
{
    _w = w;
}

void Edge::shiftWeight(double dw)
{
    dw *= LEARNING_RATE;
    _w += dw;
    _last_shift = dw;
}
double Edge::getLastShift()const 
{
    return _last_shift;
}

double Edge::backPropogationMemory()const
{
    return _backpropagation_memory;
}

void Edge::setBackPropogationMemory(double value) 
{
    _backpropagation_memory = value;
}