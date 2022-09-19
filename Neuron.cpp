#include "Neuron.hpp"
#include <algorithm>


Neuron::Neuron(int id_neuron, Layer * layer, ActivationFunction function, bool isBiased)
: _id_neuron(id_neuron), _layer(layer), _activation_function(function),_is_bias(isBiased)
{

}

Neuron::~Neuron()
{
    for(Edge * e : _next)
        delete e;

}

void Neuron::trigger()
{
    for(Edge * e : _next)
    {
        cout << this->getNeuronId() << "->" << e->neuron()->getNeuronId() << " " << out() << "*" << e->weight() << "=" << out()*e->weight() << "("<< outRaw() << ")" << endl;
        e->propogate(out());
    }
}

double Neuron::in()
{
    return _accumulated;
}

double Neuron::out()
{
    if(_is_bias)
        return 1;
    if(_layer->getType()== LayerType::INPUT)
        return outRaw();
    
    if(_activation_function == ActivationFunction::LINEAR)
        return _accumulated;
    if(_activation_function == ActivationFunction::RELU)
        return relu(_accumulated);
    if(_activation_function == ActivationFunction::SIGMOID)
        return sigmoid(_accumulated);
    return outRaw();

}
double Neuron::outDerivative()
{
     if(_activation_function == ActivationFunction::LINEAR)
        return _accumulated;
    if(_activation_function == ActivationFunction::RELU)
        return relu_derivative(_accumulated);
    if(_activation_function == ActivationFunction::SIGMOID)
        return sigmoid_derivative(_accumulated);
    return _accumulated;

}

double Neuron::outRaw()
{
    return _accumulated;
}
void Neuron::clean()
{
    setAccumulated(0);
}
void Neuron::addAccumulated(double v)
{
    setAccumulated(_accumulated + v);
}
void Neuron::addNext(Neuron * n)
{
    _next.push_back(new Edge(n,this,random(-5,5)));
        n->addPrevious(_next.back());
}

void Neuron::addPrevious(Edge * e)
{
    _previous.push_back(e);
}
int Neuron::getNeuronId() const
{
    return _id_neuron;
}
void Neuron::setAccumulated(double v)
{
    _accumulated = v;
}
void Neuron::alterWeights(const vector<double>& weights)
{
    for(size_t i_edge = 0;i_edge < weights.size(); ++i_edge)
        _next[i_edge]->alterWeight(weights[i_edge]);
}

vector<double> Neuron::getWeights()
{
    vector<double> w;
    for(size_t i_edge = 0; i_edge<_next.size();++i_edge)
    {
        w.push_back(_next[i_edge]->weight());

    }
    return std::move(w);
}
void Neuron::randomizeAllWeights(double abs_value)
{
    for(Edge * e: _next)
        e->alterWeight(random(-abs_value,abs_value));
}
string Neuron::toString()
{
    string weights;
    for(Edge * e : _next)
        weights.append(to_string(e->weight())+ ", ");
    string str = "[" + to_string(_layer->getId()) + "," + to_string(_id_neuron) + "]" + "("+ weights + ")";
    return str;
}

void Neuron::shiftWeights(float range)
{
    for(Edge * e : _next)
    {
        e->alterWeight(e->weight() + random(-range,range));
    }
}


// Gradient descent
vector<double> Neuron::getBackpropagationShifts(const vector<double>& target)
{
    vector<double> dw(_previous.size(),0);
    if(_layer->getType()== LayerType::OUTPUT)
    {
        double d0 = out();
        double d1 = out() - target[this->getNeuronId()];
        double d2 = outDerivative();
        for(size_t i = 0; i < _previous.size();++i)
        {
            dw[i] = (-d1 *d2 * _previous[i]->neuronb()->out());
            _previous[i]->setBackPropogationMemory(d1 * d2);
        }
        //cout << _layer->getId() << " " << d1 << " " << d2 << " " << d3 << " " << d1*d2*d3 << endl;

    }
    else 
    {
        float d = 0;
        for(size_t i = 0; i < _next.size(); i++)
            d += _next[i]->backPropogationMemory() * _next [i]->weight();
    }
    return dw;
}
bool Neuron::isBiased() const
{
    return _is_bias;
}