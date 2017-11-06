#include "Neuron.h"

#include <iostream>

Neuron::Neuron(unsigned int numberOfOutputs, unsigned int index, double eta, double alpha) :
	_index(index), _outputValue(0.0), _gradient(0.0), _eta(eta), _alpha(alpha)
{
	for (unsigned int connexions = 0; connexions < numberOfOutputs; ++connexions)
	{
		this->_outputWeights.push_back(Connection());
		std::cout << "New connection with a weight of: " << this->_outputWeights.back().getWeight() << std::endl;
	}
}

Neuron::~Neuron()
{
}

const double Neuron::getWeightForIndex(const int index) const
{
	return this->_outputWeights[index].getWeight();
}

const double Neuron::getWeightDeltaForIndex(const int index) const
{
	return this->_outputWeights[index].getDelta();
}

const double Neuron::getOutputValue() const
{
	return this->_outputValue;
}

void Neuron::setOutputValue(const double & value)
{
	this->_outputValue = value;
}

const double Neuron::getGradient() const
{
	return this->_gradient;
}

double Neuron::sumDOW(const Layer & layer)
{
	double sum = 0.0;

	//Sum our contribution of the errors at the nodes we feed

	for (unsigned int neuronIx = 0; neuronIx < layer.size() - 1; ++neuronIx)
	{
		sum += this->_outputWeights[neuronIx].getWeight() * layer[neuronIx].getGradient();
	}

	return sum;
}

void Neuron::feedForward(const Layer & previousLayer)
{
	double sum = 0.;

	for (unsigned int neuronIx = 0; neuronIx < previousLayer.size(); ++neuronIx)
		sum += previousLayer[neuronIx].getOutputValue() * previousLayer[neuronIx].getWeightForIndex(this->_index);

	//Activation Function
	this->_outputValue = Neuron::activationFunction(sum);
}

void Neuron::calculateOutputGradients(double targetValue)
{
	double delta = targetValue - this->_outputValue;
	this->_gradient = delta * Neuron::activationFunctionDerivative(this->_outputValue);
}

void Neuron::calculateHiddenGradients(const Layer & nextLayer)
{
	double dow = this->sumDOW(nextLayer);
	this->_gradient = dow * Neuron::activationFunctionDerivative(this->_outputValue);

}

void Neuron::updateInputWeights(Layer & previousLayer)
{
	// The weights to be updated are in the Connection class
	// in the neurons in the preceding layer

	for (unsigned int neuronIx = 0; previousLayer.size() - 1; ++neuronIx)
	{
		Neuron & neuron = previousLayer[neuronIx];
		double oldDeltaWeight = neuron.getWeightDeltaForIndex(this->_index);

		double newDeltaWieght = \
			//Individual input, magnified by the gradient and the train rate
			this->_eta \
			* neuron.getOutputValue() * this->_gradient \
			// Also add momemtum = a fraction of the previous wieght
			* this->_alpha \
			* oldDeltaWeight;
	}
}

double Neuron::activationFunction(double x)
{
	return std::tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
	return 1.0 - x * x;
}
