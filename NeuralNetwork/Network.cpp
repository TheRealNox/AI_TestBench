#include "Network.h"
#include <iostream>
#include <cassert>

Network::Network(const std::vector<unsigned int> & topology) :
	_errors(0.0),
	_recentAverageError(0.0),
	_recentAverageSmoothingFactor(0.0)
{
	unsigned int depth = topology.size();

	for (unsigned int layerIndex = 0; layerIndex < depth; ++layerIndex)
	{
		this->_layers.push_back(Layer());

		//No output for the bottom layer
		unsigned int numberOfOutputs = layerIndex == topology.size() - 1 ? 0 : topology[layerIndex + 1];
		//Now let's add the neurons
		//We went <= because we need a bias per layer
		for (unsigned int neuronIndex = 0; neuronIndex <= topology[layerIndex]; ++neuronIndex)
		{
			this->_layers.back().push_back(Neuron(numberOfOutputs, neuronIndex));
			std::cout << "A neuron has born!" << std::endl;
		}
	}

	// Force the bias node's output value to 1.0. It's the last neuron created above.
	this->_layers.back().back().setOutputValue(1.0);
}

Network::~Network()
{
}

void Network::feedForward(const std::vector<double>& inputs)
{
	//Don't forget the bias neuron
	assert(inputs.size() == this->_layers[0].size() - 1);

	//Assign (latch) the input values into the input neurons (top layer)
	for (unsigned int i = 0; i < inputs.size(); ++i)
	{
		this->_layers[0][i].setOutputValue(inputs[i]);
	}

	//And now let's propagate but skip the first input layer
	for (unsigned int layerIndex = 1; layerIndex < this->_layers.size(); ++layerIndex)
	{
		Layer & previousLayer = this->_layers[layerIndex - 1];
		for (unsigned int neuronIndex = 0; neuronIndex < this->_layers[layerIndex].size() - 1; ++neuronIndex)
			this->_layers[layerIndex][neuronIndex].feedForward(previousLayer);
	}
}

void Network::backPropagation(const std::vector<double>& targets)
{
	// Here we need to calculate the overall net error (RMS of the ouput neuron errors)
	// That's what we try to minize
	Layer & outputLayer = this->_layers.back();

	this->_errors = 0.;

	for (unsigned int n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targets[n] - outputLayer[n].getOutputValue();
		this->_errors += delta * delta;
	}
	

	// get the average
	this->_errors /= outputLayer.size() - 1;
	//and here is the RMS
	this->_errors = std::sqrt(this->_errors);

	// Let's have a nice recent average measurement
	this->_recentAverageError = (this->_recentAverageError * this->_recentAverageSmoothingFactor + this->_errors) / (this->_recentAverageSmoothingFactor + 1.0);

	// Calculate ouput layer gradients
	for (unsigned int neuronIx = 0; neuronIx < outputLayer.size() - 1; ++neuronIx)
	{
		outputLayer[neuronIx].calculateOutputGradients(targets[neuronIx]);
	}

	// Calculate gradients on hidden layers (not input nor output)
	for (unsigned int layerIx = this->_layers.size() - 2; layerIx > 0; --layerIx)
	{
		Layer & hiddenLayer = this->_layers[layerIx];
		Layer & nextLayer = this->_layers[layerIx + 1];

		for (unsigned int neuronIx = 0; hiddenLayer.size(); ++neuronIx)
		{
			hiddenLayer[neuronIx].calculateHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layers
	// we update the connection weights

	for (unsigned int layerIx = this->_layers.size() - 1; layerIx > 0; --layerIx)
	{
		Layer & currentLayer = this->_layers[layerIx];
		Layer & previousLayer = this->_layers[layerIx - 1];

		for (unsigned int neuronIx = 0; currentLayer.size() - 1; ++neuronIx)
		{
			currentLayer[neuronIx].updateInputWeights(previousLayer);
		}
	}

}

const std::vector<double> Network::getResults() const
{
	std::vector<double> results;

	for (unsigned int neuronIx = 0; neuronIx < this->_layers.size() - 1; ++neuronIx)
	{
		results.push_back(this->_layers.back()[neuronIx].getOutputValue());
	}

	return results;
}
