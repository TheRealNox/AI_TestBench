#pragma once

#include "Connection.h"
#include "Misc.h"

class Neuron
{
private:
	unsigned int _index;
	double _outputValue;
	double _gradient;

	// overall learning rate
	// 0.0 - slow learner
	// 0.2 - medium learner
	// 1.0 - reckless learner
	double _eta;

	// multiplier of the last weight change (momentum)
	// 0.0 - no momentum
	// 0.5 - moderate momentum
	double _alpha;
	std::vector<Connection> _outputWeights;

public:
	Neuron(unsigned int numberOfOutputs, unsigned int index, double eta = 0.15, double alpha = 0.5);
	~Neuron();
private:
	Neuron() = delete;

public:
	const double getWeightForIndex(const int index) const;
	const double getWeightDeltaForIndex(const int index) const;

	const double getOutputValue() const;
	void setOutputValue(const double & value);

	const double getGradient() const;

	double sumDOW(const Layer & layer);

	void feedForward(const Layer & previousLayer);

	void calculateOutputGradients(double targetValue);
	void calculateHiddenGradients(const Layer & nextLayer);

	void updateInputWeights(Layer & previousLayer);

private:
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);
};

