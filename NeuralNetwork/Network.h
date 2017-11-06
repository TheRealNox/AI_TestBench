#pragma once

#include "Neuron.h"
#include "Misc.h"

class Network
{
private:
	std::vector<Layer> _layers;
	double _errors;
	double _recentAverageError;
	double _recentAverageSmoothingFactor;

public:
	Network(const std::vector<unsigned int> & topology);
	~Network();
private:
	Network() = delete;

public:
	void feedForward(const std::vector<double> & inputs);
	void backPropagation(const std::vector<double> & targets);
	const std::vector<double> getResults() const;
};

