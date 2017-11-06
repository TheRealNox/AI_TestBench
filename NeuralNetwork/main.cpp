// main.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <vector>
#include "Network.h"

int main()
{
	std::vector<double> inputValues;
	std::vector<double> targetValues;
	std::vector<double> resultValues;

	std::vector<unsigned int> topo = { 3, 2, 1 };

	Network myFirstNN(topo);

	myFirstNN.feedForward(inputValues);
	myFirstNN.backPropagation(inputValues);
	resultValues = myFirstNN.getResults();

    return 0;
}

