#include "Connection.h"

#include <random>

//TODO: We could come up with a nice RandomNumberGenerator
static std::random_device rd;  //Will be used to obtain a seed for the random number engine
static std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
static std::uniform_real_distribution<> dis(0.0, 1.0);

Connection::Connection() : _delta(0.0)
{
	this->_weight = dis(gen);
}

const double Connection::getWeight() const
{
	return this->_weight;
}

const double Connection::getDelta() const
{
	return this->_delta;
}
