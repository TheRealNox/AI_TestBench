#pragma once
class Connection
{
private:
	double _weight;
	double _delta;

public:
	Connection();

public:
	const double getWeight() const;
	const double getDelta() const;
};

