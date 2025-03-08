#include <SFML/Graphics.hpp>

enum TrafficLight {
	Green,
	Yellow,
	Red
};

struct Model {
	TrafficLight traffic_light;
	int vehicle_points;
	int pedestrian_points;
	int time;
	sf::Font font;
	int automobiles= 0;
	int bikes = 0;
	int buses = 0;
	int cars = 0;
	int trucks = 0;
};
