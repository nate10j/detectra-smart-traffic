#include <chrono>
#include <cmath> #include <iostream>
#include <iomanip>
#include "inference.h"
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <string>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <format>
#include <thread>
#include <vector>
#include "model.h"

void renderTrafficLight(sf::RenderWindow& window, Model& app) {
	window.clear(sf::Color::Black);

	sf::Text text(app.font);
	text.setString(std::format("vehicle score: {}", app.vehicle_points));
	text.setCharacterSize(24);
	window.draw(text);

	sf::Text text2(app.font);
	text2.setPosition({0, 30});
	text2.setString(std::format("pedestrian score: {}", app.pedestrian_points));
	text2.setCharacterSize(24);
	window.draw(text2);

	sf::RectangleShape green;
	green.setPosition({0, 75});
	green.setSize(sf::Vector2f(300, 200));
	green.setOutlineThickness(10);
	green.setOutlineColor(sf::Color::Black);
	green.setFillColor(sf::Color::Green);

	sf::RectangleShape yellow;
	yellow.setPosition({0, 275});
	yellow.setSize(sf::Vector2f(300, 200));
	yellow.setOutlineColor(sf::Color::Black);
	yellow.setOutlineThickness(10);
	yellow.setFillColor(sf::Color::Yellow);

	sf::RectangleShape red;
	red.setPosition({0, 475});
	red.setSize(sf::Vector2f(300, 200));
	red.setOutlineColor(sf::Color::Black);
	red.setOutlineThickness(10);
	red.setFillColor(sf::Color::Red);

	sf::Sprite pedestrian_light(app.texture);
	pedestrian_light.setPosition({220, 100});
	
	switch (app.traffic_light) {
		case Green:
			pedestrian_light.setTextureRect(sf::IntRect({20, 20}, {380, 225}));
			window.draw(pedestrian_light);
			window.draw(green);
			break;
		case Yellow:
			window.draw(yellow);
			break;
		case Red:
			pedestrian_light.setTextureRect(sf::IntRect({20, 250}, {380, 425}));
			window.draw(pedestrian_light);
			window.draw(red);
			break;
	}

	window.display();
}

void Detector(YOLO_V8*& car_p, YOLO_V8*& pedestrian_p, sf::RenderWindow& traffic_light_window, Model& app) {
	cv::Mat frame;
	cv::Mat framePedestrian;
	cv::VideoCapture cap;
	cv::VideoCapture capPedestrian;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API

	cap.open(std::string(ASSETS) + "/video.mp4", apiID);
	capPedestrian.open(std::string(ASSETS) + "/pedestrian.mp4", apiID);

	// check if we succeeded
	if (!cap.isOpened()) {
		std::cerr << "ERROR! Unable to open camera for vehicles\n";
	}
	if (!capPedestrian.isOpened()) {
		std::cerr << "ERROR! Unable to open camera for pedestrians\n";
	}

	for (;;) {
		cap.read(frame);
		capPedestrian.read(framePedestrian);
		// frame = cv::imread(std::string(ASSETS) + "/image.jpg");

		// vehicles
		if (frame.empty()) {
			std::cerr << "Error: frame empty" << std::endl;
			return;
		}


		std::vector<DL_RESULT> res;
		car_p->RunSession(frame, res);

		if (res.empty()) {
			std::cout << "no objects detected" << std::endl;
		} else {
			std::cout << res.size() << std::endl;
		}

		app.automobiles = 0;
		app.bikes = 0;
		app.buses = 0;
		app.cars = 0;
		app.trucks = 0;

		for (auto& re : res)
		{
			cv::RNG rng(cv::getTickCount());
			cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

			cv::rectangle(frame, re.box, color, 3);

			float confidence = floor(100 * re.confidence) / 100;
			std::cout << std::fixed << std::setprecision(2);
			std::string label = car_p->classes[re.classId] + " " +
				std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

			switch (re.classId) {
				case 1:
					app.automobiles++;
					break;
				case 2:
					app.bikes++;
					break;
				case 3:
					app.buses++;
					break;
				case 4:
					app.cars++;
					break;
				case 5:
					app.trucks++;
					break;
			}

			cv::rectangle(
				frame,
				cv::Point(re.box.x, re.box.y - 25),
				cv::Point(re.box.x + label.length() * 15, re.box.y),
				color,
				cv::FILLED
			);

			cv::putText(
				frame,
				label,
				cv::Point(re.box.x, re.box.y - 5),
				cv::FONT_HERSHEY_SIMPLEX,
				0.75,
				cv::Scalar(0, 0, 0),
				2
			);


		}

		cv::imshow("Live",frame);

		// pedestrian detect
		
		app.pedestrians = 0;

		cv::Mat img = framePedestrian.clone();

		std::vector<DL_RESULT> pedestrian_res;
		pedestrian_p->RunSession(img, pedestrian_res);

		if (pedestrian_res.empty()) {
			std::cout << "no objects detected" << std::endl;
		} else {
			std::cout << pedestrian_res.size() << std::endl;
		}

		for (auto& re : pedestrian_res)
		{
			app.pedestrians++;
			cv::RNG rng(cv::getTickCount());
			cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

			cv::rectangle(img, re.box, color, 3);

			float confidence = floor(100 * re.confidence) / 100;
			std::cout << std::fixed << std::setprecision(2);
			std::string label = pedestrian_p->classes[re.classId] + " " +
				std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

			cv::rectangle(
				img,
				cv::Point(re.box.x, re.box.y - 25),
				cv::Point(re.box.x + label.length() * 15, re.box.y),
				color,
				cv::FILLED
			);

			cv::putText(
				img,
				label,
				cv::Point(re.box.x, re.box.y - 5),
				cv::FONT_HERSHEY_SIMPLEX,
				0.75,
				cv::Scalar(0, 0, 0),
				2
			);
		}


		/// Show
		resize(img, img, cv::Size(854, 480));
		imshow("detected person", img);

		// check all the window's events that were triggered since the last iteration of the loop
		while (const std::optional event = traffic_light_window.pollEvent())
		{
			// "close requested" event: we close the window
			if (event->is<sf::Event::Closed>())
				traffic_light_window.close();
		}

		renderTrafficLight(traffic_light_window, app);

		if (cv::waitKey(1) != -1)
		{
			std::cout << "finished by user\n";
			break;
		}
	}
	cv::destroyAllWindows();
}

void run(sf::RenderWindow& traffic_light_window, Model& app)
{
	YOLO_V8* carsDetector = new YOLO_V8;
	YOLO_V8* pedestrianDetector = new YOLO_V8;
	carsDetector->classes = std::vector<std::string> {"auto", "bike", "bus", "car", "truck"};
	pedestrianDetector->classes = std::vector<std::string> {"person", "paralyzed", "person"};
	DL_INIT_PARAM cars_params;
	DL_INIT_PARAM pedestrian_params;
	cars_params.rectConfidenceThreshold = 0.37;
	cars_params.iouThreshold = 0.45;
	cars_params.modelPath = std::string(ASSETS) + "/car.onnx";
	cars_params.imgSize = { 640, 640 };

	pedestrian_params.rectConfidenceThreshold = 0.02;
	pedestrian_params.iouThreshold = 0.45;
	pedestrian_params.modelPath = std::string(ASSETS) + "/pedestrian.onnx";
	pedestrian_params.imgSize = { 640, 640 };
#ifdef USE_CUDA
	cars_params.cudaEnable = true;

	// GPU FP32 inference
	cars_params.modelType = YOLO_DETECT_V8;
	// GPU FP16 inference
	//Note: change fp16 onnx model
	//cars_params.modelType = YOLO_DETECT_V8_HALF;

#else
	// CPU inference
	cars_params.modelType = YOLO_DETECT_V8;
	cars_params.cudaEnable = false;

	pedestrian_params.modelType = YOLO_DETECT_V8;
	pedestrian_params.cudaEnable = false;
#endif
	std::cout << cars_params.modelType << std::endl;
	carsDetector->CreateSession(cars_params);
	pedestrianDetector->CreateSession(pedestrian_params);
	Detector(carsDetector, pedestrianDetector, traffic_light_window, app);
}

void traffic_light_sequence(Model* app) {
	// waiting
	std::this_thread::sleep_for(std::chrono::seconds(3));

	while (true) {
		sf::Clock pedestrianTimer;
		sf::Clock carTimer;
		int waitingTime;

		app->vehicle_points =
			(app->automobiles * 5) +
			(app->bikes * 5) +
			(app->buses * 10) +
			(app->cars * 5) +
			(app->trucks * 5);

		app->pedestrian_points = app->pedestrians * 10;

		if (abs(app->vehicle_points - app->pedestrian_points) <= 5) {
			// around the same range
			waitingTime = 15;
		} else if (app->pedestrian_points > app->vehicle_points) {
			waitingTime = 12;
		} else if (app->pedestrian_points < app->vehicle_points) {
			waitingTime = 25;
		} else if (app->pedestrians == 0) {
			waitingTime = 0;
		}

		app->traffic_light = Green;
		std::this_thread::sleep_for(std::chrono::seconds(waitingTime));
		app->traffic_light = Yellow;
		std::this_thread::sleep_for(std::chrono::seconds(2));

		app->vehicle_points =
			(app->automobiles * 5) +
			(app->bikes * 5) +
			(app->buses * 10) +
			(app->cars * 5) +
			(app->trucks * 5);

		app->pedestrian_points = app->pedestrians * 10;

		if (abs(app->vehicle_points - app->pedestrian_points) <= 5) {
			// around the same range
			waitingTime = 15;
		} else if (app->pedestrian_points > app->vehicle_points) {
			waitingTime = 25;
		} else if (app->pedestrian_points < app->vehicle_points) {
			waitingTime = 12;
		} else if (app->cars == 0) {
			waitingTime = 0;
		}

		app->traffic_light = Red;
		std::this_thread::sleep_for(std::chrono::seconds(waitingTime));

		app->traffic_light = Yellow;
		std::this_thread::sleep_for(std::chrono::seconds(2));
	}
}

int main()
{
	Model app;
	sf::RenderWindow traffic_light_window(sf::VideoMode({600, 675}), "My window");
	app.traffic_light = Green;
	sf::Font font(std::string(ASSETS) + "/font.ttf");
	app.font = font;
	sf::Texture texture(std::string(ASSETS) + "/pedestrian-light.jpg");
	app.texture = texture;

	std::thread traffic_thread(traffic_light_sequence, &app);

	run(traffic_light_window, app);

	return 0;
}
