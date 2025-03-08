#include <iostream>
#include <iomanip>
#include "inference.h"
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <string>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <format>
#include "model.h"

void renderTrafficLight(sf::RenderWindow& window, Model& app) {
	window.clear(sf::Color::Black);

	sf::Text text(app.font);
	text.setString(std::format("vehicle points: {}", app.vehicle_points));
	text.setCharacterSize(24);
	window.draw(text);

	sf::RectangleShape green;
	green.setPosition({0, 50});
	green.setSize(sf::Vector2f(300, 200));
	green.setOutlineThickness(10);
	green.setOutlineColor(sf::Color::Black);
	green.setFillColor(sf::Color::Green);

	sf::RectangleShape yellow;
	yellow.setPosition({0, 250});
	yellow.setSize(sf::Vector2f(300, 200));
	yellow.setOutlineColor(sf::Color::Black);
	yellow.setOutlineThickness(10);
	yellow.setFillColor(sf::Color::Yellow);

	sf::RectangleShape red;
	red.setPosition({0, 450});
	red.setSize(sf::Vector2f(300, 200));
	red.setOutlineColor(sf::Color::Black);
	red.setOutlineThickness(10);
	red.setFillColor(sf::Color::Red);

	switch (app.traffic_light) {
		case Green:
			window.draw(green);
			break;
		case Yellow:
			window.draw(yellow);
			break;
		case Red:
			window.draw(red);
			break;
	}

	window.display();
}

void Detector(YOLO_V8*& p, sf::RenderWindow& traffic_light_window, Model& app) {
	cv::Mat frame;
	cv::VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API

	cap.open(std::string(ASSETS) + "/video.mp4", apiID);

	// check if we succeeded
	if (!cap.isOpened()) {
		std::cerr << "ERROR! Unable to open camera\n";
	}

	for (;;) {
		cap.read(frame);
		// frame = cv::imread(std::string(ASSETS) + "/image.jpg");

		if (frame.empty()) {
			std::cerr << "Error: frame empty" << std::endl;
			return;
		}


		std::vector<DL_RESULT> res;
		p->RunSession(frame, res);

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
			std::string label = p->classes[re.classId] + " " +
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

		// check all the window's events that were triggered since the last iteration of the loop
		while (const std::optional event = traffic_light_window.pollEvent())
		{
			// "close requested" event: we close the window
			if (event->is<sf::Event::Closed>())
				traffic_light_window.close();
		}

		app.vehicle_points =
			(app.automobiles * 5) +
			(app.bikes * 5) +
			(app.buses * 10) +
			(app.cars * 5) +
			(app.trucks * 5);

		if (app.vehicle_points < 30) {
			app.traffic_light = Red;
		} else {
			app.traffic_light = Green;
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

void Classifier(YOLO_V8*& p)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(0, 255);
	std::string img_path = std::string(ASSETS) + "/image.jpg";
	//std::cout << img_path << std::endl;
	cv::Mat img = cv::imread(img_path);
	std::vector<DL_RESULT> res;
	char* ret = p->RunSession(img, res);

	float positionY = 50;
	for (int i = 0; i < res.size(); i++)
	{
		int r = dis(gen);
		int g = dis(gen);
		int b = dis(gen);
		cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
		cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
		positionY += 50;
	}

	cv::imshow("TEST_CLS", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
	//cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
}

int ReadCocoYaml(YOLO_V8*& p) {
	// Open the YAML file
	std::ifstream file(std::string(ASSETS) + "/data.yaml");
	if (!file.is_open())
	{
		std::cerr << "Failed to open file" << std::endl;
		return 1;
	}

	std::ifstream classes_file(std::string(ASSETS) + "/classes.txt");
	std::cout << std::string(ASSETS) + "/classes.txt" << std::endl;
	// Extract the names
	std::vector<std::string> classes;
	std::string line;
	while (std::getline(classes_file, line)) {
		classes.push_back(line);
	}

	p->classes = classes;
	return 0;
}


void run(sf::RenderWindow& traffic_light_window, Model& app)
{
	YOLO_V8* yoloDetector = new YOLO_V8;
	ReadCocoYaml(yoloDetector);
	DL_INIT_PARAM params;
	params.rectConfidenceThreshold = 0.5;
	params.iouThreshold = 0.45;
	params.modelPath = std::string(ASSETS) + "/best.onnx";
	params.imgSize = { 640, 640 };
#ifdef USE_CUDA
	params.cudaEnable = true;

	// GPU FP32 inference
	params.modelType = YOLO_DETECT_V8;
	// GPU FP16 inference
	//Note: change fp16 onnx model
	//params.modelType = YOLO_DETECT_V8_HALF;

#else
	// CPU inference
	params.modelType = YOLO_DETECT_V8;
	params.cudaEnable = false;

#endif
	std::cout << params.modelType << std::endl;
	yoloDetector->CreateSession(params);
	Detector(yoloDetector, traffic_light_window, app);
}


void ClsTest()
{
	YOLO_V8* yoloDetector = new YOLO_V8;
	std::string model_path = std::string(ASSETS) + "/best.onnx";
	ReadCocoYaml(yoloDetector);
	DL_INIT_PARAM params{ model_path, YOLO_CLS, {640, 640} };
	yoloDetector->CreateSession(params);
	Classifier(yoloDetector);
}

int main()
{
	Model app;
	sf::RenderWindow traffic_light_window(sf::VideoMode({300, 650}), "My window");
	app.traffic_light = Green;
	sf::Font font(std::string(ASSETS) + "/font.ttf");
	app.font = font;

	run(traffic_light_window, app);

	return 0;
}
