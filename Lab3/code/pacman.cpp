#include <sstream>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <iostream>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <random>
#include <utility>
#include <map>
#include "position.h"

#define GHOST_COUNT 4 

using namespace sf;
using namespace std;

vector<CircleShape> m_ballPosition;
vector<RectangleShape> m_wallPosition;

VideoMode vm(641, 728);
RenderWindow window(vm, "Pacman!!!", Style::Default);

Keyboard::Key ghostMovementDirection[GHOST_COUNT] = { Keyboard::Unknown, Keyboard::Unknown, Keyboard::Unknown, Keyboard::Unknown };
Keyboard::Key pacmanMovementDirection = Keyboard::Unknown;

Sprite spritePacman;
vector<Sprite> spriteGhost(4);

void addCoinsToVector();
void addWallsToVector();
void loadCoinsRepeat();
void loadWalls();
void moveGhosts();
Keyboard::Key getRandomDirection();
void movePacman(Keyboard::Key);
bool isValidPosition(Vector2f, Sprite);
bool doGhostPacmanIntersect();

int tempDir[4][4];

long long int score = 0;
bool powerUp = false;


enum  DIRECTION
{
	UP = 0,
	DOWN,
	LEFT,
	RIGHT
};



long long int globalI = 0, myLeft = 0, myRight = 0, myUp = 0, myDown = 0;

static bool is_seeded = false;
static mt19937 generator;
uniform_int_distribution<int> distribution(0, 3);


int main()
{

	// srand(time(0));
	// Seed once
	if (!is_seeded) {
		random_device rd;
		generator.seed(rd());
		is_seeded = true;
	}

	// srand(time(0));
	// Create a texture to hold a graphic on the GPU
	Texture textureBackground;

	textureBackground.loadFromFile("./graphics/pac-man.bmp");

	Sprite spriteBackground;

	spriteBackground.setTexture(textureBackground);
	spriteBackground.setPosition(0, 0);

	Texture texturePacman;
	texturePacman.loadFromFile("./graphics/pacman.bmp");
	spritePacman.setTexture(texturePacman);
	spritePacman.setPosition(300, 490);


	Texture textureGhost0;
	textureGhost0.loadFromFile("./graphics/red_ghosts.bmp");
	spriteGhost[0].setTexture(textureGhost0);
	
	Texture textureGhost1;
	textureGhost1.loadFromFile("./graphics/orange_ghost.bmp");
	spriteGhost[1].setTexture(textureGhost1);

	Texture textureGhost2;
	textureGhost2.loadFromFile("./graphics/pink_ghost.bmp");
	spriteGhost[2].setTexture(textureGhost2);

	Texture textureGhost3;
	textureGhost3.loadFromFile("./graphics/blue_ghost.bmp");
	spriteGhost[3].setTexture(textureGhost3);

	for (int i = 0; i < GHOST_COUNT; i++) {
		spriteGhost[i].setPosition(302, 245);
	}



	Texture textureMaze;
	textureMaze.loadFromFile("./graphics/maze.bmp");
	Sprite spriteMaze;
	spriteMaze.setTexture(textureMaze);
	spriteMaze.setPosition(0, 0);

	sf::Text messageText;
	sf::Text scoreText;

	sf::Font font;
	font.loadFromFile("./fonts/KOMIKAP_.ttf");


	messageText.setFont(font);
	scoreText.setFont(font);

	messageText.setString("Press Enter to start!");
	scoreText.setString("Score = 0");

	messageText.setCharacterSize(35);
	scoreText.setCharacterSize(50);

	messageText.setFillColor(Color::White);
	scoreText.setFillColor(Color::White);

	FloatRect textRect = messageText.getLocalBounds();

	messageText.setOrigin(textRect.left +
		textRect.width / 2.0f,
		textRect.top +
		textRect.height / 2.0f);

	messageText.setPosition(320, 360);

	scoreText.setPosition(20, 20);


	bool paused = true;
	bool acceptInput = false;

	addWallsToVector();
	addCoinsToVector();

	while (window.isOpen())
	{
		//cout<<"ball:: "<<m_ballPosition.size()<<endl;
		//cout<<"wall:: "<<m_wallPosition.size()<<endl;

		window.clear();
		score++;
		Event event;
		while (window.pollEvent(event))
		{


			if (event.type == Event::KeyReleased && !paused)
			{
				acceptInput = true;
			}

		}


		if (Keyboard::isKeyPressed(Keyboard::Escape))
		{
			window.close();
		}


		//start the game
		if (Keyboard::isKeyPressed(Keyboard::Return)) {
			paused = false;
			moveGhosts();
		}

		if (Keyboard::isKeyPressed(Keyboard::Left)) {
			movePacman(Keyboard::Left);
		}

		if (Keyboard::isKeyPressed(Keyboard::Right)) {
			movePacman(Keyboard::Right);
		}

		if (Keyboard::isKeyPressed(Keyboard::Up)) {
			movePacman(Keyboard::Up);
		}

		if (Keyboard::isKeyPressed(Keyboard::Down)) {
			movePacman(Keyboard::Down);
		}


		window.draw(spriteMaze);
		if (paused)
		{
			window.draw(spriteBackground);
			window.draw(messageText);
		}


		window.draw(spritePacman);

		loadWalls();

		if (!paused) {
			loadCoinsRepeat();

			moveGhosts();
			movePacman(pacmanMovementDirection);
		}

		for (int i = 0; i < GHOST_COUNT; i++) {
			window.draw(spriteGhost[i]);
		}

		// scoreText.setString("Score = " + to_string(score));
		// window.draw(scoreText);
		window.display();

		if (doGhostPacmanIntersect()) {
			exit(1);
		}
	}

	return 0;
}

void eatCoin() {
	for (auto ballIterator = m_ballPosition.begin(); ballIterator != m_ballPosition.end(); ballIterator++) {
		if (ballIterator->getGlobalBounds().intersects(spritePacman.getGlobalBounds())) {
			// score++;
			m_ballPosition.erase(ballIterator);
			break;
		}
	}
}

bool doGhostPacmanIntersect() {
	for (int i = 0; i < GHOST_COUNT; i++) {
		if (spriteGhost[i].getGlobalBounds().intersects(spritePacman.getGlobalBounds())) {
			return true;
		}
	}

	return false;
}



void movePacman(Keyboard::Key keyPressed) {
	if (keyPressed == Keyboard::Unknown) {
		// keyPressed = pacmanMovementDirection;
	}

	if (keyPressed != Keyboard::Unknown) {
		bool breakLoop = false;
		// while(!breakLoop) {
		Vector2f pacmanCurrentPosition = spritePacman.getPosition();
		switch (keyPressed) {
		case Keyboard::Left:
			pacmanCurrentPosition.x -= 1;

			if (isValidPosition(pacmanCurrentPosition, spritePacman)) {
				spritePacman.setPosition(pacmanCurrentPosition);
				pacmanMovementDirection = Keyboard::Left;
				breakLoop = !breakLoop;
			}
			else {
				// cout<<"here left"<<endl;
				keyPressed = Keyboard::Down;
			}
			break;

		case Keyboard::Right:
			pacmanCurrentPosition.x += 1;

			if (isValidPosition(pacmanCurrentPosition, spritePacman)) {
				spritePacman.setPosition(pacmanCurrentPosition);
				pacmanMovementDirection = Keyboard::Right;
				breakLoop = !breakLoop;
			}
			else {
				// cout<<"here right"<<endl;
				keyPressed = Keyboard::Up;
			}
			break;

		case Keyboard::Up:
			pacmanCurrentPosition.y -= 1;

			if (isValidPosition(pacmanCurrentPosition, spritePacman)) {
				spritePacman.setPosition(pacmanCurrentPosition);
				pacmanMovementDirection = Keyboard::Up;
				breakLoop = !breakLoop;
			}
			else {
				// cout<<"here up"<<endl;
				keyPressed = Keyboard::Left;
			}
			break;

		case Keyboard::Down:
			pacmanCurrentPosition.y += 1;

			if (isValidPosition(pacmanCurrentPosition, spritePacman)) {
				spritePacman.setPosition(pacmanCurrentPosition);
				pacmanMovementDirection = Keyboard::Down;
				breakLoop = !breakLoop;
			}
			else {
				// cout<<"here down"<<endl;
				keyPressed = Keyboard::Right;
			}
			break;
		}//switch

		if (breakLoop) {
			eatCoin();
		}
		// }//while
	}//if
}//movepacman code


bool isValidPosition(Vector2f newPosition, Sprite objectSprite) {

	objectSprite.setPosition(Vector2f(newPosition.x, newPosition.y));
	for (int k = 0; k < m_wallPosition.size(); k++) {
		if (m_wallPosition[k].getGlobalBounds().intersects(objectSprite.getGlobalBounds())) {

			// cout<<"wall:: "<<m_wallPosition[k].getPosition().x<<" "<<m_wallPosition[k].getPosition().y<<endl;


			// for(int i=0; i<GHOST_COUNT; i++) {
			// 	cout<<"dir:: "<<i<<":: "<<tempDir[i][0]<<" "<<tempDir[i][1]<<" "<<tempDir[i][2]<<" "<<tempDir[i][3]<<endl;
			// }

			return false;
		}
	}

	return true;
} // isValidPosition function


void moveGhosts() {

	for (int i = 0; i < GHOST_COUNT; i++) {
		globalI = i;
		if (ghostMovementDirection[i] == Keyboard::Unknown) {
			ghostMovementDirection[i] = getRandomDirection();
		}

		bool breakLoop = false;
		while (!breakLoop) {
			Vector2f ghostCurrentPosition = spriteGhost[i].getPosition();
			switch (ghostMovementDirection[i]) {
			case Keyboard::Left:
				ghostCurrentPosition.x -= 1;

				if (isValidPosition(ghostCurrentPosition, spriteGhost[i])) {
					spriteGhost[i].setPosition(ghostCurrentPosition);
					ghostMovementDirection[i] = Keyboard::Left;
					breakLoop = !breakLoop;
				}
				else {
					ghostMovementDirection[i] = getRandomDirection();
				}
				break;

			case Keyboard::Right:
				ghostCurrentPosition.x += 1;

				if (isValidPosition(ghostCurrentPosition, spriteGhost[i])) {
					spriteGhost[i].setPosition(ghostCurrentPosition);
					ghostMovementDirection[i] = Keyboard::Right;
					breakLoop = !breakLoop;
				}
				else {
					ghostMovementDirection[i] = getRandomDirection();
				}
				break;

			case Keyboard::Up:
				ghostCurrentPosition.y -= 1;

				if (isValidPosition(ghostCurrentPosition, spriteGhost[i])) {
					spriteGhost[i].setPosition(ghostCurrentPosition);
					ghostMovementDirection[i] = Keyboard::Up;
					breakLoop = !breakLoop;
				}
				else {
					ghostMovementDirection[i] = getRandomDirection();
				}
				break;

			case Keyboard::Down:
				ghostCurrentPosition.y += 1;

				if (isValidPosition(ghostCurrentPosition, spriteGhost[i])) {
					spriteGhost[i].setPosition(ghostCurrentPosition);
					ghostMovementDirection[i] = Keyboard::Down;
					breakLoop = !breakLoop;
				}
				else {
					ghostMovementDirection[i] = getRandomDirection();
				}
				break;
			}//switch
		}//while

	} // ghost count for loop
}

void loadCoinsRepeat() {
	for (int c = 0; c < m_ballPosition.size(); c++)
	{
		window.draw(m_ballPosition[c]);
	}
}

void loadWalls() {
	for (int w = 0; w < m_wallPosition.size(); w++)
	{
		window.draw(m_wallPosition[w]);
	}
}

void addCoinsToVector() {
	float k = 0.0;

	for (int i = 0; i < 29; i++)
	{
		float s = 0.0;
		for (int j = 0; j < 26; j++)
		{

			m_ballPosition.push_back(CircleShape(2.5f));
			m_ballPosition.back().setPosition(Vector2f(60.f + s, 55.f + k));


			m_ballPosition.back().setFillColor(Color(255, 255, 255));

			if ((i > 7 && i < 19) && (j > 5 && j < 20))
			{
				m_ballPosition.pop_back();
			}

			if ((i > 7 && i < 19) && (j < 5 || j > 20))
			{
				m_ballPosition.pop_back();
			}

			for (int k = 0; k < m_wallPosition.size(); k++)
			{
				if (m_wallPosition[k].getGlobalBounds().intersects(m_ballPosition.back().getGlobalBounds()))
				{
					m_ballPosition.pop_back();
				}
			}

			s += 20.5;

		}

		k += 20.5;
	}

}


void pushRect(int row, int col, float x_cor, float y_cor)
{
	RectangleShape wall; //row*col
	m_wallPosition.push_back(wall);
	m_wallPosition.back().setSize(Vector2f(20.5 * col, 20.5 * row));
	m_wallPosition.back().setPosition(Vector2f(x_cor , y_cor ));
	m_wallPosition.back().setFillColor(Color::Transparent);
}
void addWallsToVector()
{

	//Drawing the horizontal rectangles for walls
	//Left
	pushRect(2, 3, 83.5, 77);
	pushRect(2, 4, 182.5, 77);
	pushRect(1, 3, 83.5, 159);
	pushRect(1, 3, 246.5, 158);
	pushRect(1, 3, 205.5, 220);
	pushRect(1, 3, 244.5, 405);
	pushRect(1, 3, 83.5, 466);
	pushRect(1, 4, 184.2, 466);
	pushRect(1, 2, 43.5, 529);
	pushRect(1, 3, 246.5, 529);
	pushRect(1, 9, 84.5, 589);
	//Right
	pushRect(2, 4, 368.5, 77);
	pushRect(2, 3, 493.5, 77);
	pushRect(1, 3, 492.5, 159);
	pushRect(1, 4, 303.5, 158);
	pushRect(1, 3, 368.5, 220);
	pushRect(1, 4, 306.5, 405);
	pushRect(1, 3, 492.5, 466);
	pushRect(1, 4, 367.5, 466);
	pushRect(1, 2, 551.5, 529);
	pushRect(1, 4, 306.5, 529);
	pushRect(1, 9, 370.5, 589);

	//Drawing the vertical rectangles for the walls
	//Left
	pushRect(7, 1, 182.5, 159);
	pushRect(4, 1, 182.5, 344);
	pushRect(3, 1, 182.5, 529);
	pushRect(3, 1, 122.5, 487);
	//Center
	pushRect(4, 1, 305.5, 35);
	pushRect(3, 1, 305.5, 178);
	pushRect(3, 1, 305.5, 425);
	pushRect(3, 1, 305.5, 547);
	//Right
	pushRect(7, 1, 429.5, 159);
	pushRect(4, 1, 429.5, 344);
	pushRect(3, 1, 429.5, 529);
	pushRect(3, 1, 491.5, 487);

	//Outer wall boundaries
	//Up
	pushRect(1, 28, 25.5, 21);
	//Down
	pushRect(1, 28, 27.5, 646);
	//Left
	pushRect(8, 1, 21.5, 45);
	pushRect(10, 1, 21.5, 430);
	//Right
	pushRect(8, 1, 593.5, 45);
	pushRect(10, 1, 593.5, 430);
	//Tunnel
	pushRect(4, 5, 37, 218);
	pushRect(4, 5, 37, 344);
	pushRect(5, 6, 493.5, 218);
	pushRect(5, 6, 493.5, 343);

}



Keyboard::Key getRandomDirection() {

	 int direction = rand()%4;


	switch (direction) {
	case 0:
		myLeft++;
		return Keyboard::Left;

	case 1:
		myRight++;
		return Keyboard::Right;

	case 2:
		myUp++;
		return Keyboard::Up;

	case 3:
		myDown++;
		return Keyboard::Down;
	}
}