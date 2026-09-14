#include <SFML/Graphics.hpp>
#include "Player.hpp"
#include "Level.hpp"

int main()
{
    sf::RenderWindow window(sf::VideoMode({1920, 1080}), "My Version of Level Devil!");

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.display();
    }
}
