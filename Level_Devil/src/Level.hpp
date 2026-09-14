#ifndef LEVEL_HPP
#define LEVEL_HPP
#include <SFML/Graphics.hpp>

class Blocks{
private:
    sf::Vector2f size;
    sf::Vector2f position;
    sf::Color fill;
public:
    sf::RectangleShape shape;
    Blocks(sf::Vector2f size, sf::Vector2f position, sf::Color fill){
        this->size = size;
        this->position = position;
        this->fill = fill;
    }
    void CreateRectangle(){
        shape.setFillColor(fill);
        shape.setSize(size);
        shape.setPosition(position);
    }

    void move(sf::Vector2f new_pose){
        
    }

    void resize(){

    }

    void draw(sf::RenderWindow& window){
        window.draw(shape);
    }
};

class Portals{
private:
    sf::Vector2f size;
    sf::Vector2f position;
    sf::Color fill;
    sf::Color outline;
    float thickness;

public:
    sf::RectangleShape shape;
    Portals(sf::Vector2f size, sf::Vector2f position, sf::Color fill, sf::Color outline, float thickness){
        this->size = size;
        this->position = position;
        this->fill = fill;
        this->outline = outline;
        this->thickness = thickness;
    }
    void CreatePortal(){
        shape.setSize(size);
        shape.setFillColor(fill);
        shape.setOutlineThickness(thickness);
        shape.setOutlineColor(outline);
        shape.setPosition(position);
    }
    void draw(sf::RenderWindow& window){
        window.draw(shape);
    }
};

class Spikes{
private:
    float height;
    float width;
    int num_spikes;
    sf::Vector2f startPos{0.0, 0.0};
    sf::Vector2f point0{width/2.f, 0.f};
    sf::Vector2f point1{0.f, height};
    sf::Vector2f point2{width, height};
    sf::Color fill;

public:
    sf::ConvexShape triangle;
    Spikes(float width, float height, sf::Color fill, int num_spikes, sf::Vector2f startPos){
        this->height = height;
        this->width = width;
        this->fill = fill;
        this->num_spikes = num_spikes;
        this->startPos = startPos;
    }
    void CreateSpikes(){
        triangle.setFillColor(fill);
        triangle.setPoint(0,point0);
        triangle.setPoint(1,point1);
        triangle.setPoint(2,point2);
    }
    void DrawMultipleSpikes(sf::RenderWindow& window){
        for(int i = 0; i < num_spikes; i++){
            triangle.setPosition({startPos.x + (i * width), startPos.y});
            window.draw(triangle);
        }
    }

};

class Teleports{
private:
    sf::Vector2f size;
    sf::Vector2f position;
    sf::Color fill;
    sf::Color outline;

public:
    sf::RectangleShape shape;
    Teleports(sf::Vector2f size, sf::Vector2f position, sf::Color fill, sf::Color outline){
        this->size = size;
        this->position = position;
        this->fill = fill;
        this-> outline = outline;
    }

    void draw(sf::RenderWindow& window){
        window.draw(shape);
    }
};

#endif