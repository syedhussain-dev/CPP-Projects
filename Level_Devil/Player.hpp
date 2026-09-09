#ifndef PLAYER_HPP
#define PLAYER_HPP
#include <windows.h>
#include <cmath>
const double g = 9.81;

struct Position{
    double x = 0.0;
    double y = 0.0;
};

struct Velocity{
    double x = 0.0;
    double y = 0.0;
};

struct Acceleration{
    double x = 0.0;
    double y = g;
};

struct Size{
    int x = 20;
    int y = 20;
};

struct Normal{
    int x = 0;
    int y = 0;
};

class Player{
private:

    Size playersize;
    double mass{0.5};
    bool isAlive{false};
    bool isJumping{false};
    Position playerpose;
    Velocity playervel;
    Acceleration playeracc;

public:

    void update(double deltaTime);

};

void Player::update(double deltaTime){
    bool CurrentWPress = GetAsyncKeyState('W');
    bool CurrentAPress = GetAsyncKeyState('A');
    bool CurrentDPress = GetAsyncKeyState('D');
    if(CurrentWPress && !isJumping){
        isJumping = true;
        playervel.y = -3;
    }
    if(CurrentAPress && CurrentDPress){
        playervel.x = 0;
    }
    else if(CurrentAPress){
        playervel.x = -3;
    }
    else if(CurrentDPress){
        playervel.x = 3;
    }
    playervel.x += playeracc.x * deltaTime;
    playervel.y += playeracc.y * deltaTime;
    playerpose.x += playervel.x * deltaTime;
    playerpose.y += (playervel.y * deltaTime) + (0.5 * playeracc.y * pow(deltaTime, 2));

}

class Collision{
private:
    double restitution;
    double mass;

public:

    Collision(double mass, double restitution){
        this->restitution = restitution;
        this->mass = mass;
    }

    double ImpulseCalculation(double before_vel, Normal normal);

};

double Collision::ImpulseCalculation(double before_vel, Normal normal){
    double delta_vel = before_vel;
    if(restitution != 0.0){
        double after_vel = restitution * before_vel;
        delta_vel = after_vel - before_vel;
    }
    double impulse = 0.0;
    if (normal.x == 0){
        impulse = mass * delta_vel * normal.y;
    }
    else{
        impulse = mass * delta_vel * normal.x;
    }
    return impulse;
}
#endif