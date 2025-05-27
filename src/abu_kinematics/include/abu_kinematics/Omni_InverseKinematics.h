#pragma once

#include <cmath>
#include <vector>
#include <algorithm> // for std::max_element

class InverseKinematic
{
private:
    double r;
    double R;

public:

    InverseKinematic(double wheel_radius, double robot_radius)
        : r(wheel_radius), R(robot_radius) {}

    // w1 ล้อหน้า w2 หลังขวา w3 หลังซ้าย
    std::vector<double> solve(double Vx, double Vy, double W)
    {
       
        // 1. Calculate wheel speeds in rad/s
        std::vector<double> wheelSpeed;
        float w1 = (-2*Vx - R * W ) / r;
        float w2 = (Vx + sqrt(3)/2 * Vy - R * W ) / r;
        float w3 = (Vx - sqrt(3)/2 * Vy - R * W ) / r;

        wheelSpeed.push_back(w1);
        wheelSpeed.push_back(w2);
        wheelSpeed.push_back(w3);

        double maxSpeed = 0.0;
        for (double speed : wheelSpeed)
        {
            if (fabs(speed) > maxSpeed)
            {
                maxSpeed = fabs(speed); // Find the maximum speed
            }
        }
        
        // 3. If any wheel exceeds maxMotorSpeed (27 rad/s), scale ALL speeds down
        if (maxSpeed > 27.0)
        {
            double scaleFactor = 27.0 / maxSpeed;
            for (double &speed : wheelSpeed)
            {
                speed *= scaleFactor; // Correct scaling
            }
        }

        return wheelSpeed;
    }
};