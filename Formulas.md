# Formulas Used 
In this code, the following formulas are used to calculate human motion:

# Distance Calculation
1. Euclidean Distance: calculate_distance function uses NumPy's linalg.norm to calculate the distance between two points (e.g., shoulder positions).

d = √((x2 - x1)^2 + (y2 - y1)^2)

# Angle Calculation
1. Dot Product Formula: calculate_angle function uses the dot product formula to calculate the angle between three points (e.g., shoulder, elbow, and wrist).

cos(θ) = (a · b) / (|a| |b|)

where:
- a and b are vectors
- θ is the angle between the vectors
- · denotes the dot product
- |a| and |b| are the magnitudes of the vectors

# Calculation
1. Speed Formula: Speed is calculated as the distance traveled divided by the time difference.

Speed = Distance / Time

These formulas are used to calculate various aspects of human motion, such as:

1. Distance traveled: By calculating the distance between consecutive shoulder positions.
2. Speed: By calculating the distance traveled divided by the time difference.
3. Elbow angle: By calculating the angle between the shoulder, elbow, and wrist positions.
