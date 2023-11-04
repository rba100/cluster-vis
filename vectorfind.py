import numpy as np

def find_angle_between_vectors(v1, v2):
    # Assuming v1 and v2 are normalized, their dot product is the cosine of the angle between them
    cos_theta = np.dot(v1, v2)
    # Numerical errors might slightly push the cosine out of the valid range [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)

def create_rotation_matrix(theta):
    # Create a 2D rotation matrix that rotates by angle -theta
    return np.array([[np.cos(-theta), -np.sin(-theta)],
                     [np.sin(-theta), np.cos(-theta)]])

def create_transformation_matrix(v1, basis2):
    # Stack the basis vectors to form a transformation matrix from 2D to high-dimensional space
    return np.column_stack((v1, basis2))

# Normalized high-dimensional vectors
v1 = np.array([1, 0, 0, 0, 0], dtype=float)
v2 = np.array([0, 0, 0, 1, 0], dtype=float)

# Compute the angle between v1 and v2
theta = find_angle_between_vectors(v1, v2)

# Create a 2D rotation matrix to rotate by -theta
rotation_matrix = create_rotation_matrix(theta)

# Find the orthogonal basis for v2 with respect to v1
basis2 = v2 - np.dot(v2, v1) * v1
basis2 /= np.linalg.norm(basis2)

# Create a transformation matrix from the 2D plane spanned by v1 and basis2 back to high-dimensional space
transformation_matrix = create_transformation_matrix(v1, basis2)

# Rotate v1 in the 2D plane by -theta to get point_2d
point_2d = rotation_matrix @ np.array([1, 0])

# Map the rotated 2D point back to high-dimensional space
point_nd = transformation_matrix @ point_2d

print("Angle between v1 and v2 (in radians):", theta)
print("Rotated 2D point:", point_2d)
print("Rotated point in high-dimensional space:", point_nd)
