import matplotlib.pyplot as plt
from scene_cube import edges, vertices

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for start, end in edges:
    x = [vertices[start][0], vertices[end][0]]
    y = [vertices[start][1], vertices[end][1]]
    z = [vertices[start][2], vertices[end][2]]

    ax.plot(x, y, z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("3D Cube")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.show()
