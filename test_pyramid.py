import matplotlib.pyplot as plt
from scene_pyramid import pyramid_edges, pyramid_vertices

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for start, end in pyramid_edges:
    x = [pyramid_vertices[start][0], pyramid_vertices[end][0]]
    y = [pyramid_vertices[start][1], pyramid_vertices[end][1]]
    z = [pyramid_vertices[start][2], pyramid_vertices[end][2]]

    ax.plot(x, y, z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("3D Pyramid")

ax.set_xlim(1, 5)
ax.set_ylim(-2, 3)
ax.set_zlim(-2, 2)

plt.show()