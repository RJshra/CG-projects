#include <cmath>
#include "star.h"

Star::Star(const glm::vec2& position, float rotation, float radius, float aspect)
	: _position(position), _rotation(rotation), _radius(radius) {
	// TODO: assemble the vertex data of the star
	// write your code here
	// -------------------------------------
	float x = _position.x;
	float y = _position.y;
	float PI = 3.1415926;
	float short_radius = radius / sin(54 * (PI / 180)) * sin(18 * (PI / 180));
	float pre_bias_x = short_radius * sin(36 * (PI / 180) - rotation);
	float pre_bias_y = short_radius * cos(36 * (PI / 180) - rotation);
	for (auto i = 0; i < 5; i++) {
		_vertices.push_back({ x,y });

		float long_x = sin(-rotation) * radius;
		float long_y = radius * cos(-rotation);
		float tem_x = long_x * cos(i * 72 * (PI / 180)) - long_y * sin(i * 72 * (PI / 180));
		float tem_y = long_x * sin(i * 72 * (PI / 180)) + long_y * cos(i * 72 * (PI / 180));
		_vertices.push_back({ x + tem_x / aspect,y + tem_y });
		_vertices.push_back({ x + pre_bias_x / aspect,pre_bias_y + y });


		_vertices.push_back({ x,y });
		_vertices.push_back({ x + tem_x / aspect,y + tem_y });
		tem_x = pre_bias_x;
		tem_y = pre_bias_y;
		pre_bias_x = tem_x * cos(72 * (PI / 180)) - tem_y * sin(72 * (PI / 180));
		pre_bias_y = tem_x * sin(72 * (PI / 180)) + tem_y * cos(72 * (PI / 180));
		_vertices.push_back({ x + pre_bias_x / aspect,pre_bias_y + y });
	}
	
	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * _vertices.size(), _vertices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
}

Star::Star(Star&& rhs) noexcept
	: _position(rhs._position), _rotation(rhs._rotation), _radius(rhs._radius),
	_vao(rhs._vao), _vbo(rhs._vbo) {
	rhs._vao = 0;
	rhs._vbo = 0;
}

Star::~Star() {
	if (_vbo) {
		glDeleteVertexArrays(1, &_vbo);
		_vbo = 0;
	}

	if (_vao) {
		glDeleteVertexArrays(1, &_vao);
		_vao = 0;
	}
}

void Star::draw() const {
	glBindVertexArray(_vao);
	glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(_vertices.size()));
}