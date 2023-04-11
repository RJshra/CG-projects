#pragma once

#include <memory> 
#include <unordered_map>
#include<cstdbool>
#include "../base/application.h"
#include "../base/glsl_program.h"
#include "../base/camera.h"
#include "../base/texture2d.h"
enum class RenderMode {
	Line, Texture
};

typedef struct vertex {
	glm::vec3 Lines;
	glm::vec2 TexLines;
}Vertex;
typedef struct mass {
	glm::vec3 v;
	glm::vec3 position;
	glm::vec3 force;
	float m;
	float g;
	bool isFixed;

}Mass;

typedef struct edge {
	int index1, index2;
	float freeLength;
	float stiffness;
	float damping;
}Edge;

class HelloTriangle : public Application {
public:
	HelloTriangle(const Options& options);

	~HelloTriangle();

private:
	std::unique_ptr<PerspectiveCamera> _camera;
	std::shared_ptr<Texture2D> mapKd;
	glm::mat4 model;
	std::vector<Mass> Masses;
	std::vector<Edge> Spring;
	std::vector<glm::vec2> texture_pos;
	float stiffness = 1000.0f;
	float windForce = 0;
	float deltaT = 0.01f;
	bool flag;	
	std::vector<Vertex> Vertexes;	


	GLuint _vao = 0;

	GLuint _vbo = 0;


	void load_data(const char* filepath);

	void constructFunction();

	std::unique_ptr<GLSLProgram> _shader;

	std::unique_ptr<GLSLProgram> _TexureShader;

	virtual void handleInput();

	virtual void renderFrame();

	enum RenderMode _renderMode = RenderMode::Line;
};
