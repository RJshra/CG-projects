#include <iostream>
#include "hello_triangle.h"
#include<fstream>
#include<string>
#include<Eigen/Dense>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include<cmath>
//vertex array and edge array
const char* filepath = "../../media/obj/my-MassSpring.obj";
double damping = 0.5;


const std::string earthTextureRelPath = "../../media/texture/miscellaneous/earthmap.jpg";
HelloTriangle::HelloTriangle(const Options& options): Application(options) {
	load_data(filepath);
	Masses[0].isFixed = Masses[90].isFixed = true;
	flag = false;
	

	glGenVertexArrays(1, &_vao);
	// create a vertex buffer object
	glGenBuffers(1, &_vbo);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Vertex) * Vertexes.size(), Vertexes.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Lines));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexLines));
	glEnableVertexAttribArray(1);

	
	// create shader
	const char* vsCode =
		"#version 330 core\n"
		"layout(location = 0) in vec3 aPos;\n"
		"uniform mat4 projection;\n"
		"uniform mat4 view;\n"
		"uniform mat4 model;\n"
		"void main() {\n"
		"	gl_Position =  projection * view *model*vec4(aPos, 1.0f);\n"
		
		"}\n";

	const char* fsCode =
		"#version 330 core\n"
		"out vec4 outColor;\n"
		"void main() {\n"
		"	outColor = vec4(1.0f);\n"
		"}\n";

	_shader.reset(new GLSLProgram);
	_shader->attachVertexShader(vsCode);
	_shader->attachFragmentShader(fsCode);
	_shader->link();

	const char* TvsCode =
		"#version 330 core\n"
		"layout(location = 0) in vec3 aPosition;\n"
		//"layout(location = 1) in vec3 aNormal;\n"
		"layout(location = 1) in vec2 aTexCoord;\n"
		"out vec2 fTexCoord;\n"
		"uniform mat4 projection;\n"
		"uniform mat4 view;\n"
		"uniform mat4 model;\n"

		"void main() {\n"
		"	fTexCoord = aTexCoord;\n"
		"	gl_Position = projection * view * model * vec4(aPosition, 1.0f);\n"
		"}\n";

	const char* TfsCode =
		"#version 330 core\n"
		"in vec2 fTexCoord;\n"
		"out vec4 color;\n"
		"uniform sampler2D mapKd;\n"
		"void main() {\n"
		"	color = texture(mapKd, fTexCoord);\n"
		"}\n";

	_TexureShader.reset(new GLSLProgram);
	_TexureShader->attachVertexShader(TvsCode);
	_TexureShader->attachFragmentShader(TfsCode);
	_TexureShader->link();
	// init camera
	_camera.reset(new PerspectiveCamera(
		glm::radians(45.0f),
		1.0f * _windowWidth / _windowHeight,
		0.1f, 10000.0f));

	_camera->transform.position = glm::vec3(8.5f,20.0f, 50.0f);
	_camera->transform.rotation = glm::angleAxis(-glm::radians(20.0f), _camera->transform.getRight());
	model = glm::mat4(1.0f);

	std::shared_ptr<Texture2D> earthTexture =
		std::make_shared<ImageTexture2D>(getAssetFullPath(earthTextureRelPath));

	mapKd = earthTexture;
	// init imgui
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(_window, true);
	ImGui_ImplOpenGL3_Init();
}

HelloTriangle::~HelloTriangle() {
	if (_vbo != 0) {
		glDeleteBuffers(1, &_vbo);
		_vbo = 0;
	}

	if (_vao != 0) {
		glDeleteVertexArrays(1, &_vao);
		_vao = 0;
	}
}

//(M - h df/dv -h^2 df/dx) dx = h( f + h df/dx v)
void HelloTriangle::constructFunction() {
	printf("1");
	for (int i = 0; i < Spring.size(); i++) {
		// x = |x1-x2|-l
		float x = glm::distance(Masses[Spring[i].index2].position, Masses[Spring[i].index1].position) - Spring[i].freeLength;
		//index1->index2
		glm::vec3 direction1 = glm::normalize(Masses[Spring[i].index2].position - Masses[Spring[i].index1].position);
		//index2->index1
		glm::vec3 direction2 = -direction1;

		//v2-v1
		glm::vec3 deltaV = Masses[Spring[i].index2].v- Masses[Spring[i].index1].v;
		Masses[Spring[i].index1].force += Spring[i].stiffness*direction1 * x 
			- direction1*Spring[i].damping*glm::dot(direction1,deltaV);

		Masses[Spring[i].index2].force += Spring[i].stiffness * direction2 * x 
			+ direction1 * Spring[i].damping * glm::dot(direction1, deltaV);
		
	}
	
	int n = Masses.size();
	Eigen::MatrixXf A=Eigen::MatrixXf::Zero(3*n, 3*n);
	Eigen::VectorXf b=Eigen::VectorXf::Zero(3*n);
	Eigen::VectorXf massV = Eigen::VectorXf::Zero(3 * n);
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 3; j++) {
			//A is the M matrix and dialog
			A(i * 3 + j, i * 3 + j) = Masses[i].m;
		}
		//massV is all masses' Velocity with 3 directions
		massV(i*3) = Masses[i].v.x;
		massV(i * 3+1) = Masses[i].v.y;
		massV(i * 3+2) = Masses[i].v.z;
		//b is the force
		b(i * 3) = Masses[i].force.x;
		b(i * 3+1) = Masses[i].force.y-Masses[i].g * Masses[i].m;
		b(i * 3+2) = Masses[i].force.z-windForce;
		Masses[i].force = glm::vec3(0.0f);
	}

	Eigen::MatrixXf f2v = Eigen::MatrixXf::Zero(3 * n, 3 * n);	//	df/dv
	Eigen::MatrixXf f2x = Eigen::MatrixXf::Zero(3 * n, 3 * n);  //  df/dx 

	// f = mg + k||x-l||*x/||x|| - damping*v.*norm(x)*norm(x)
	// df/dv = -damping * norm(x)*norm(x), 3x3 submatrix
	for (int i = 0; i < Spring.size(); i++) {
		int index1 = Spring[i].index1, index2 = Spring[i].index2;
		// index2-index1
		glm::vec3 deltaX = Masses[index2].position - Masses[index1].position;
		deltaX = glm::normalize(deltaX);
		for (int j = 0; j < 3; j++) {
			f2v(index1 * 3 + j, index1 * 3 + j) += Spring[i].damping *deltaX[j] * deltaX[j];
			f2v(index2 * 3 + j, index2 * 3 + j) += Spring[i].damping * deltaX[j] * deltaX[j];
		}
		f2v(index1 * 3 + 1, index1 * 3) += Spring[i].damping * deltaX.x * deltaX.y;
		f2v(index1 * 3, index1 * 3 + 1) += Spring[i].damping * deltaX.x*deltaX.y;
		f2v(index1 * 3 + 2, index1 * 3) += Spring[i].damping * deltaX.x * deltaX.z;
		f2v(index1 * 3, index1 * 3 + 2) +=Spring[i].damping * deltaX.x * deltaX.z;
		f2v(index1 * 3 + 1, index1 * 3 + 2) += Spring[i].damping * deltaX.y * deltaX.z;
		f2v(index1 * 3 + 2, index1 * 3 + 1) += Spring[i].damping * deltaX.y * deltaX.z ;

		f2v(index2 * 3 + 1, index2 * 3) += Spring[i].damping * deltaX.x * deltaX.y;
		f2v(index2 * 3, index2 * 3 + 1) += Spring[i].damping * deltaX.x * deltaX.y;
		f2v(index2 * 3 + 2, index2 * 3) += Spring[i].damping * deltaX.x * deltaX.z;
		f2v(index2 * 3, index2 * 3 + 2) += Spring[i].damping * deltaX.x * deltaX.z;
		f2v(index2 * 3 + 1, index2 * 3 + 2) += Spring[i].damping * deltaX.y * deltaX.z;
		f2v(index2 * 3 + 2, index2 * 3 + 1) += Spring[i].damping * deltaX.y * deltaX.z;
	}

	// df/dx i = -kI +klI / |xi - xj| -kl norm(xi-xj)*norm (xi-xj)T /|xi - xj|
	//    damping=    ...
	for (int i = 0; i < Spring.size(); i++) {
		int index1 = Spring[i].index1;
		int index2 = Spring[i].index2;
		// index2-index1
		glm::vec3 deltaX = Masses[index2].position - Masses[index1].position;
		if(glm::distance(deltaX,glm::vec3(0.0f)))
			deltaX = glm::normalize(deltaX);
		Eigen::Vector3f normalizeX(deltaX.x,deltaX.y,deltaX.z);
		Eigen::Matrix3f X = normalizeX * normalizeX.transpose();
		// -kI+klI / |xi - xj|
		Eigen::Matrix3f part1 = Spring[i].stiffness*Eigen::Matrix3f::Identity(3, 3) * (
			Spring[i].freeLength/(glm::distance(Masses[index2].position, Masses[index1].position))-1
			);
		//-kl norm(xi-xj)*norm (xi-xj)T /|xi - xj|
		Eigen::Matrix3f part2 = -Spring[i].stiffness * Spring[i].freeLength * X
				/(glm::distance(Masses[index2].position, Masses[index1].position));
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				f2x(index1 * 3 + j, index1 * 3 + k) += part1(j, k) + part2(j, k);
				f2x(index2 * 3 + j, index2 * 3 + k) += part1(j, k) + part2(j, k);
			}
		}
		//damping: -damping*(normal(xj-xi)T*normal(vj-vi)*I+normal(xj-xi)*normal(vj-vi)T)*
		//(normal(xj-xi)*normal(xj-xi)T-I)/|xj-xi|
		glm::vec3 deltaV = Masses[index2].v - Masses[index1].v;
		if(glm::distance(deltaV,glm::vec3(0.0f)))
			deltaV = glm::normalize(deltaV);
		Eigen::Vector3f V(deltaV.x,deltaV.y,deltaV.z);
		//part1: normal(X)T*normal(V)*I+normal(X)*normal(V)T
		Eigen::Matrix3f damp1 = normalizeX.dot(V) * Eigen::Matrix3f::Identity(3, 3)+normalizeX*V.transpose();
		
		//part2: (normal(xj-xi)*normal(xj-xi)T-I)/|xj-xi|
		Eigen::Matrix3f damp2 = (normalizeX * normalizeX.transpose() - Eigen::Matrix3f::Identity(3, 3)) 
			/ glm::distance(Masses[index2].position, Masses[index1].position);
		Eigen::Matrix3f damp = -Spring[i].damping * damp1 * damp2;
		
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				f2x(3*index1 + j, 3*index1 + k) +=damp(j,k);
				f2x(3*index2 + j, 3*index2 + k) += damp(j, k);
			}
		}
	}

	//(M - h df/dv -h^2 df/dx) dx = h( f + h df/dx v)
	A = A - deltaT * f2v - deltaT * deltaT * f2x;
	b = (b + deltaT * f2x*massV) * deltaT;


	//Conjugate Gradient
	//deltaV is the vector to be solve
	Eigen::VectorXf deltaV = Eigen::VectorXf::Zero(3 * n);
	Eigen::VectorXf r = b;
	Eigen::VectorXf p = r;
	
	double accuracy = 0.001f;
	
	while (1) {
		double a = r.transpose() * r;
		double num = p.transpose() * A * p;
		a /= num;
		
		deltaV += a * p;
		Eigen::VectorXf temp = r;
		r -= a * A * p;
		if (r.norm() < accuracy)
			break;
		double B = r.transpose() * r;
		B /= temp.transpose() * temp;
		p = r + B * p;
	}

	//update v,x
	for (int i = 0; i < n; i++) {
		if (Masses[i].isFixed)
			continue;
		for (int j = 0; j < 3; j++) {
			Masses[i].v[j] += deltaV(i * 3 + j);
			Masses[i].position[j] += deltaT * Masses[i].v[j];
		}
	}
	for (int i = 0; i < Spring.size(); i++) {
		if(Masses[Spring[i].index1].isFixed==false)
			Vertexes[2 * i].Lines = Masses[Spring[i].index1].position;
		if(Masses[Spring[i].index2].isFixed==false)
			Vertexes[2 * i+1].Lines = Masses[Spring[i].index2].position;
	}
}

void HelloTriangle::handleInput() {
	static int firstMouse = 1;
	
	if (_input.keyboard.keyStates[GLFW_KEY_ESCAPE] != GLFW_RELEASE) {
		glfwSetWindowShouldClose(_window, true);
		return ;
	}
	const float _cameraMoveSpeed = 10.0f;
	const float _cameraRotateSpeed = 0.05f;

	bool t = false;
	if (_input.keyboard.keyStates[GLFW_KEY_SPACE] != GLFW_RELEASE) {
		t = true;
	}
		
	if (t) {
		flag = !flag;
	}

	if (_input.keyboard.keyStates[GLFW_KEY_W] != GLFW_RELEASE) {
		_camera->transform.position += _camera->transform.getFront() * _cameraMoveSpeed * _deltaTime;
	}

	if (_input.keyboard.keyStates[GLFW_KEY_A] != GLFW_RELEASE) {
		_camera->transform.position -= _camera->transform.getRight() * _cameraMoveSpeed * _deltaTime;
	}

	if (_input.keyboard.keyStates[GLFW_KEY_S] != GLFW_RELEASE) {
		_camera->transform.position -= _camera->transform.getFront() * _cameraMoveSpeed * _deltaTime;
	}

	if (_input.keyboard.keyStates[GLFW_KEY_D] != GLFW_RELEASE) {
		_camera->transform.position += _camera->transform.getRight() * _cameraMoveSpeed * _deltaTime;
	}

	/*if (_input.mouse.move.xNow != _input.mouse.move.xOld) {
		if (!firstMouse) {
			float mouse_movement_in_x_direction = (_input.mouse.move.xNow - _input.mouse.move.xOld) * _cameraRotateSpeed;
			_input.mouse.move.xOld = _input.mouse.move.xNow;
			float angle = glm::radians(mouse_movement_in_x_direction);
			const glm::vec3 axis = glm::vec3(0.0f,1.0f,0.0f);
			_camera->transform.rotation = glm::angleAxis(angle, axis) * _camera->transform.rotation;
		}
	}

	if (_input.mouse.move.yNow != _input.mouse.move.yOld) {
		if (!firstMouse) {
			float mouse_movement_in_y_direction = (_input.mouse.move.yNow - _input.mouse.move.yOld) * _cameraRotateSpeed;
			_input.mouse.move.yOld = _input.mouse.move.yNow;
			float angle = glm::radians(mouse_movement_in_y_direction);
			_camera->transform.rotation = glm::angleAxis(angle, _camera->transform.getRight()) * _camera->transform.rotation;
		}
		else
			firstMouse = 0;
	}*/

	_input.forwardState();
	
	if(flag)
		constructFunction();

	
}

void HelloTriangle::renderFrame() {
	showFpsInWindowTitle();
	glClearColor(_clearColor.r, _clearColor.g, _clearColor.b, _clearColor.a);
	glClear(GL_COLOR_BUFFER_BIT);

	glm::mat4 projection = _camera->getProjectionMatrix();
	glm::mat4 view = _camera->getViewMatrix();
	
	switch (_renderMode) {
	case RenderMode::Line:
		
		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Vertex) * Vertexes.size(), Vertexes.data(), GL_STATIC_DRAW);
		_shader->use();
		_shader->setUniformMat4("model",model);
		_shader->setUniformMat4("view", view);
		_shader->setUniformMat4("projection", projection);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBindVertexArray(_vao);
		glEnableVertexAttribArray(0);
		glPointSize(3);
		glDrawArrays(GL_LINES, 0, Vertexes.size());
		break;
	case RenderMode::Texture:
		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Vertex) * Vertexes.size(), Vertexes.data(), GL_STATIC_DRAW);
		_TexureShader->use();
		_TexureShader->setUniformMat4("model", model);
		_TexureShader->setUniformMat4("view", view);
		_TexureShader->setUniformMat4("projection", projection);
		mapKd->bind();
		glBindVertexArray(_vao);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glDrawArrays(GL_LINES, 0, Vertexes.size());
		mapKd->unbind();
		glBindVertexArray(0);
		break;
	}
	

	// draw ui elements
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const auto flags =
		ImGuiWindowFlags_AlwaysAutoResize |
		ImGuiWindowFlags_NoSavedSettings;

	if (!ImGui::Begin("Control Panel", nullptr, flags)) {
		ImGui::End();
	}
	else {
		ImGui::Text("Render Mode");
		ImGui::Separator();
		ImGui::RadioButton("Draw Lines", (int*)&_renderMode, (int)(RenderMode::Line));
		ImGui::NewLine();

		ImGui::RadioButton("Draw texture", (int*)&_renderMode, (int)(RenderMode::Texture));
		ImGui::NewLine();
		ImGui::SliderFloat("add wind force##3", &windForce, 0.0f, 15.0f);
		ImGui::NewLine();
		ImGui::SliderFloat("change the stiffness##4", &stiffness, 1000.0f, 10000.0f);
		ImGui::NewLine();
		ImGui::SliderFloat("change time steps##3", &deltaT, 0.01f, 0.1f);
		ImGui::NewLine();
		ImGui::End();
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void HelloTriangle::load_data(const char* filepath) {
	std::ifstream file;
	file.open(filepath);
	if (file.fail()) {
		file.close();
		return;
	}

	if (file.peek() != EOF) {
		while (!file.eof()) {
			std::string data;
			std::getline(file, data);
			if (data.c_str()[0] == 'v' && data.c_str()[1] != 't') {
				char* temp = strtok((char*)data.c_str(), " ");
				std::vector<double> pos;
				while (temp != NULL) {
					if (temp[0] != 'v')
						pos.push_back(atof(temp));
					temp = strtok(NULL, " ");
				}
				Mass mass;
				mass.g = 9.8;
				mass.isFixed = false;
				mass.m = 1.0;
				glm::vec3 v(0.0f);
				mass.v = v;
				mass.force = v;
				mass.position = glm::vec3(10.0 * pos[0], 5*pos[1],10* pos[2]);
				Masses.push_back(mass);

				
			}
			else if (data.c_str()[0] == 'e') {
				char* temp = strtok((char*)data.c_str(), " ");
				std::vector<int> index;
				while (temp != NULL) {
					if (temp[0] != 'e')
						index.push_back(atoi(temp));
					temp = strtok(NULL, "/");
				}
				Edge edge;
				edge.damping = damping;
				edge.stiffness = stiffness;
				edge.index1 = index[0];
				edge.index2 = index[1];
				edge.freeLength = glm::distance(Masses[edge.index1].position, Masses[edge.index2].position);
				Spring.push_back(edge);
				Vertex v1,v2;
				v1.Lines = Masses[edge.index1].position;
				v1.TexLines = texture_pos[edge.index1];
				v2.Lines = Masses[edge.index2].position;
				v2.TexLines = texture_pos[edge.index2];
				Vertexes.push_back(v1);
				Vertexes.push_back(v2);
				
			}
			else if (data.c_str()[0] == 'v' && data.c_str()[1] == 't') {
				char* temp = strtok((char*)data.c_str(), " ");
				std::vector<double> pos;
				while (temp != NULL) {
					if (temp[0] != 'v')
						pos.push_back(atof(temp));
					temp = strtok(NULL, " ");
				}
				glm::vec2 tex_pos(pos[0], pos[1]);
				texture_pos.push_back(tex_pos);
			}
			else
				continue;
		}
	}
	file.close();
}